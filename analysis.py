import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import find_peaks

from config import (
    DATA_DIR, LAT_COL, LON_COL,
    MATCH_RADIUS,
    PEAK_HEIGHT, PEAK_DISTANCE, PEAK_PROMINENCE,
    FIXTURE_THRESHOLDS, DEFAULT_THRESHOLDS,
    CROSSWALK_RADIUS,
)


# ── 1. Single-file ingestion + peak detection ─────────────────────────────────

def _parse_cal(line: str) -> tuple[float, float, float]:
    a = float(re.search(r"a=([-\d.]+)", line).group(1))
    b = float(re.search(r"b=([-\d.]+)", line).group(1))
    c = float(re.search(r"c=([-\d.]+)", line).group(1))
    return a, b, c


def parse_metadata(filepath: str) -> dict:
    """Read the 7-line CSV header. Returns date, description, per-sensor cals."""
    meta = {
        "file":        os.path.basename(filepath),
        "date":        None,
        "description": None,
        "cals":        {},          # {sensor_index: (a, b, c)}
    }
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i > 7:
                break
            if "Description" in line:
                try:
                    meta["description"] = re.search(
                        r"Description:\s*(.+)", line
                    ).group(1).strip()
                except AttributeError:
                    pass
            if "Date" in line and meta["date"] is None:
                try:
                    meta["date"] = re.search(
                        r"Date:\s*(.+)", line
                    ).group(1).strip()
                except AttributeError:
                    pass
            m = re.search(r"LuxCal(\d+):", line)
            if m:
                meta["cals"][int(m.group(1))] = _parse_cal(line)
    return meta


def process_file(filepath: str) -> tuple[pd.DataFrame | None, dict]:
    """
    Load one CSV, apply per-sensor calibration, filter bad rows,
    detect lux peaks. Returns (peak_df, meta) or (None, meta) on failure.

    A 'peak' is a local maximum in lux_max that clears the thresholds in
    config — each peak represents one pole passage event.
    """
    meta = parse_metadata(filepath)

    try:
        df = pd.read_csv(filepath, skiprows=7)
    except Exception as e:
        print(f"  [WARN] Could not read {filepath}: {e}")
        return None, meta

    # ── calibrate each sensor ─────────────────────────────────────────────
    raw_lux_cols = [c for c in df.columns if re.match(r"^Lux\d+$", c)]
    cal_cols = []
    for col in raw_lux_cols:
        idx = int(col.replace("Lux", ""))
        if idx in meta["cals"]:
            a, b, c = meta["cals"][idx]
            cal_col = f"{col}_cal"
            df[cal_col] = a * df[col] ** 2 + b * df[col] + c
            cal_cols.append(cal_col)

    if not cal_cols:
        return None, meta

    df["lux_max"] = df[cal_cols].max(axis=1)   # brightest sensor per row

    # ── filter: valid GPS + moving ────────────────────────────────────────
    df = df[
        df[LAT_COL].notna()
        & (df[LAT_COL] != 0)
        & df[LON_COL].notna()
        & (df[LON_COL] != 0)
        & (df["Speed_mph"] > 2)
    ].reset_index(drop=True)

    if len(df) < 50:
        return None, meta

    # ── detect peaks ──────────────────────────────────────────────────────
    # Each peak = car is directly under (or very near) a light pole.
    # height     : minimum lux to count — filters ambient road surface noise
    # distance   : minimum rows between peaks — prevents double-counting
    # prominence : peak must rise clearly above local baseline
    peak_idxs, props = find_peaks(
        df["lux_max"],
        height=PEAK_HEIGHT,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
    )

    if len(peak_idxs) == 0:
        return None, meta

    peak_df = df.iloc[peak_idxs].copy()
    peak_df["peak_lux"]   = df["lux_max"].iloc[peak_idxs].values
    peak_df["prominence"] = props["prominences"]
    peak_df["source_file"] = meta["file"]

    parsed_date = pd.to_datetime(meta["date"], errors="coerce")
    peak_df["file_date"] = parsed_date
    peak_df["campaign"]  = (
        "July"
        if pd.notnull(parsed_date)
        and (parsed_date.month == 7
             or (parsed_date.month == 8 and parsed_date.day <= 3))
        else "August"
    )

    return peak_df, meta


def load_all_peaks(data_dir=None, verbose=True) -> pd.DataFrame:
    """
    Scan every CSV in data_dir, detect peaks in each, return combined DataFrame.
    One row = one pole-passage event (a lux peak).
    """
    if data_dir is None:
        data_dir = DATA_DIR

    patterns = [
        os.path.join(data_dir, "Confidential_City_of_sugar_land*.csv"),
        os.path.join(data_dir, "Confidential_City_of_Sugar_Land*.csv"),
        os.path.join(data_dir, "Confidential_City_of_sugarland*.csv"),
        os.path.join(data_dir, "Confidential_Richmond*.csv"),
    ]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p))
    all_files = list(set(all_files))   # deduplicate

    if verbose:
        print(f"📂 Processing {len(all_files)} CSV files...")

    peak_frames = []
    for fpath in sorted(all_files):
        peak_df, meta = process_file(fpath)
        if peak_df is not None and len(peak_df) > 0:
            peak_frames.append(peak_df)
            if verbose:
                print(f"  ✓ {meta['file']:60s}  {len(peak_df):>4} peaks")
        else:
            if verbose:
                print(f"  – {meta['file']:60s}  (skipped)")

    if not peak_frames:
        raise ValueError("No peaks detected across any file. Check DATA_DIR and thresholds.")

    all_peaks = pd.concat(peak_frames, ignore_index=True)
    if verbose:
        print(f"\n✅ Total peaks: {len(all_peaks):,} across {len(peak_frames)} files")
    return all_peaks


def match_peaks_to_poles(
    peaks_df: pd.DataFrame,
    poles_df: pd.DataFrame,
    radius_deg: float = None,
) -> pd.DataFrame:
    """
    For each peak, find the nearest known pole within radius_deg.
    Adds columns: pole_idx, pole_dist, matched, fixture, facility_id.
    Unmatched peaks (no pole nearby) are kept with matched=False.
    """
    if radius_deg is None:
        radius_deg = MATCH_RADIUS

    valid_poles = poles_df[["lat_", "long_"]].dropna()
    tree = cKDTree(valid_poles.values)

    peak_coords = peaks_df[[LAT_COL, LON_COL]].values
    dists, idxs = tree.query(peak_coords, distance_upper_bound=radius_deg)
    matched_mask = dists < radius_deg

    out = peaks_df.copy()
    out["pole_idx"]  = np.where(matched_mask, idxs, -1)
    out["pole_dist"] = np.where(matched_mask, dists, np.nan)
    out["matched"]   = matched_mask

    matched_rows = out[matched_mask]
    out.loc[matched_mask, "fixture"]     = poles_df.loc[
        matched_rows["pole_idx"].values, "FIXTUREWAT"
    ].values
    out.loc[matched_mask, "facility_id"] = poles_df.loc[
        matched_rows["pole_idx"].values, "FACILITYID"
    ].values

    return out


def aggregate_per_pole(
    matched_peaks: pd.DataFrame,
    poles_df: pd.DataFrame,
    gps_all: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Collapse all peaks for a pole into one summary row.

    Key columns in output:
      peak_lux_median  — typical brightness across all passes (primary health signal)
      peak_lux_max     — brightest single pass ever recorded
      peak_lux_latest  — most recent pass brightness (catches recent failures)
      pass_count       — how many times detected (survey coverage depth)
      surveyed         — True if vehicle drove within range at least once
      status           — 'failed' | 'atrisk' | 'healthy' | 'unsurveyed'
    """
    hits = matched_peaks[matched_peaks["matched"]].copy()
    hits = hits.sort_values("file_date")   # ensures 'last' = most recent

    pole_health = hits.groupby("pole_idx").agg(
        peak_lux_median = ("peak_lux", "median"),
        peak_lux_max    = ("peak_lux", "max"),
        peak_lux_latest = ("peak_lux", "last"),
        pass_count      = ("peak_lux", "count"),
        lat             = (LAT_COL,    "mean"),
        lon             = (LON_COL,    "mean"),
        fixture         = ("fixture",  "first"),
        facility_id     = ("facility_id", "first"),
        last_seen       = ("file_date", "max"),
        campaign        = ("campaign", "last"),
    ).reset_index()

    pole_health["surveyed"] = True
    pole_health = classify_health(pole_health)

    if gps_all is not None:
        surveyed_idxs = set(pole_health["pole_idx"].values)
        valid_poles   = poles_df[["lat_", "long_"]].dropna()
        gps_tree      = cKDTree(gps_all[["Latitude", "Longitude"]].values)
        dists_to_gps, _ = gps_tree.query(
            valid_poles.values, distance_upper_bound=MATCH_RADIUS
        )
        in_route = dists_to_gps < MATCH_RADIUS

        dark_idxs = [
            i for i in range(len(valid_poles))
            if in_route[i] and i not in surveyed_idxs
        ]
        if dark_idxs:
            dark_poles = poles_df.iloc[dark_idxs].copy().reset_index(drop=True)
            dark_df = pd.DataFrame({
                "pole_idx":        dark_idxs,
                "peak_lux_median": 0.0,
                "peak_lux_max":    0.0,
                "peak_lux_latest": 0.0,
                "pass_count":      0,
                "lat":             dark_poles["lat_"].values,
                "lon":             dark_poles["long_"].values,
                "fixture":         dark_poles["FIXTUREWAT"].values,
                "facility_id":     dark_poles["FACILITYID"].values,
                "last_seen":       pd.NaT,
                "campaign":        "unknown",
                "surveyed":        True,
                "status":          "failed",
            })
            pole_health = pd.concat([pole_health, dark_df], ignore_index=True)

    return pole_health


def classify_health(pole_health: pd.DataFrame) -> pd.DataFrame:
    """
    Assign status per pole using fixture-specific lux thresholds.
    Thresholds are peak lux.

      - 'failed' : no detectable spike — pole is dark or sensor missed it
      - 'atrisk' : weak spike — lamp is dim, aging, or partially blocked
      - 'healthy': clear bright spike — lamp is working normally
    """
    def _classify_row(row):
        lux = row["peak_lux_median"]
        thresholds = FIXTURE_THRESHOLDS.get(row["fixture"], DEFAULT_THRESHOLDS)
        if lux < thresholds["failed"]:
            return "failed"
        elif lux < thresholds["atrisk"]:
            return "atrisk"
        else:
            return "healthy"

    pole_health["status"] = pole_health.apply(_classify_row, axis=1)
    return pole_health

def flag_at_risk_crosswalks(
    crosswalks: pd.DataFrame,
    pole_health: pd.DataFrame,
    radius_deg: float = None,
) -> pd.DataFrame:
    if radius_deg is None:
        radius_deg = CROSSWALK_RADIUS

    bad = (
        pole_health[pole_health["status"].isin(["failed", "atrisk"])]
        [["lat", "lon", "peak_lux_median", "status"]]
        .dropna()
        .reset_index(drop=True)
    )

    cw = crosswalks.copy()
    cw_coords = cw[["lat_", "long_"]].dropna().values

    if len(bad) == 0 or len(cw_coords) == 0:
        cw["at_risk"]     = False
        cw["risk_score"]  = 0.0
        cw["risk_tier"]   = "safe"
        cw["nearest_pole_lux"] = np.nan
        return cw

    bad_tree = cKDTree(bad[["lat", "lon"]].values)

    dists, idxs = bad_tree.query(cw_coords, distance_upper_bound=radius_deg)
    in_range     = dists < radius_deg
    safe_idxs    = np.clip(idxs, 0, len(bad) - 1)

    # Convert degrees to approximate meters
    dist_m = dists * 111_000

    TYPE_WEIGHT = {
        "Uncontrolled":   1.0,
        "Midblock":       0.8,
        "Stop Controlled":0.5,
    }

    STATUS_WEIGHT = {"failed": 1.0, "atrisk": 0.5}

    scores = []
    for i, row in cw.iterrows():
        if not in_range[i]:
            scores.append(0.0)
            continue
        pole_idx    = safe_idxs[i]
        cw_type     = str(row.get("Type", "Stop Controlled"))
        pole_status = bad.iloc[pole_idx]["status"]
        distance    = dist_m[i]

        tw = TYPE_WEIGHT.get(cw_type, 0.5)
        sw = STATUS_WEIGHT.get(pole_status, 0.5)
        dw = max(0.0, 1.0 - (distance / 80))  # decay over 80m

        scores.append(round(tw * sw * dw, 3))

    cw["risk_score"] = scores
    cw["at_risk"]    = cw["risk_score"] > 0.0

    cw["risk_tier"] = pd.cut(
        cw["risk_score"],
        bins=[-0.001, 0.0, 0.35, 0.65, 1.0],
        labels=["safe", "low", "high-risk", "critical"]
    )

    cw["nearest_pole_lux"] = np.where(
        in_range,
        bad["peak_lux_median"].iloc[safe_idxs].values,
        np.nan,
    )
    return cw