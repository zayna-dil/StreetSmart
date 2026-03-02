import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

from config import STATUS_COLORS, CROSSWALK_RADIUS

# Template lives next to this file in visualization/
_TEMPLATE_PATH = Path(__file__).parent / "map_template.html"


# ── Data serialization ────────────────────────────────────────────────────────

def _build_poles_json(pole_health: pd.DataFrame) -> str:
    """
    Compact JSON array. Each pole element:
      [lat, lon, status, color, fixture,
       lux_median, lux_max, lux_pct, pass_count, facility_id]
    """
    healthy = pole_health[pole_health["status"] == "healthy"]
    fixture_baselines = (
        healthy.groupby("fixture")["peak_lux_median"].median().to_dict()
    )
    FALLBACK_BASELINE = 25.0

    records = []
    for _, row in pole_health.iterrows():
        status  = str(row.get("status", "unsurveyed"))
        color   = STATUS_COLORS.get(status, "#444466")
        fixture = str(row.get("fixture", "—"))
        med     = float(row.get("peak_lux_median", 0))
        mx      = float(row.get("peak_lux_max", 0))
        baseline = fixture_baselines.get(fixture, FALLBACK_BASELINE)
        lux_pct  = min(100, round(med / baseline * 100)) if baseline > 0 else 0

        records.append([
            round(float(row["lat"]), 6),
            round(float(row["lon"]), 6),
            status,
            color,
            fixture,
            round(med, 1),
            round(mx, 1),
            int(lux_pct),
            int(row.get("pass_count", 0)),
            str(row.get("facility_id", "—")),
        ])
    return json.dumps(records, separators=(",", ":"))


def _build_crosswalks_json(cw: pd.DataFrame) -> str:
    """[lat, lon, at_risk, type, nearest_pole_lux | null]"""
    records = []
    for _, row in cw.iterrows():
        lat = row.get("lat_")
        lon = row.get("long_")
        if pd.isna(lat) or pd.isna(lon):
            continue
        near = row.get("nearest_pole_lux")
        near_val = round(float(near), 1) if (near and not math.isnan(float(near))) else None
        records.append([
            round(float(lat), 6),
            round(float(lon), 6),
            bool(row.get("at_risk", False)),
            str(row.get("Type", "—")),
            near_val,
        ])
    return json.dumps(records, separators=(",", ":"))


def _build_fixture_summary_json(pole_health: pd.DataFrame) -> str:
    """{ "45 LED D": { failed, atrisk, healthy, avg_lux }, … }"""
    summary = {}
    for fixture, grp in pole_health.groupby("fixture"):
        summary[str(fixture)] = {
            "failed":  int((grp["status"] == "failed").sum()),
            "atrisk":  int((grp["status"] == "atrisk").sum()),
            "healthy": int((grp["status"] == "healthy").sum()),
            "avg_lux": round(float(grp["peak_lux_median"].mean()), 1),
        }
    return json.dumps(summary, separators=(",", ":"))


def _flag_crosswalks(crosswalks, pole_health, radius_deg=None):
    if radius_deg is None:
        radius_deg = CROSSWALK_RADIUS
    bad = (
        pole_health[pole_health["status"].isin(["failed", "atrisk"])]
        [["lat", "lon", "peak_lux_median"]]
        .dropna()
        .reset_index(drop=True)
    )
    cw = crosswalks.copy()
    cw_coords = cw[["lat_", "long_"]].dropna().values
    if len(bad) == 0 or len(cw_coords) == 0:
        cw["at_risk"] = False
        cw["nearest_pole_lux"] = float("nan")
        return cw
    bad_tree = cKDTree(bad[["lat", "lon"]].values)
    dists, idxs = bad_tree.query(cw_coords, distance_upper_bound=radius_deg)
    at_risk_mask = dists < radius_deg
    safe_idxs = np.clip(idxs, 0, len(bad) - 1)
    cw["at_risk"] = at_risk_mask
    cw["nearest_pole_lux"] = np.where(
        at_risk_mask,
        bad["peak_lux_median"].iloc[safe_idxs].values,
        float("nan"),
    )
    return cw


# ── Public API ────────────────────────────────────────────────────────────────

def make_map(pole_health, crosswalks, out_path="sugarland_lights.html"):
    """
    Generate the Leaflet map HTML file.

    Parameters
    ----------
    pole_health : output of analysis.aggregate_per_pole()
    crosswalks  : raw crosswalks DataFrame (from CROSSWALK_XLSX)
    out_path    : where to write the HTML (default: project root)
    """
    cw           = _flag_crosswalks(crosswalks, pole_health)
    poles_json   = _build_poles_json(pole_health)
    cw_json      = _build_crosswalks_json(cw)
    fixture_json = _build_fixture_summary_json(pole_health)

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (template
            .replace('\x01POLES\x01', poles_json)
            .replace('\x01CROSSWALKS\x01', cw_json)
            .replace('\x01FIXTURE\x01', fixture_json)
            )

    Path(out_path).write_text(html, encoding="utf-8")

    n_failed  = int((pole_health["status"] == "failed").sum())
    n_atrisk  = int((pole_health["status"] == "atrisk").sum())
    n_healthy = int((pole_health["status"] == "healthy").sum())
    print(
        f"✅ Saved → {out_path}\n"
        f"   {n_failed} failed · {n_atrisk} at-risk · {n_healthy} healthy\n"
        f"   {int(cw['at_risk'].sum())} crosswalks at risk"
    )