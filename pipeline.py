import argparse
import pandas as pd
from pathlib import Path

from config import POLES_XLSX, CROSSWALK_XLSX, CACHE_DIR
from analysis import load_all_peaks, match_peaks_to_poles, aggregate_per_pole
from dash.map import make_map

PEAKS_CACHE     = CACHE_DIR / "peaks.parquet"
POLE_STATS_CACHE = CACHE_DIR / "pole_stats.parquet"


def run(force: bool = False, map_only: bool = False):
    poles      = pd.read_excel(POLES_XLSX)
    crosswalks = pd.read_excel(CROSSWALK_XLSX)

    if map_only and POLE_STATS_CACHE.exists():
        print("⚡ Map-only mode — loading cached pole stats")
        pole_health = pd.read_parquet(POLE_STATS_CACHE)
    else:
        # ── peaks ─────────────────────────────────────────────────────────
        if PEAKS_CACHE.exists() and not force:
            print("⚡ Loading cached peaks")
            all_peaks = pd.read_parquet(PEAKS_CACHE)
        else:
            all_peaks = load_all_peaks()
            all_peaks.to_parquet(PEAKS_CACHE)
            print(f"   Cached → {PEAKS_CACHE}")

        # ── match + aggregate ──────────────────────────────────────────────
        # Build GPS coverage from peaks (good enough — every peak row has a GPS point)
        matched = match_peaks_to_poles(all_peaks, poles)

        # For 'route coverage' we need raw GPS too so we can flag dark poles
        # Pass None here if you don't want to load all CSVs a second time;
        # dark poles (in route, no peak) will simply not appear in the output.
        pole_health = aggregate_per_pole(matched, poles, gps_all=None)
        pole_health.to_parquet(POLE_STATS_CACHE)
        print(f"   Cached → {POLE_STATS_CACHE}")

    make_map(pole_health, crosswalks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sugar Land light health pipeline")
    parser.add_argument("--force",    action="store_true", help="Ignore all caches")
    parser.add_argument("--map-only", action="store_true", help="Only regenerate map HTML")
    args = parser.parse_args()
    run(force=args.force, map_only=args.map_only)