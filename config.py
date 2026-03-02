from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("City of Sugar Land Light luminosity")          # CSVs live here (or pass override to load_all_peaks)
CACHE_DIR      = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

POLES_XLSX     = Path("City of Sugar Land Streetlight locations.xlsx")
TRAFFIC_XLSX   = Path("City of Sugar Land Traffic Roadway Lights locations.xlsx")
CROSSWALK_XLSX = Path("City of Sugar Land Crosswalk Survey with Coordinates.xlsx")

# ── Column names ───────────────────────────────────────────────────────────────
LAT_COL  = "Latitude"
LON_COL  = "Longitude"

# ── Spatial matching ───────────────────────────────────────────────────────────
MATCH_RADIUS     = 0.0003   # degrees ≈ 33 m  — peak→pole match window
CROSSWALK_RADIUS = 0.0008    # degrees ≈ 111 m — bad pole→crosswalk risk window

# ── Peak detection (scipy.signal.find_peaks) ───────────────────────────────────
# height     : a reading must exceed this lux to be considered a pole passage.
#              Set above ambient road surface noise (~3-5 lux calibrated).
# distance   : minimum rows between peaks. At 10 Hz and 30 mph the car moves
#              ~1.3 m/reading; poles are ≥30 m apart → ~23 rows minimum.
# prominence : peak must rise this much above its local baseline. Prevents
#              counting gradual brightness changes as pole detections.
PEAK_HEIGHT     = 8     # lux
PEAK_DISTANCE   = 15    # rows
PEAK_PROMINENCE = 4     # lux

# ── Health thresholds (peak_lux_median, per fixture type) ─────────────────────
# Values derived from observed median peak lux in Day61 data.
# 'failed' upper bound  = ~30% of expected median (lamp essentially dark)
# 'atrisk' upper bound  = ~65% of expected median (lamp dim / degraded)
# anything above atrisk = 'healthy'
FIXTURE_THRESHOLDS = {
    "45 LED D":  {"failed":  8, "atrisk": 16},   # median healthy: ~25 lux
    "95 LED A":  {"failed":  9, "atrisk": 19},   # median healthy: ~29 lux
    "115 LED B": {"failed": 14, "atrisk": 30},   # median healthy: ~47 lux
    "150 HPS P": {"failed":  8, "atrisk": 16},   # limited sample
    "250 HPS S": {"failed":  6, "atrisk": 12},   # median healthy: ~18 lux
    "400 HPS T": {"failed":  8, "atrisk": 16},   # limited sample
    "100 HPS L": {"failed":  7, "atrisk": 15},   # limited sample
    "70 HPS G":  {"failed":  6, "atrisk": 12},   # limited sample
}

# Fallback for fixture types not in the table above
DEFAULT_THRESHOLDS = {"failed": 8, "at risk": 16}

# ── Map display ────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "failed":    "#ff4444",
    "at risk":    "#ffaa00",
    "healthy":   "#44cc44",
    "unsurveyed":"#444466",
}

MAP_CENTER  = [29.63, -95.61]
MAP_ZOOM    = 13