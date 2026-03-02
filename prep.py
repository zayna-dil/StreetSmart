import pandas as pd
import glob, os, re
from pathlib import Path
from config import DATA_DIR, CACHE_DIR, LAT_COL, LON_COL, LUX_COLS, SKIPROWS

CACHE_FILE = CACHE_DIR / "readings.parquet"

def parse_cal(line):
    a = float(re.search(r"a=([-\d.]+)", line).group(1))
    b = float(re.search(r"b=([-\d.]+)", line).group(1))
    c = float(re.search(r"c=([-\d.]+)", line).group(1))
    return (a, b, c)

def apply_cal(raw, cal):
    a, b, c = cal
    return a * raw**2 + b * raw + c

def parse_metadata(filepath):
    meta = {"description": None, "date": None,
            "file": os.path.basename(filepath), "cal1": None, "cal2": None}
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i > 7: break
            if i == 1 and "Description" in line:
                meta["description"] = re.search(r"Description:\s*(.+)", line).group(1).strip()
            if i == 2 and "Date" in line:
                meta["date"] = re.search(r"Date:\s*(.+)", line).group(1).strip()
            if i == 5 and "LuxCal1" in line:
                meta["cal1"] = parse_cal(line)
            if i == 6 and "LuxCal2" in line:
                meta["cal2"] = parse_cal(line)
    return meta

def load_files(data_dir=DATA_DIR, force=False):
    if CACHE_FILE.exists() and not force:
        print(f"⚡ Loading from cache: {CACHE_FILE}")
        return pd.read_parquet(CACHE_FILE)

    print("📂 Scanning CSVs (first run, will cache)...")
    all_files = glob.glob(str(data_dir / "*.csv"))
    dfs = []
    for filepath in all_files:
        temp = pd.read_csv(filepath, skiprows=SKIPROWS)
        temp['source_file'] = os.path.basename(filepath)
        meta = parse_metadata(filepath)
        temp['file_date'] = pd.to_datetime(meta['date'], errors='coerce')
        for i, col in enumerate(LUX_COLS, start=1):
            cal = meta[f'cal{i}']
            if cal and col in temp.columns:
                temp[col] = temp[col].apply(lambda x: apply_cal(x, cal))
        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(CACHE_FILE)   # ← saved for next run
    print(f"✅ Cached {len(df):,} rows to {CACHE_FILE}")
    return df