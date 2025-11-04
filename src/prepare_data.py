"""
Prepare processed dataset for M2 forecasting:
- Load raw FRED file (DATE, M2SL)
- Ensure monthly frequency and proper ordering
- Create log(M2) and monthly growth target g_t = 100 * diff(log(M2))
- Save processed parquet and split info (train/test indices)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    DATA_RAW, DATA_PROCESSED, SPLIT_INFO,
    DATE_COL, LEVEL_COL, LOG_COL, TARGET_COL,
    TEST_FRACTION
)

def ensure_dirs():
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)

def load_raw(path: Path) -> pd.DataFrame:
    """Load raw CSV with DATE, M2SL; parse dates and sort index monthly."""
    df = pd.read_csv(path)
    # Basic sanity checks
    if DATE_COL not in df.columns or LEVEL_COL not in df.columns:
        raise ValueError(f"Expected columns {DATE_COL}, {LEVEL_COL} in {path}")
    # Parse date and set index
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    # Enforce monthly start frequency (will insert NaNs if gaps exist)
    df = df.asfreq("MS")
    return df

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create log(M2) and monthly growth g_t = 100 * diff(log(M2))."""
    # Drop non-positive values to be safe for log
    df = df[df[LEVEL_COL] > 0].copy()

    # Create log level
    df[LOG_COL] = np.log(df[LEVEL_COL])

    # Monthly growth in percentage points
    df[TARGET_COL] = 100 * df[LOG_COL].diff()

    # Drop initial NaN from diff
    df = df.dropna(subset=[TARGET_COL])
    return df

def train_test_split_index(n: int, test_fraction: float) -> dict:
    """Return integer cut index and slices for train/test (last % as test)."""
    n_test = max(int(np.ceil(n * test_fraction)), 1)
    cut = n - n_test
    return {"n": n, "n_test": n_test, "cut": cut}

def main():
    ensure_dirs()

    # 1) Load raw
    raw = load_raw(DATA_RAW)

    # 2) Build target g_t
    proc = build_target(raw)

    # 3) Compute split info (just indexes; no slicing here)
    split = train_test_split_index(len(proc), TEST_FRACTION)

    # 4) Save outputs
    proc.to_parquet(DATA_PROCESSED)
    with open(SPLIT_INFO, "w") as f:
        json.dump({
            "n_obs": split["n"],
            "n_test": split["n_test"],
            "cut_index": split["cut"],
            "index_start": str(proc.index.min().date()),
            "index_end": str(proc.index.max().date()),
            "test_fraction": TEST_FRACTION,
            "columns": proc.columns.tolist()
        }, f, indent=2)

    # 5) Console summary (useful feedback when you run the script)
    print("=== Prepare Data ===")
    print(f"Raw path:      {DATA_RAW}")
    print(f"Processed:     {DATA_PROCESSED}")
    print(f"Split info:    {SPLIT_INFO}")
    print(f"Obs total:     {split['n']}")
    print(f"Obs test:      {split['n_test']}  (last {int(TEST_FRACTION*100)}%)")
    print(f"Train end idx: {split['cut']-1}  | Test start idx: {split['cut']}")
    print(f"Period:        {proc.index.min().date()} → {proc.index.max().date()}")
    print(f"Columns:       {proc.columns.tolist()}")

if __name__ == "__main__":
    main()
