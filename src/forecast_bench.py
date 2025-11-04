# src/forecast_bench.py
"""
Benchmarks for one-step-ahead forecasting of g_t:
- Naive (RW on growth): ĝ_{t+1|t} = g_t
- Mean (constant mean estimated on train)

Inputs:
  - data/processed/m2_growth.parquet  (must contain column TARGET_COL, e.g. 'g')
  - data/processed/split_info.json    (must contain 'cut_index' for 80/20 split)
Outputs:
  - results/forecasts/forecast_bench.csv  with columns: actual, naive, mean
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

from config import ROOT, DATA_PROCESSED, SPLIT_INFO, TARGET_COL

# Where to save forecasts
FORECASTS_DIR = ROOT / "results" / "forecasts"
OUT_CSV = FORECASTS_DIR / "forecast_bench.csv"


def ensure_dirs() -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)


def load_series() -> pd.Series:
    """Load processed dataset and return the target series g_t as a pandas Series."""
    df = pd.read_parquet(DATA_PROCESSED)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column {TARGET_COL!r} not found in {DATA_PROCESSED}")
    s = df[TARGET_COL].copy().dropna()
    # Ensure datetime index (should already be)
    s.index = pd.to_datetime(s.index)
    return s


def load_split() -> int:
    """Read cut index (train end) from split_info.json."""
    with open(SPLIT_INFO, "r") as f:
        meta = json.load(f)
    return int(meta["cut_index"])


def make_benchmarks(s: pd.Series, cut: int) -> pd.DataFrame:
    """
    Build one-step-ahead forecasts for test period using:
      - naive: ĝ_{t+1|t} = g_t  (implemented via shift(1))
      - mean:  constant mean of train (no re-fitting during test)

    Returns a DataFrame indexed by test dates with columns: actual, naive, mean.
    """
    train = s.iloc[:cut]           # everything up to cut-1
    test = s.iloc[cut:]            # last 20% (holdout)

    # Naïve via shift(1): for each test time t, prediction equals actual at t-1.
    # The first test prediction uses the last train value (consistent with expanding-window use).
    naive = s.shift(1).iloc[cut:]

    # Mean benchmark: constant mean computed on TRAIN ONLY (no "peeking" into test).
    mean_level = float(train.mean())
    mean = pd.Series(mean_level, index=test.index)

    df_fc = pd.DataFrame({
        "actual": test,
        "naive": naive,
        "mean": mean,
    })
    return df_fc


def save_forecasts(df_fc: pd.DataFrame) -> Path:
    ensure_dirs()
    df_fc.to_csv(OUT_CSV, index=True, date_format="%Y-%m-01")
    return OUT_CSV


def main():
    s = load_series()
    cut = load_split()
    fc = make_benchmarks(s, cut)
    out_path = save_forecasts(fc)

    # Console summary
    print("=== Benchmark forecasts ===")
    print(f"Train length: {cut} | Test length: {len(fc)}")
    print(f"Saved: {out_path}")
    print("Preview:")
    print(fc.head(5))


if __name__ == "__main__":
    main()
