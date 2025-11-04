# src/evaluate.py
"""
Evaluate one-step-ahead forecasts on the test set:
- Reads benchmark forecasts (naive, mean) and ARIMA forecasts
- Aligns on the common test index
- Computes RMSFE, MAFE, MSFE and MSFE ratio vs Naive
- Saves a summary table and per-date errors for future plots

Inputs
------
results/forecasts/forecast_bench.csv  -> columns: actual, naive, mean
results/forecasts/forecast_arima.csv  -> columns: actual, arima

Outputs
-------
results/metrics/metrics_summary.csv
results/metrics/errors_by_date.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import ROOT

# ---- Paths ----
FORECASTS_DIR = ROOT / "results" / "forecasts"
METRICS_DIR   = ROOT / "results" / "metrics"
BENCH_CSV = FORECASTS_DIR / "forecast_bench.csv"
ARIMA_CSV = FORECASTS_DIR / "forecast_arima.csv"
OUT_SUMMARY = METRICS_DIR / "metrics_summary.csv"
OUT_ERRORS  = METRICS_DIR / "errors_by_date.csv"


def ensure_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_forecasts() -> pd.DataFrame:
    """Load and align benchmark + ARIMA forecasts on common dates."""
    bench = pd.read_csv(BENCH_CSV, parse_dates=["observation_date"]).set_index("observation_date")
    arima = pd.read_csv(ARIMA_CSV, parse_dates=["observation_date"]).set_index("observation_date")

    # Intersect indices to be safe (should be full test for both)
    idx = bench.index.intersection(arima.index).sort_values()

    df = pd.DataFrame(index=idx)
    df["actual"] = bench.loc[idx, "actual"]
    df["naive"]  = bench.loc[idx, "naive"]
    df["mean"]   = bench.loc[idx, "mean"]
    df["arima"]  = arima.loc[idx, "arima"]
    return df


def rmsfe(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mafe(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSFE, MAFE, MSFE and MSFE ratio vs Naive for each model."""
    models = ["naive", "mean", "arima"]
    y = df["actual"]

    # Base metrics
    rows = []
    msfe_naive = np.mean((y - df["naive"]) ** 2)
    for m in models:
        msfe_m = np.mean((y - df[m]) ** 2)
        row = {
            "model": m,
            "RMSFE": rmsfe(y, df[m]),
            "MAFE":  mafe(y, df[m]),
            "MSFE":  float(msfe_m),
            "MSFE_ratio_vs_naive": float(msfe_m / msfe_naive),
        }
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model").sort_values("RMSFE")

    # Errors by date (useful for later plots)
    errors = pd.DataFrame(index=df.index)
    for m in models:
        errors[f"e_{m}"] = y - df[m]

    return summary, errors


def main():
    ensure_dirs()
    df = load_forecasts()
    summary, errors = evaluate(df)

    # Save
    summary.to_csv(OUT_SUMMARY, float_format="%.6f")
    errors.to_csv(OUT_ERRORS, date_format="%Y-%m-01", float_format="%.6f")

    # Console preview
    print("=== Evaluation (test set) ===")
    print(summary)
    print(f"\nSaved summary: {OUT_SUMMARY}")
    print(f"Saved errors:  {OUT_ERRORS}")


if __name__ == "__main__":
    main()