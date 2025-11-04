# src/plotting.py
"""
Plot utilities for test-period forecasts:
- Actual vs forecasts (naive, mean, arima)
- Errors over time for each model
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from config import ROOT

# Paths
FORECASTS_DIR = ROOT / "results" / "forecasts"
FIG_DIR       = ROOT / "results" / "figures"

BENCH_CSV = FORECASTS_DIR / "forecast_bench.csv"   # actual, naive, mean
ARIMA_CSV = FORECASTS_DIR / "forecast_arima.csv"   # actual, arima


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_forecasts_merged() -> pd.DataFrame:
    """Return a DataFrame (test set only) with columns: actual, naive, mean, arima."""
    bench = pd.read_csv(BENCH_CSV, parse_dates=["observation_date"]).set_index("observation_date")
    arima = pd.read_csv(ARIMA_CSV, parse_dates=["observation_date"]).set_index("observation_date")
    idx = bench.index.intersection(arima.index).sort_values()
    df = pd.DataFrame(index=idx)
    df["actual"] = bench.loc[idx, "actual"]
    df["naive"]  = bench.loc[idx, "naive"]
    df["mean"]   = bench.loc[idx, "mean"]
    df["arima"]  = arima.loc[idx, "arima"]
    return df


def plot_actual_vs_forecasts(df: pd.DataFrame, out_path: Path) -> Path:
    """Line plot of Actual vs forecasts on the test period."""
    plt.figure(figsize=(10, 4.5))
    plt.plot(df.index, df["actual"], label="Actual")
    plt.plot(df.index, df["naive"],  label="Naive",  alpha=0.9)
    plt.plot(df.index, df["mean"],   label="Mean",   alpha=0.9)
    plt.plot(df.index, df["arima"],  label="ARIMA",  alpha=0.9)

    plt.title("Monthly M2 Growth – Actual vs 1-step Forecasts (Test)")
    plt.xlabel("Date")
    plt.ylabel("g_t (% points)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_errors_over_time(df: pd.DataFrame, out_path: Path) -> Path:
    """Line plot of errors (Actual - Forecast) over time for each model on the test period."""
    errors = pd.DataFrame(index=df.index)
    errors["e_naive"] = df["actual"] - df["naive"]
    errors["e_mean"]  = df["actual"] - df["mean"]
    errors["e_arima"] = df["actual"] - df["arima"]

    plt.figure(figsize=(10, 4.5))
    plt.plot(errors.index, errors["e_naive"], label="Error Naive")
    plt.plot(errors.index, errors["e_mean"],  label="Error Mean")
    plt.plot(errors.index, errors["e_arima"], label="Error ARIMA")

    plt.axhline(0.0, linewidth=1)
    plt.title("Forecast Errors over Time (Test)")
    plt.xlabel("Date")
    plt.ylabel("Error = Actual − Forecast")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def make_all_plots() -> None:
    ensure_dirs()
    df = load_forecasts_merged()
    out1 = FIG_DIR / "01_actual_vs_forecasts.png"
    out2 = FIG_DIR / "02_errors_over_time.png"
    plot_actual_vs_forecasts(df, out1)
    plot_errors_over_time(df, out2)
    print("Saved figures:")
    print(f"- {out1}")
    print(f"- {out2}")


if __name__ == "__main__":
    make_all_plots()
