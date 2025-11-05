# src/evaluate.py
"""
Evaluate one-step-ahead forecasts on the test set:
- Reads benchmark (naive, mean) and ARIMA forecasts
- Aligns on the common test index
- Computes RMSFE, MAFE, MSFE and MSFE ratio vs Naive
- Performs Diebold–Mariano tests vs Naive (ARIMA vs Naive, Mean vs Naive)
- Saves a summary table and per-date errors + DM results

Inputs
------
results/forecasts/forecast_bench.csv  -> columns: actual, naive, mean
results/forecasts/forecast_arima.csv  -> columns: actual, arima

Outputs
-------
results/metrics/metrics_summary.csv
results/metrics/errors_by_date.csv
results/metrics/dm_tests.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t  # for HLN p-values

from config import ROOT

# ---- Paths ----
FORECASTS_DIR = ROOT / "results" / "forecasts"
METRICS_DIR   = ROOT / "results" / "metrics"
BENCH_CSV = FORECASTS_DIR / "forecast_bench.csv"
ARIMA_CSV = FORECASTS_DIR / "forecast_arima.csv"
OUT_SUMMARY = METRICS_DIR / "metrics_summary.csv"
OUT_ERRORS  = METRICS_DIR / "errors_by_date.csv"
OUT_DM      = METRICS_DIR / "dm_tests.csv"


def ensure_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_forecasts() -> pd.DataFrame:
    """Load and align benchmark + ARIMA forecasts on common dates."""
    bench = pd.read_csv(BENCH_CSV, parse_dates=["observation_date"]).set_index("observation_date")
    arima = pd.read_csv(ARIMA_CSV, parse_dates=["observation_date"]).set_index("observation_date")

    idx = bench.index.intersection(arima.index).sort_values()

    df = pd.DataFrame(index=idx)
    df["actual"] = bench.loc[idx, "actual"]
    df["naive"]  = bench.loc[idx, "naive"]
    df["mean"]   = bench.loc[idx, "mean"]
    df["arima"]  = arima.loc[idx, "arima"]
    return df


# --------- Metrics ----------
def rmsfe(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mafe(y, yhat) -> float:
    return float(np.mean(np.abs(y - yhat)))


# --------- Diebold–Mariano ----------
def _newey_west_var(d: np.ndarray, lag: int) -> float:
    """
    Newey–West long-run variance of d_t, with Bartlett weights and given lag.
    For h-step ahead forecasts, lag should be h-1. Here usually h=1 -> lag=0.
    """
    T = len(d)
    d = d - d.mean()
    # gamma_0
    gamma0 = np.sum(d * d) / T
    if lag <= 0:
        return gamma0

    # Autocovariances up to 'lag' with Bartlett weights
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.sum(d[k:] * d[:-k]) / T
        weight = 1.0 - k / (lag + 1.0)
        var += 2.0 * weight * gamma_k
    return var


def diebold_mariano(
    y: np.ndarray,
    yhat_i: np.ndarray,
    yhat_j: np.ndarray,
    h: int = 1,
    power: int = 2,
    use_harvey_correction: bool = True,
) -> tuple[float, float, float]:
    """
    DM test for equal predictive accuracy between model i and j.
    Loss = |e|^power (power=2 -> squared error). Returns (dm_stat, p_value, dbar).

    We compute d_t = loss_i - loss_j. If dbar < 0, model i has lower loss (better).
    For 1-step ahead (h=1), NW lag = 0.
    """
    e_i = y - yhat_i
    e_j = y - yhat_j
    # loss difference
    d = np.abs(e_i) ** power - np.abs(e_j) ** power
    T = len(d)
    dbar = float(d.mean())

    # Long-run variance of dbar using Newey–West with lag = h - 1
    lag = max(h - 1, 0)
    s2 = _newey_west_var(d.astype(float), lag=lag)
    # Std error of the sample mean
    se = np.sqrt(s2 / T)

    # Raw DM statistic (asymptotic normal)
    dm = dbar / se if se > 0 else np.inf

    # Harvey–Leybourne–Newbold small-sample correction (t approx)
    if use_harvey_correction:
        # HLN scaling factor
        k = h
        scale = np.sqrt((T + 1 - 2 * k + (k * (k - 1)) / T) / T)
        dm_adj = dm * scale
        df = T - 1
        pval = 2.0 * (1.0 - student_t.cdf(np.abs(dm_adj), df=df))
        return float(dm_adj), float(pval), dbar
    else:
        # Asymptotic normal (fallback)
        from scipy.stats import norm
        pval = 2.0 * (1.0 - norm.cdf(np.abs(dm)))
        return float(dm), float(pval), dbar


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute RMSFE, MAFE, MSFE, MSFE ratio vs Naive.
    Also compute DM tests vs Naive for ARIMA and Mean.
    Returns:
      summary (per model), errors_by_date, dm_table (pairwise vs naive)
    """
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

    # Errors by date (for plots)
    errors = pd.DataFrame(index=df.index)
    for m in models:
        errors[f"e_{m}"] = y - df[m]

    # DM tests vs Naive (1-step, squared error)
    dm_rows = []
    for m in ["mean", "arima"]:
        dm_stat, dm_p, dbar = diebold_mariano(
            y.values, df[m].values, df["naive"].values, h=1, power=2, use_harvey_correction=True
        )
        dm_rows.append({
            "model_i": m,
            "model_j": "naive",
            "h": 1,
            "loss": "squared",
            "dbar": dbar,                 # mean(loss_i - loss_j); <0 => i better
            "DM_stat": dm_stat,
            "DM_pvalue": dm_p,
        })
        # add to summary table as extra columns (only for non-naive rows)
        summary.loc[m, "DM_vs_naive_stat"] = dm_stat
        summary.loc[m, "DM_vs_naive_pvalue"] = dm_p

    # Naive row has N/A for DM vs naive
    summary.loc["naive", ["DM_vs_naive_stat", "DM_vs_naive_pvalue"]] = np.nan

    dm_table = pd.DataFrame(dm_rows).set_index(["model_i", "model_j"])

    return summary, errors, dm_table


def main():
    ensure_dirs()
    df = load_forecasts()
    summary, errors, dm_table = evaluate(df)

    # Save
    summary.to_csv(OUT_SUMMARY, float_format="%.6f")
    errors.to_csv(OUT_ERRORS, date_format="%Y-%m-01", float_format="%.6f")
    dm_table.to_csv(OUT_DM, float_format="%.6f")

    # Console preview
    print("=== Evaluation (test set) ===")
    print(summary)
    print(f"\nSaved summary: {OUT_SUMMARY}")
    print(f"Saved errors:  {OUT_ERRORS}")
    print(f"Saved DM:      {OUT_DM}")


if __name__ == "__main__":
    main()