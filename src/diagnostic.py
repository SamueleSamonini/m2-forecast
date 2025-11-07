# src/diagnostics.py
"""
Residual diagnostics for ARIMA fitted on the TRAIN sample only.

Inputs
------
data/processed/m2_growth.parquet  (contains TARGET_COL, e.g., 'g')
data/processed/split_info.json    (contains 'cut_index')

Config (from src/config.py)
---------------------------
ARIMA_ORDER        tuple, e.g. (1,0,1)
SEASONAL_ORDER     tuple, e.g. (0,0,0,0) or (P,0,Q,12)
ARIMA_TREND        'c' or 'n'
ARIMA_MAXITER      int (optional, default 200)

Outputs
-------
results/metrics/diagnostics_arima_train.csv
results/figures/03_residuals_train.png
results/figures/04_resid_acf_pacf.png
results/figures/05_resid_hist_qq.png
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.gofplots import qqplot

import config

# ---- Paths ----
ROOT = config.ROOT
DATA_PROCESSED = config.DATA_PROCESSED
SPLIT_INFO = config.SPLIT_INFO
TARGET_COL = config.TARGET_COL

FIG_DIR     = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"
OUT_CSV     = METRICS_DIR / "diagnostics_arima_train.csv"

# ---- ARIMA hyperparams (with defaults) ----
ARIMA_ORDER     = getattr(config, "ARIMA_ORDER", (1, 0, 1))
SEASONAL_ORDER  = getattr(config, "SEASONAL_ORDER", (0, 0, 0, 0))
ARIMA_TREND     = getattr(config, "ARIMA_TREND", "c")
MAXITER         = getattr(config, "ARIMA_MAXITER", 200)

# ---- Ljung–Box lags to report ----
LB_LAGS = [6, 12, 18, 24]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_train_series() -> pd.Series:
    """Load processed target and return TRAIN sample only, based on split_info."""
    df = pd.read_parquet(DATA_PROCESSED)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column {TARGET_COL!r} not found in {DATA_PROCESSED}")
    s = df[TARGET_COL].copy().dropna()
    s.index = pd.to_datetime(s.index)

    with open(SPLIT_INFO, "r") as f:
        meta = json.load(f)
    cut = int(meta["cut_index"])
    train = s.iloc[:cut].copy()
    return train


def fit_arima_train(train: pd.Series):
    """Fit ARIMA on TRAIN only and return fitted results."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(
            train,
            order=ARIMA_ORDER,
            seasonal_order=SEASONAL_ORDER,
            trend=ARIMA_TREND,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(method_kwargs={"maxiter": MAXITER})
    return res


def compute_diagnostics(residuals: pd.Series) -> pd.DataFrame:
    """Compute Ljung–Box, Jarque–Bera, and ARCH LM diagnostics."""
    residuals = pd.Series(residuals).dropna()
    out_rows = []

    # --- Ljung–Box at multiple lags ---
    lb = acorr_ljungbox(residuals, lags=LB_LAGS, return_df=True)
    # For each lag, keep statistic and pvalue
    for lag in LB_LAGS:
        out_rows.append({
            "test": f"LjungBox_lag{lag}",
            "stat": float(lb.loc[lag, "lb_stat"]) if lag in lb.index else np.nan,
            "pvalue": float(lb.loc[lag, "lb_pvalue"]) if lag in lb.index else np.nan,
        })

    # --- Jarque–Bera normality ---
    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
    out_rows.append({"test": "JarqueBera", "stat": float(jb_stat), "pvalue": float(jb_p)})
    out_rows.append({"test": "Skewness",   "stat": float(skew),    "pvalue": np.nan})
    out_rows.append({"test": "Kurtosis",   "stat": float(kurt),    "pvalue": np.nan})

    # --- ARCH LM (heteroskedasticity) ---
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=12)
    out_rows.append({"test": "ARCH_LM_L12", "stat": float(lm_stat), "pvalue": float(lm_pvalue)})
    out_rows.append({"test": "ARCH_F_L12",  "stat": float(f_stat),  "pvalue": float(f_pvalue)})

    # Assemble
    diag = pd.DataFrame(out_rows)
    return diag


def save_plots(train: pd.Series, residuals: pd.Series) -> None:
    """Save residual time series, ACF/PACF, and histogram + QQ-plot."""
    # 1) Residual time series
    plt.figure(figsize=(10, 4))
    plt.plot(residuals.index, residuals.values, label="Residuals")
    plt.axhline(0.0, linewidth=1)
    plt.title("ARIMA Residuals (Train)")
    plt.xlabel("Date"); plt.ylabel("Residual")
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_residuals_train.png", dpi=200); plt.close()

    # 2) ACF/PACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, lags=36, ax=axes[0]); axes[0].set_title("ACF Residuals (36)")
    plot_pacf(residuals, lags=36, ax=axes[1], method="ywm"); axes[1].set_title("PACF Residuals (36)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_resid_acf_pacf.png", dpi=200); plt.close(fig)

    # 3) Histogram + QQ
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(residuals, bins=30)
    ax1.set_title("Residuals Histogram")
    ax1.set_xlabel("Residual"); ax1.set_ylabel("Frequency")

    ax2 = fig.add_subplot(1, 2, 2)
    qqplot(residuals, line="s", ax=ax2)
    ax2.set_title("QQ-Plot Residuals")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_resid_hist_qq.png", dpi=200); plt.close(fig)


def main():
    ensure_dirs()
    train = load_train_series()
    res = fit_arima_train(train)

    residuals = pd.Series(res.resid, index=train.index).dropna()

    # Diagnostics
    diag = compute_diagnostics(residuals)

    # Save CSV (add model/meta info in a companion row-block)
    meta_rows = [{
        "test": "META_model",
        "stat": f"order={ARIMA_ORDER}, seasonal={SEASONAL_ORDER}, trend={ARIMA_TREND}",
        "pvalue": np.nan
    },{
        "test": "META_lengths",
        "stat": f"train_n={len(train)}, resid_n={len(residuals)}",
        "pvalue": np.nan
    }]
    diag_full = pd.concat([pd.DataFrame(meta_rows), diag], ignore_index=True)
    diag_full.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # Save figures
    save_plots(train, residuals)

    # Console summary
    print("=== ARIMA Diagnostics (Train) ===")
    print(f"Model: order={ARIMA_ORDER}, seasonal={SEASONAL_ORDER}, trend={ARIMA_TREND}")
    print(f"Train length: {len(train)} | Residuals: {len(residuals)}")
    print("Saved CSV:", OUT_CSV)
    print("Saved figs:",
          ROOT / 'results' / 'figures' / '03_residuals_train.png', ",",
          ROOT / 'results' / 'figures' / '04_resid_acf_pacf.png', ",",
          ROOT / 'results' / 'figures' / '05_resid_hist_qq.png')
    # Optional: print first few diagnostics
    print("\nDiagnostics head():")
    print(diag_full.head(8))


if __name__ == "__main__":
    main()
