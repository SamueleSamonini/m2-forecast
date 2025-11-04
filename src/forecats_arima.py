# src/forecast_arima.py
"""
One-step-ahead ARIMA forecasts for g_t using an expanding (rolling-refit) window.

Inputs
------
- data/processed/m2_growth.parquet   (must contain TARGET_COL, e.g. 'g')
- data/processed/split_info.json     (contains 'cut_index' for 80/20 split)

Outputs
-------
- results/forecasts/forecast_arima.csv  with columns: actual, arima
"""

from __future__ import annotations
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import config  # we read paths + (optional) ARIMA params from here

# ---- Paths ----
ROOT = config.ROOT
DATA_PROCESSED = config.DATA_PROCESSED
SPLIT_INFO = config.SPLIT_INFO
TARGET_COL = config.TARGET_COL

FORECASTS_DIR = ROOT / "results" / "forecasts"
OUT_CSV = FORECASTS_DIR / "forecast_arima.csv"

# ---- ARIMA hyperparameters (with safe defaults) ----
ARIMA_ORDER = getattr(config, "ARIMA_ORDER", (1, 0, 1))          # (p,d,q) on g_t
SEASONAL_ORDER = getattr(config, "SEASONAL_ORDER", (0, 0, 0, 0))  # (P,D,Q,s) if needed
ARIMA_TREND = getattr(config, "ARIMA_TREND", "c")                 # 'n' no-const, 'c' constant

# You can tweak fit robustness here if needed
MAXITER = getattr(config, "ARIMA_MAXITER", 200)


def ensure_dirs() -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)


def load_series() -> pd.Series:
    """Load processed dataset and return the target series g_t (monthly growth)."""
    df = pd.read_parquet(DATA_PROCESSED)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column {TARGET_COL!r} not found in {DATA_PROCESSED}")
    s = df[TARGET_COL].copy().dropna()
    s.index = pd.to_datetime(s.index)  # ensure datetime index
    return s


def load_cut_index() -> int:
    """Read the train/test cut index from split_info.json."""
    with open(SPLIT_INFO, "r") as f:
        meta = json.load(f)
    return int(meta["cut_index"])


def arima_fit_forecast_one_step(history: pd.Series) -> float:
    """
    Fit ARIMA to 'history' and return 1-step ahead forecast.
    We suppress convergence warnings to keep console clean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(
            history,
            order=ARIMA_ORDER,
            seasonal_order=SEASONAL_ORDER,
            trend=ARIMA_TREND,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(method_kwargs={"maxiter": MAXITER})
        fc = res.forecast(steps=1).iloc[0]
    return float(fc)


def rolling_arima_forecasts(s: pd.Series, cut: int) -> pd.Series:
    """
    Expanding window:
      for each test time t, fit ARIMA on s[:t] and forecast t+1 (which aligns to index t).
    Returns a Series indexed by the test dates.
    """
    test_index = s.index[cut:]
    preds = []
    history = s.iloc[:cut].copy()

    for t in test_index:
        try:
            yhat = arima_fit_forecast_one_step(history)
        except Exception as e:
            # Fallback: naive last value (robustness if ARIMA fails at some step)
            yhat = float(history.iloc[-1])
        preds.append(yhat)
        # expand history with the actual observed value at time t
        history = pd.concat([history, pd.Series([s.loc[t]], index=[t])])

    return pd.Series(preds, index=test_index, name="arima")


def save_forecasts(df_fc: pd.DataFrame) -> Path:
    ensure_dirs()
    df_fc.to_csv(OUT_CSV, index=True, date_format="%Y-%m-01")
    return OUT_CSV


def main():
    s = load_series()
    cut = load_cut_index()

    arima_hat = rolling_arima_forecasts(s, cut)
    actual = s.iloc[cut:]

    df_fc = pd.DataFrame({"actual": actual, "arima": arima_hat})
    out_path = save_forecasts(df_fc)

    print("=== ARIMA forecasts (rolling 1-step) ===")
    print(f"Order: {ARIMA_ORDER}, Seasonal: {SEASONAL_ORDER}, Trend: {ARIMA_TREND}")
    print(f"Train length: {cut} | Test length: {len(df_fc)}")
    print(f"Saved: {out_path}")
    print("Preview:")
    print(df_fc.head(5))


if __name__ == "__main__":
    main()
