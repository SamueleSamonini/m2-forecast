# src/tune_arima.py
"""
Grid search for ARIMA/SARIMA on TRAIN ONLY (no leakage).
- Tries a small grid over (p,q), seasonal (P,Q,s=12), and trend in {'n','c'} with d=D=0.
- Ranks by chosen information criterion (BIC default, AIC optional).
- Saves full grid and best spec.

Outputs
-------
results/metrics/arima_grid_search.csv
results/metrics/best_arima.json
"""

from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import config

ROOT = config.ROOT
DATA_PROCESSED = config.DATA_PROCESSED
SPLIT_INFO = config.SPLIT_INFO
TARGET_COL = config.TARGET_COL
METRICS_DIR = ROOT / "results" / "metrics"
OUT_GRID = METRICS_DIR / "arima_grid_search.csv"
OUT_BEST = METRICS_DIR / "best_arima.json"

# Search grid (balanced: ~200 fits)
PQS = [(0,0), (1,0), (0,1), (1,1)]   # seasonal (P,Q) at s=12
PS = [0,1,2,3,4]                      # p
QS = [0,1,2,3,4]                      # q
TRENDS = ["n", "c"]                   # no-const, const
S = 12                                # monthly seasonality
MAXITER = getattr(config, "ARIMA_MAXITER", 200)

@dataclass
class FitResult:
    p: int
    q: int
    P: int
    Q: int
    s: int
    trend: str
    aic: float
    bic: float
    n_params: int
    converged: bool
    error: str | None = None

def load_train() -> pd.Series:
    df = pd.read_parquet(DATA_PROCESSED)
    s = df[TARGET_COL].dropna().copy()
    s.index = pd.to_datetime(s.index)
    meta = json.loads(Path(SPLIT_INFO).read_text())
    cut = int(meta["cut_index"])
    return s.iloc[:cut].copy()

def try_fit(y: pd.Series, order: Tuple[int,int,int], seasonal_order: Tuple[int,int,int,int], trend: str) -> FitResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(
                y,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(method_kwargs={"maxiter": MAXITER})
            aic = float(res.aic)
            bic = float(res.bic)
            k = int(res.params.shape[0])
            conv = bool(res.mle_retvals.get("converged", True))
            return FitResult(order[0], order[2], seasonal_order[0], seasonal_order[2], seasonal_order[3], trend, aic, bic, k, conv, None)
        except Exception as e:
            return FitResult(order[0], order[2], seasonal_order[0], seasonal_order[2], seasonal_order[3], trend, np.inf, np.inf, 0, False, str(e))

def grid_search(y: pd.Series, criterion: str = "BIC") -> pd.DataFrame:
    rows = []
    for p in PS:
        for q in QS:
            for (P,Q) in PQS:
                for trend in TRENDS:
                    fr = try_fit(y, (p,0,q), (P,0,Q,S), trend)
                    rows.append({
                        "p": fr.p, "q": fr.q, "P": fr.P, "Q": fr.Q, "s": fr.s,
                        "trend": fr.trend, "AIC": fr.aic, "BIC": fr.bic,
                        "n_params": fr.n_params, "converged": fr.converged, "error": fr.error
                    })
    df = pd.DataFrame(rows)
    key = "BIC" if criterion.upper() == "BIC" else "AIC"
    df = df.sort_values(key, ascending=True, kind="mergesort")  # stable sort
    return df

def main(criterion: str = "BIC"):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    y = load_train()
    grid = grid_search(y, criterion=criterion)
    grid.to_csv(OUT_GRID, index=False, float_format="%.6f")

    # pick best converged and finite
    ok = grid[np.isfinite(grid[criterion.upper()]) & grid["converged"]]
    if ok.empty:
        best = grid.iloc[0].to_dict()
    else:
        best = ok.iloc[0].to_dict()

    # Save best spec
    best_spec = {
        "criterion": criterion.upper(),
        "ARIMA_ORDER": [int(best["p"]), 0, int(best["q"])],
        "SEASONAL_ORDER": [int(best["P"]), 0, int(best["Q"]), int(best["s"])],
        "ARIMA_TREND": str(best["trend"]),
        "AIC": float(best["AIC"]),
        "BIC": float(best["BIC"]),
        "n_params": int(best["n_params"]),
        "converged": bool(best["converged"]),
    }
    Path(OUT_BEST).write_text(json.dumps(best_spec, indent=2))
    print("=== ARIMA/SARIMA grid search (TRAIN) ===")
    print(f"Criterion: {criterion.upper()}")
    print("Top 5:")
    print(grid.head(5))
    print(f"\nSaved grid: {OUT_GRID}")
    print(f"Saved best: {OUT_BEST}")
    print("\nSuggested config.py update:")
    print(f"ARIMA_ORDER = tuple({best_spec['ARIMA_ORDER']})")
    print(f"SEASONAL_ORDER = tuple({best_spec['SEASONAL_ORDER']})")
    print(f"ARIMA_TREND = '{best_spec['ARIMA_TREND']}'")

if __name__ == "__main__":
    # optionally allow: python -m src.tune_arima BIC  (or AIC)
    import sys
    crit = sys.argv[1] if len(sys.argv) > 1 else "BIC"
    main(crit)
