# run.py
"""
Orchestrates the minimal pipeline for M2 monthly growth forecasting.

Core steps:
  1) prepare_data     -> build target g_t and split info
  2) forecast_bench   -> naive & mean, 1-step ahead (test)
  3) forecast_arima   -> ARIMA/SARIMA, rolling 1-step (test)
  4) evaluate         -> RMSFE/MAFE + DM vs Naive

Extras (useful):
  - diagnostics       -> ARIMA residual checks on TRAIN
  - tune              -> ARIMA/SARIMA grid search on TRAIN (AIC/BIC)
  - use-best          -> patch config at runtime with best spec from tuning
  - clean             -> wipe results/ before running

Examples:
  python run.py --all
  python run.py --clean --all
  python run.py --tune --criterion BIC
  python run.py --arima --use-best --eval
  python run.py --diagnostics
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import shutil

# --- make 'src/' importable ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # central config (paths, model params, root paths)


# ---------- helpers ----------
def _apply_best_spec_if_requested(use_best: bool) -> None:
    """
    If --use-best is passed, read results/metrics/best_arima.json (from tune_arima)
    and patch config.ARIMA_ORDER / SEASONAL_ORDER / ARIMA_TREND at runtime.
    """
    if not use_best:
        return
    best_path = config.ROOT / "results" / "metrics" / "best_arima.json"
    if not best_path.exists():
        print(f"[warn] --use-best requested but {best_path} not found. Skipping.")
        return
    best = json.loads(best_path.read_text())
    # Patch config module so downstream imports see updated values
    config.ARIMA_ORDER = tuple(best.get("ARIMA_ORDER", getattr(config, "ARIMA_ORDER", (1, 0, 1))))
    config.SEASONAL_ORDER = tuple(best.get("SEASONAL_ORDER", getattr(config, "SEASONAL_ORDER", (0, 0, 0, 0))))
    config.ARIMA_TREND = best.get("ARIMA_TREND", getattr(config, "ARIMA_TREND", "c"))
    print(f"[info] Using best spec: order={config.ARIMA_ORDER}, seasonal={config.SEASONAL_ORDER}, trend={config.ARIMA_TREND}")


def _clean_results() -> None:
    """Delete files inside results/* (forecasts, metrics, figures) to avoid stale artifacts."""
    base = config.ROOT / "results"
    subdirs = ["forecasts", "metrics", "figures"]
    for sd in subdirs:
        d = base / sd
        if d.exists():
            for p in d.glob("*"):
                try:
                    p.unlink() if p.is_file() else shutil.rmtree(p)
                except Exception as e:
                    print(f"[warn] could not remove {p}: {e}")
    print("[info] results/ cleaned.")


# ---------- step wrappers (late imports so patched config is used) ----------
def step_prepare_data():
    print(">>> Step: prepare_data")
    from prepare_data import main as _main
    _main()

def step_bench():
    print(">>> Step: forecast_bench (naive/mean)")
    from forecast_bench import main as _main
    _main()

def step_arima(use_best: bool):
    print(">>> Step: forecast_arima (rolling 1-step)")
    _apply_best_spec_if_requested(use_best)
    from forecast_arima import main as _main
    _main()

def step_evaluate():
    print(">>> Step: evaluate (RMSFE/MAFE + DM)")
    from evaluate import main as _main
    _main()

def step_diagnostics():
    print(">>> Step: diagnostics (ARIMA train residuals)")
    from diagnostics import main as _main
    _main()

def step_tune(criterion: str):
    print(f">>> Step: tune_arima (criterion={criterion})")
    from tune_arima import main as _main
    _main(criterion)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="M2 Forecasting Pipeline Orchestrator")
    p.add_argument("--all", action="store_true",
                   help="Run full pipeline: prep -> bench -> arima -> eval")
    p.add_argument("--prep", action="store_true", help="Run data preparation")
    p.add_argument("--bench", action="store_true", help="Run benchmarks (naive/mean)")
    p.add_argument("--arima", action="store_true", help="Run ARIMA forecasts (rolling expanding)")
    p.add_argument("--eval", action="store_true", help="Run evaluation (metrics + DM)")
    p.add_argument("--diagnostics", action="store_true", help="Run ARIMA residual diagnostics on TRAIN")
    p.add_argument("--tune", action="store_true", help="Run ARIMA/SARIMA grid search on TRAIN only")
    p.add_argument("--criterion", choices=["AIC", "BIC"], default="BIC",
                   help="Criterion for --tune (default: BIC)")
    p.add_argument("--use-best", action="store_true",
                   help="Before ARIMA, load results/metrics/best_arima.json and patch config")
    p.add_argument("--clean", action="store_true",
                   help="Delete files in results/* before running")
    return p.parse_args()


def main():
    args = parse_args()

    if args.clean:
        _clean_results()

    # default: run --all if no specific flags provided
    if not any([args.all, args.prep, args.bench, args.arima, args.eval, args.diagnostics, args.tune]):
        args.all = True

    try:
        if args.all:
            step_prepare_data()
            step_bench()
            step_arima(use_best=args.use_best)
            step_evaluate()
        else:
            if args.prep: step_prepare_data()
            if args.bench: step_bench()
            if args.tune: step_tune(args.criterion)
            if args.arima: step_arima(use_best=args.use_best)
            if args.eval: step_evaluate()
            if args.diagnostics: step_diagnostics()
    except Exception as e:
        print(f"[error] Pipeline aborted: {e}")
        raise
    else:
        print("\n=== Pipeline completed successfully ===")


if __name__ == "__main__":
    main()
