# run.py
"""
Orchestrates the full pipeline:
prepare_data -> benchmark -> arima -> evaluate -> plots -> report tables
"""

import sys
from pathlib import Path

# Ensure we can import modules from ./src
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import pipeline steps
from prepare_data import main as prep_main
from forecast_bench import main as bench_main
from forecast_arima import main as arima_main
from evaluate import main as eval_main
from plotting import make_all_plots
from report_tables import main as tables_main


def main():
    print(">>> Step 1/6: prepare_data")
    prep_main()

    print("\n>>> Step 2/6: forecast_bench (naive/mean)")
    bench_main()

    print("\n>>> Step 3/6: forecast_arima (rolling 1-step)")
    arima_main()

    print("\n>>> Step 4/6: evaluate (RMSFE/MAFE + files)")
    eval_main()

    print("\n>>> Step 5/6: plotting (figures)")
    make_all_plots()

    print("\n>>> Step 6/6: report tables (MD/LaTeX)")
    tables_main()

    print("\n=== Pipeline completed ===")


if __name__ == "__main__":
    main()
