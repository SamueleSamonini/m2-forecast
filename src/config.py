# src/config.py
from pathlib import Path

# ---- Paths (relative to repo root) ----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "M2SL.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "m2_growth.parquet"
SPLIT_INFO = ROOT / "data" / "processed" / "split_info.json"

# ---- Reproducibility and split ----
RANDOM_SEED = 42
TEST_FRACTION = 0.20  # holdout = last 20% of observations

# ---- Column names ----
DATE_COL = "observation_date"
LEVEL_COL = "M2SL"
LOG_COL = "M2_log"
TARGET_COL = "g"      # Monthly Growth Rate of M2 = 100 * diff(log(M2))