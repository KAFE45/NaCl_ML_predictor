# =============================================================================
# prepare_dataset.py — Data Preparation & Train/Test Split
# NaCl Concentration Predictor | Machine Learning + Physical Chemistry
# =============================================================================
# Reads raw sensor data, engineers features, splits into train/test sets,
# and saves the 4 CSV files consumed by train_model.py.
#
# Usage:
#   python prepare_dataset.py
#   python prepare_dataset.py --input "D:\path\to\other_file.csv"
# =============================================================================

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import all settings from central config
from gg.config import (
    RAW_FILE,
    DATA_ANALYSIS_DIR,
    FEATURE_COLS,
    ENGINEERED_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
)


# =============================================================================
# CORE FUNCTION
# =============================================================================

def prepare_dataset(input_file: Path) -> None:
    """
    Full pipeline: load raw CSV → engineer features → split → save CSVs.

    Parameters
    ----------
    input_file : Path
        Path to the raw sensor CSV file.
    """

    print("=" * 55)
    print("  Data Preparation — NaCl ML Predictor")
    print("=" * 55)

    # -------------------------------------------------------------------------
    # 1. LOAD RAW DATA
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Loading raw data from:\n  {input_file}\n")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    print(f"  Rows loaded    : {len(df):,}")
    print(f"  Columns found  : {list(df.columns)}")

    # Basic sanity check — ensure required raw columns exist
    required_raw = ["EC_M02", "Temp_M02", "Target_Temp", "Mercury_Temp", TARGET_COL]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    # -------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n[2/4] Engineering features...")

    # Interaction term: captures the joint effect of conductivity and temperature.
    # Electrical conductivity is temperature-dependent; this product helps the
    # model learn that relationship directly.
    df["EC_Temp"] = df["EC_M02"] * df["Temp_M02"]
    print(f"  Created 'EC_Temp' = EC_M02 × Temp_M02  (interaction term)")

    # -------------------------------------------------------------------------
    # 3. DEFINE X AND y
    # -------------------------------------------------------------------------
    print("\n[3/4] Splitting features and target...")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"  Features (X) : {FEATURE_COLS}")
    print(f"  Target   (y) : {TARGET_COL}")
    print(f"  NaCl range   : {y.min():.3f}% – {y.max():.3f}%")

    # -------------------------------------------------------------------------
    # 4. TRAIN / TEST SPLIT & SAVE
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Splitting (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    DATA_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(DATA_ANALYSIS_DIR / "X_train.csv", index=False)
    X_test.to_csv( DATA_ANALYSIS_DIR / "X_test.csv",  index=False)
    y_train.to_csv(DATA_ANALYSIS_DIR / "y_train.csv", index=False)
    y_test.to_csv( DATA_ANALYSIS_DIR / "y_test.csv",  index=False)

    print(f"\n  Train samples  : {len(X_train):,}")
    print(f"  Test  samples  : {len(X_test):,}")
    print(f"\n  Saved to: {DATA_ANALYSIS_DIR}")
    print("    X_train.csv  X_test.csv  y_train.csv  y_test.csv")

    print("\n" + "=" * 55)
    print("  Data preparation complete.")
    print("=" * 55)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare NaCl dataset for ML training."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_FILE,
        help="Path to raw sensor CSV (default: from config.py)",
    )
    args = parser.parse_args()

    prepare_dataset(args.input)
