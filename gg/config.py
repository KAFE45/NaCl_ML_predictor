# =============================================================================
# config.py — Central Project Configuration
# NaCl Concentration Predictor | Machine Learning + Physical Chemistry
# =============================================================================
# All paths, feature definitions, and model hyperparameters live here.
# Edit this file to adapt the project to new data or experiments.
# =============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------------------------------

# Root data directory
DATA_ROOT = Path(r"D:\NaCl_ML_predictor\data")

# Raw input CSV (latest acquisition file)
RAW_DIR  = DATA_ROOT / "RAW_CUT"
RAW_FILE = RAW_DIR / "data_RAW_CUT_20260309_1430.csv"

# Processed split data (output of prepare_dataset.py)
DATA_ANALYSIS_DIR = DATA_ROOT / "DATA_ANALYSIS"

# Model artefacts and results (output of train_model.py)
MODELS_DIR   = DATA_ROOT / "MODELS"
RESULTS_DIR  = DATA_ROOT / "RESULTS"

# -----------------------------------------------------------------------------
# COLUMN NAMES
# -----------------------------------------------------------------------------

# Raw sensor columns used as features
RAW_FEATURE_COLS = [
    "EC_M02",        # Electrical Conductivity (mS/cm or S/m)
    "Temp_M02",      # Solution Temperature (°C)
    "Target_Temp",   # Target chamber temperature setpoint (°C)
    "Mercury_Temp",  # Mercury thermometer reference reading (°C)
]

# Engineered features (created in prepare_dataset.py)
ENGINEERED_COLS = [
    "EC_Temp",       # Interaction term: EC_M02 × Temp_M02
]

# Full ordered feature list fed to the model
FEATURE_COLS = RAW_FEATURE_COLS + ENGINEERED_COLS

# Target variable
TARGET_COL = "NaCl_Percent"

# -----------------------------------------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------

TEST_SIZE    = 0.20   # 20% held out for testing
RANDOM_STATE = 42     # Fixed seed for reproducibility

# -----------------------------------------------------------------------------
# RANDOM FOREST HYPERPARAMETERS
# -----------------------------------------------------------------------------

RF_PARAMS = {
    "n_estimators"    : 200,    # Number of trees
    "max_depth"       : None,   # None = grow until leaves are pure
    "min_samples_split": 2,     # Min samples to split an internal node
    "min_samples_leaf" : 1,     # Min samples at a leaf node
    "random_state"    : RANDOM_STATE,
    "n_jobs"          : -1,     # Use all CPU cores
}

# -----------------------------------------------------------------------------
# OUTPUT FILE NAMES
# -----------------------------------------------------------------------------

MODEL_FILENAME      = "rf_nacl_model.joblib"
PLOT_FILENAME       = "actual_vs_predicted.png"
IMPORTANCE_FILENAME = "feature_importances.png"
METRICS_FILENAME    = "metrics.txt"
