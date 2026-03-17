# =============================================================================
# predict.py — Inference / Deployment Script
# NaCl Concentration Predictor | Machine Learning + Physical Chemistry
# =============================================================================
# Loads the saved model and predicts %NaCl for new sensor readings.
#
# Usage (single prediction):
#   python predict.py --ec 12.5 --temp 25.0 --target 25.0 --mercury 24.8
#
# Usage (batch CSV):
#   python predict.py --csv "D:\path\to\new_readings.csv"
#
# The CSV must contain columns: EC_M02, Temp_M02, Target_Temp, Mercury_Temp
# EC_Temp is computed automatically (EC_M02 × Temp_M02).
# =============================================================================

import argparse
import pandas as pd
import joblib
from pathlib import Path

from gg.config import (
    MODELS_DIR,
    MODEL_FILENAME,
    FEATURE_COLS,
    RAW_FEATURE_COLS,
)


# =============================================================================
# HELPERS
# =============================================================================

def load_model():
    model_path = MODELS_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run train_model.py first to generate it."
        )
    model = joblib.load(model_path)
    print(f"  Model loaded from: {model_path}")
    return model


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the EC_Temp engineered feature and return columns in the correct order.
    Input df must contain: EC_M02, Temp_M02, Target_Temp, Mercury_Temp.
    """
    missing = [c for c in RAW_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    df = df.copy()
    df["EC_Temp"] = df["EC_M02"] * df["Temp_M02"]
    return df[FEATURE_COLS]


def predict_single(model, ec: float, temp: float,
                   target_temp: float, mercury_temp: float) -> float:
    """Predict %NaCl for a single set of sensor readings."""
    row = pd.DataFrame([{
        "EC_M02"      : ec,
        "Temp_M02"    : temp,
        "Target_Temp" : target_temp,
        "Mercury_Temp": mercury_temp,
    }])
    X = build_features(row)
    return model.predict(X)[0]


def predict_batch(model, csv_path: Path) -> pd.DataFrame:
    """Predict %NaCl for all rows in a CSV file."""
    df_raw = pd.read_csv(csv_path)
    X = build_features(df_raw)
    df_raw["NaCl_Predicted"] = model.predict(X)
    return df_raw


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict %%NaCl concentration from sensor readings."
    )
    # Single-prediction arguments
    parser.add_argument("--ec",      type=float, help="Electrical Conductivity (EC_M02)")
    parser.add_argument("--temp",    type=float, help="Solution Temperature (Temp_M02, °C)")
    parser.add_argument("--target",  type=float, help="Target Temperature (Target_Temp, °C)")
    parser.add_argument("--mercury", type=float, help="Mercury Thermometer reading (°C)")
    # Batch argument
    parser.add_argument("--csv",     type=Path,  help="Path to CSV with new readings")

    args = parser.parse_args()

    print("=" * 55)
    print("  NaCl Predictor — Inference")
    print("=" * 55)

    model = load_model()

    # --- BATCH MODE ---
    if args.csv:
        print(f"\n  Batch mode: {args.csv}")
        results = predict_batch(model, args.csv)
        out_path = args.csv.parent / (args.csv.stem + "_predictions.csv")
        results.to_csv(out_path, index=False)
        print(f"\n  Predictions saved → {out_path}")
        print(results[["EC_M02", "Temp_M02", "NaCl_Predicted"]].to_string(index=False))

    # --- SINGLE MODE ---
    elif all(v is not None for v in [args.ec, args.temp, args.target, args.mercury]):
        result = predict_single(model, args.ec, args.temp, args.target, args.mercury)
        print(f"\n  Input readings:")
        print(f"    EC_M02       : {args.ec}")
        print(f"    Temp_M02     : {args.temp} °C")
        print(f"    Target_Temp  : {args.target} °C")
        print(f"    Mercury_Temp : {args.mercury} °C")
        print(f"    EC_Temp      : {args.ec * args.temp:.4f}  (engineered)")
        print(f"\n  ➜  Predicted %NaCl : {result:.4f} %")

    else:
        parser.print_help()

    print("\n" + "=" * 55)
