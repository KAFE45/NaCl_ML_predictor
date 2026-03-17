# =============================================================================
# End-to-End ML Pipeline: NaCl Concentration Prediction
# 1. Data Preparation & Feature Engineering
# 2. Random Forest Model Training
# 3. Evaluation, Visualization & Model Saving
# =============================================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================
def prepare_dataset(input_file):
    print("=" * 60)
    print(" 🛠️ PART 1: DATA PREPARATION & FEATURE ENGINEERING")
    print("=" * 60)
    
    df = pd.read_csv(input_file)

    # -------------------
    # Feature Engineering
    # -------------------
    # สร้างตัวแปรใหม่ (Interaction Term) ช่วยให้โมเดลฉลาดขึ้น
    df["EC_Temp"] = df["EC_M02"] * df["Temp_M02"]

    # -------------------
    # Define X and y
    # -------------------
    y = df["NaCl_Percent"]
    X = df[
        [
            "EC_M02",
            "Temp_M02",
            "Target_Temp",
            "Mercury_Temp",
            "EC_Temp",
        ]
    ]

    # -------------------
    # Train/Test split
    # -------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # -------------------
    # Save dataset
    # -------------------
    input_path = Path(input_file)
    data_dir = input_path.parent.parent

    out_dir = data_dir / "DATA_ANALYSIS"
    out_dir.mkdir(exist_ok=True)

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    print("✅ Data prepared for ML models")
    print(f"   Train size : {len(X_train)} rows")
    print(f"   Test size  : {len(X_test)} rows")
    print(f"   Saved to   : {out_dir}\n")
    
    return out_dir

# =============================================================================
# PART 2: MODEL TRAINING & EVALUATION
# =============================================================================
def train_and_evaluate(data_dir):
    print("=" * 60)
    print(" 🧠 PART 2: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    print("[1/4] Loading prepared data...")
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test  = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test  = pd.read_csv(data_dir / "y_test.csv").values.ravel()

    print("[2/4] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=42, 
        n_jobs=-1 
    )
    rf_model.fit(X_train, y_train)

    print("[3/4] Evaluating model on Test Set...")
    y_pred = rf_model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n   -----------------------------------------")
    print(f"   🏆 R² Score : {r2:.4f}")
    print(f"   🎯 MAE      : {mae:.4f} %NaCl")
    print(f"   📉 RMSE     : {rmse:.4f} %NaCl")
    print(f"   -----------------------------------------\n")

    print("[4/4] Saving model and generating plot...")
    # -- Save Model --
    model_path = data_dir / "rf_nacl_model.joblib"
    joblib.dump(rf_model, model_path)
    
    # -- Visualization --
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    ax.scatter(y_test, y_pred, color="#2563EB", edgecolors="#1E3A8A", alpha=0.75, s=60, zorder=3, label="Predictions")
    
    all_vals = np.concatenate([y_test, y_pred])
    line_min, line_max = all_vals.min() * 0.95, all_vals.max() * 1.05
    ax.plot([line_min, line_max], [line_min, line_max], color="#EF4444", linewidth=2, linestyle="--", zorder=2, label="Perfect Prediction")

    metrics_text = f"R²   = {r2:.4f}\nMAE  = {mae:.4f}\nRMSE = {rmse:.4f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.9), family="monospace")

    ax.set_xlabel("Actual %NaCl", fontsize=12)
    ax.set_ylabel("Predicted %NaCl", fontsize=12)
    ax.set_title("Random Forest — Actual vs. Predicted %NaCl", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    plt.tight_layout()

    plot_path = data_dir / "actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    
    print(f"✅ Pipeline Complete! Outputs saved to: {data_dir}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. กำหนดไฟล์ Input ที่ถูกตัด (CUT) มาแล้ว
    INPUT_FILE =  r"D:\NaCl_ML_predictor\data\RAW_CUT\data_RAW_CUT_20260309_1430.csv"
    
    # 2. รัน Part 1: เตรียมข้อมูลและดึงที่อยู่โฟลเดอร์ออกมา
    DATA_ANALYSIS_DIR = prepare_dataset(INPUT_FILE)
    
    # 3. รัน Part 2: เทรนโมเดล โชว์กราฟ และเซฟไฟล์
    train_and_evaluate(DATA_ANALYSIS_DIR)