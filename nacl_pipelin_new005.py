# =============================================================================
# nacl_pipeline.py — End-to-End ML Pipeline: NaCl Concentration Prediction
# =============================================================================
# ขั้นตอนทั้งหมดในไฟล์เดียว:
#   PART 1 : Data Preparation & Feature Engineering
#   PART 2 : Random Forest Training & Evaluation
#   PART 3 : Visualisation (Scatter + Feature Importances)
#   PART 4 : Model Saving + Inference Example
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from sklearn.linear_model import LinearRegression
# =============================================================================
# ⚙️  CONFIGURATION — แก้ไขที่นี่ที่เดียว
# =============================================================================

INPUT_FILE = Path(r"D:\NaCl_ML_predictor\data\MERGED\data_MERGED_20260317_1355.csv")

# Feature columns (ต้องตรงกับที่อยู่ใน CSV ต้นฉบับ + EC_Temp ที่สร้างใหม่)
# FEATURE_COLS = [
#     "EC_M02",        # Electrical Conductivity (ค่าการนำไฟฟ้าที่ได้จากเซนเซอร์)
#     "EC_25C",      # 🔧 Engineered: EC_M02 ที่ชดเชยอุณหภูมิแล้ว (Temperature Compensation) ตามมาตรฐาน IEC 60746
#     "Temp_M02_C",      # Solution Temperature — °C (อุณหภูมิสารละลายที่ได้จากเซนเซอร์)
#     "Mercury_Temp",  # Mercury thermometer reference — °C (อุณหภูมิที่ได้จากปรอทซึ่งเป็นอุณภูมิจริงของ NaCl_Percent ที่แม่นยำกว่าเซนเซอร์)
#     "EC_TrueTemp",   # 🔧 Engineered: EC_M02 × Mercury_Temp (interaction term ที่ใช้ปรอทแทนเซนเซอร์)
#     "Temp_Error",    # 🔧 Engineered: Temp_M02_C - Mercury_Temp (เซนเซอร์เพี้ยนจากความจริงแค่ไหน)
# ]

FEATURE_COLS = [
    "EC_M01", "EC_M02", "EC_M03", 
    "EC_25C_M01", "EC_25C_M02", "EC_25C_M03",
    "Temp_M01_C", "Temp_M02_C", "Temp_M03_C",
    "Mercury_Temp",
    "EC_TrueTemp_M01", "EC_TrueTemp_M02", "EC_TrueTemp_M03",
    "Temp_Error_M01",  "Temp_Error_M02",  "Temp_Error_M03",
]

TARGET_COL   = "NaCl_Percent"   # ตัวแปรเป้าหมาย: ความเข้มข้น %NaCl
TEST_SIZE    = 0.20             # 20% สำหรับ Test set
RANDOM_STATE = 42               # Seed ตายตัว — ผลลัพธ์ reproducible

# Random Forest hyperparameters
RF_PARAMS = dict(
    n_estimators    = 200,   # จำนวนต้นไม้ในป่า
    max_depth       = None,  # None = ต้นไม้เติบโตจนใบบริสุทธิ์
    min_samples_split = 2,
    min_samples_leaf  = 1,
    random_state    = RANDOM_STATE,
    n_jobs          = -1,    # ใช้ทุก CPU core
)

# =============================================================================
# PART 1: DATA PREPARATION & FEATURE ENGINEERING
# =============================================================================
 # --- Feature Engineering ---
    # โมเดลยึดอุณหภูมิจริง (Mercury_Temp) เป็นหลัก แต่ก็อยากให้โมเดลเรียนรู้ "ความเพี้ยน" ของเซนเซอร์ (Temp_M02) ไปด้วย
    # EC_Temp = EC × Temp ช่วยให้โมเดลจับความสัมพันธ์ร่วม (interaction) ระหว่างค่าการนำไฟฟ้าและอุณหภูมิ ซึ่งมีความสัมพันธ์กันทางฟิสิกส์เคมีของสารละลาย NaCl
    # 1. Physics-based Interaction: ใช้ค่า EC คู่อุณหภูมิจริงจากปรอท (แม่นยำที่สุด)
    # เพื่อให้โมเดลจับความสัมพันธ์ของความเข้มข้นสารละลายบนพื้นฐานความจริง
      # 2. Sensor Bias/Error: สร้างตัวแปรส่วนต่าง เพื่อสอนโมเดลว่าเซนเซอร์เพี้ยนจากความจริงแค่ไหน
    # โมเดลจะใช้ค่านี้เพื่อเรียนรู้พฤติกรรม (Pattern) ของเซนเซอร์ในสภาวะต่างๆ
    #Load CSV → Validate raw cols → Engineer features → Check NaN → Validate FEATURE_COLS → Split → Save 

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ALPHA = 0.02  # จะอัปเดตหลัง find_true_alpha()

    for m in ["M01", "M02", "M03"]:
        ec_col   = f"EC_{m}"
        temp_col = f"Temp_{m}"

        # แปลง RAW → °C (เฉพาะที่มีข้อมูล)
        if temp_col in df.columns:
            df[f"Temp_{m}_C"] = df[temp_col] / 100
        else:
            df[f"Temp_{m}_C"] = float("nan")

        # EC_25C per sensor
        if ec_col in df.columns and "Mercury_Temp" in df.columns:
            df[f"EC_25C_{m}"] = df[ec_col] / (
                1 + ALPHA * (df["Mercury_Temp"] - 25)
            )
            df[f"EC_TrueTemp_{m}"] = df[f"EC_25C_{m}"] * df["Mercury_Temp"]
            df[f"Temp_Error_{m}"]  = df[f"Temp_{m}_C"] - df["Mercury_Temp"]
        else:
            df[f"EC_25C_{m}"]     = float("nan")
            df[f"EC_TrueTemp_{m}"]= float("nan")
            df[f"Temp_Error_{m}"] = float("nan")

    return df
    

def prepare_dataset(input_file: Path) -> tuple:
    print("=" * 62)
    print("  🛠️  PART 1: DATA PREPARATION & FEATURE ENGINEERING")
    print("=" * 62)

    if not input_file.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ input: {input_file}")

    df = pd.read_csv(input_file)
    print(f"\n  ✔ โหลดข้อมูลสำเร็จ : {len(df):,} แถว, {df.shape[1]} คอลัมน์")
    print("\n  📊 RAW DATA SAMPLE (5 แถวแรก):")
    print(df[["EC_M02", "Temp_M02", "Mercury_Temp", TARGET_COL]].head())
    print("\n  📊 RAW DATA STATS:")
    print(df[["EC_M02", "Temp_M02", "Mercury_Temp"]].describe())

    # เพิ่มใน describe() ชั่วคราว
    print(df[["EC_M02", "Mercury_Temp", "NaCl_Percent"]].corr())

    # 1. ตรวจสอบคอลัมน์ดิบ
    RAW_REQUIRED = ["Mercury_Temp", TARGET_COL]

    # เพิ่มตรวจสอบว่ามี sensor อย่างน้อย 1 ตัว
    SENSOR_COLS = ["EC_M01", "EC_M02", "EC_M03"]
    available = [c for c in SENSOR_COLS if c in df.columns]
    if not available:
        raise ValueError(f"ต้องมีข้อมูลอย่างน้อย 1 sensor: {SENSOR_COLS}")
    print(f"  ✔ Sensors ที่พบ: {available}")
    missing_raw = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing_raw:
        raise ValueError(f"ขาดคอลัมน์ดิบที่จำเป็น: {missing_raw}")

    # 2. ลบแถว NaN ใน raw columns
    n_before = len(df)
    df = df.dropna(subset=RAW_REQUIRED)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  ⚠ ลบ {n_dropped} แถวที่มีค่า NaN (จาก {n_before:,} → {len(df):,} แถว)")

    # 3. Feature Engineering
    df = engineer_features(df)
    print(f"  ✔ สร้าง EC_TrueTemp = EC_M02 × Mercury_Temp")
    print(f"  ✔ สร้าง Temp_Error  = Temp_M02 - Mercury_Temp")

    # 4. ตรวจสอบ NaN ในฟีเจอร์ที่สร้างใหม่ (safety net)
    engineered = [c for c in FEATURE_COLS if c in df.columns]
    nan_counts = df[engineered].isna().sum()
    # ไม่ raise error เพราะ NaN เป็นเรื่องปกติเมื่อบางแถวไม่มี M01 หรือ M03
    if nan_counts.any():
        print(f"  ℹ NaN ใน engineered features (ปกติสำหรับแถวที่ไม่มีบาง sensor):")
        print(f"  {nan_counts[nan_counts > 0].to_dict()}")

    # 5. ตรวจสอบ FEATURE_COLS ครบทั้งหมด
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"คอลัมน์ขาดหายไป (หลังสร้าง Feature): {missing}")

    # 6. กำหนด X และ y
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    print(f"  ✔ ช่วง %NaCl : {y.min():.3f}% – {y.max():.3f}%")

    # 7. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 8. บันทึก CSV
    out_dir = input_file.parent.parent / "DATA_ANALYSIS"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv( out_dir / "X_test.csv",  index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv( out_dir / "y_test.csv",  index=False)

    print(f"\n  ✔ Train : {len(X_train):,} samples")
    print(f"  ✔ Test  : {len(X_test):,} samples  (test_size={TEST_SIZE})")
    print(f"  ✔ บันทึกไปที่ : {out_dir}")

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel(), out_dir, df


# =============================================================================
# PART 2: MODEL TRAINING & EVALUATION
# =============================================================================
def train_temperature_split_models(X_train, X_test, y_train, y_test, df_full):
    """
    เทรนโมเดลแยกตามช่วงอุณหภูมิ
    """
    TEMP_THRESHOLD = 50.0  # °C

    # แยก train/test ตาม Mercury_Temp
    idx_train_low  = X_train["Mercury_Temp"] <= TEMP_THRESHOLD
    idx_train_high = X_train["Mercury_Temp"] >  TEMP_THRESHOLD
    idx_test_low   = X_test["Mercury_Temp"]  <= TEMP_THRESHOLD
    idx_test_high  = X_test["Mercury_Temp"]  >  TEMP_THRESHOLD

    print(f"\n  🌡️  แบ่งข้อมูลตามอุณหภูมิ (threshold = {TEMP_THRESHOLD}°C):")
    print(f"  Train low  (≤{TEMP_THRESHOLD}°C): {idx_train_low.sum():,} แถว")
    print(f"  Train high (>{TEMP_THRESHOLD}°C): {idx_train_high.sum():,} แถว")
    print(f"  Test  low  (≤{TEMP_THRESHOLD}°C): {idx_test_low.sum():,} แถว")
    print(f"  Test  high (>{TEMP_THRESHOLD}°C): {idx_test_high.sum():,} แถว")

    # เทรน Model A — low temp
    model_low = RandomForestRegressor(**RF_PARAMS)
    model_low.fit(X_train[idx_train_low], y_train[idx_train_low])

    # เทรน Model B — high temp
    model_high = RandomForestRegressor(**RF_PARAMS)
    model_high.fit(X_train[idx_train_high], y_train[idx_train_high])

    # ทำนายโดยเลือก model ตาม temp
    y_pred = np.where(
        idx_test_high,
        model_high.predict(X_test),
        model_low.predict(X_test)
    )

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  📊 ผลรวม (split model):")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    # แสดงผลแยกตามช่วง
    for label, idx_test, model in [
        (f"≤{TEMP_THRESHOLD}°C", idx_test_low,  model_low),
        (f">{TEMP_THRESHOLD}°C", idx_test_high, model_high),
    ]:
        if idx_test.sum() == 0:
            continue
        yp = model.predict(X_test[idx_test])
        yt = y_test[idx_test]
        print(f"\n  [{label}] n={idx_test.sum()}")
        print(f"  R²={r2_score(yt,yp):.4f}  MAE={mean_absolute_error(yt,yp):.4f}")

    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}
    importances = pd.Series(model_high.feature_importances_, index=FEATURE_COLS)

    return model_low, model_high, y_pred, metrics, importances

def train_calibration_model(df: pd.DataFrame, out_dir: Path) -> Path:
    print("\n" + "=" * 62)
    print("  🌡️  STAGE 1: TRAINING CALIBRATION MODEL (Sensor -> True Temp)")
    print("=" * 62)

    calib_features = ["Temp_M02_C", "EC_25C_M02"]
    calib_target   = "Mercury_Temp"

    # ✅ ลบแถวที่มี NaN ใน calib features ก่อน fit
    df_calib = df[calib_features + [calib_target]].dropna()
    n_dropped = len(df) - len(df_calib)
    if n_dropped > 0:
        print(f"  ⚠ ลบ {n_dropped} แถวที่มี NaN ใน calib features")

    X_calib = df_calib[calib_features]
    y_calib = df_calib[calib_target]

    calib_model = LinearRegression()
    calib_model.fit(X_calib, y_calib)

    y_pred = calib_model.predict(X_calib)
    mae  = mean_absolute_error(y_calib, y_pred)
    rmse = np.sqrt(mean_squared_error(y_calib, y_pred))

    print(f"  ✔ เทรนด้วย {len(df_calib):,} แถว")
    print(f"  ✔ สมการชดเชย: True_Temp = ({calib_model.coef_[0]:.4f} × Temp_M02_C) "
          f"+ ({calib_model.coef_[1]:.4f} × EC_25C_M02) + {calib_model.intercept_:.4f}")
    print(f"  ✔ MAE  : {mae:.3f} °C")
    print(f"  ✔ RMSE : {rmse:.3f} °C")

    calib_model_path = out_dir / "calib_temp_model.joblib"
    joblib.dump(calib_model, calib_model_path)
    print(f"\n  ✔ บันทึก Calibration Model ไปที่: {calib_model_path.name}")

    return calib_model_path


# =============================================================================
# PART 3: VISUALISATION
# =============================================================================

def visualise(y_test, y_pred, metrics: dict,
              importances: pd.Series, out_dir: Path) -> None:
    """
    สร้าง 2 กราฟใน Figure เดียว:
      • (ซ้าย)  Scatter: Actual vs Predicted + perfect-prediction line
      • (ขวา)  Horizontal bar: Feature Importances
    """
    print("\n" + "=" * 62)
    print("  📊 PART 3: VISUALISATION")
    print("=" * 62)

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#F7F9FC")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── กราฟซ้าย: Actual vs Predicted ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#F7F9FC")

    ax1.scatter(
        y_test, y_pred,
        color="#2563EB", edgecolors="#1E3A8A",
        alpha=0.75, s=60, zorder=3, label="Test Predictions"
    )

    all_vals = np.concatenate([y_test, y_pred])
    pad = (all_vals.max() - all_vals.min()) * 0.05
    lmin, lmax = all_vals.min() - pad, all_vals.max() + pad

    ax1.plot(
        [lmin, lmax], [lmin, lmax],
        color="#EF4444", linewidth=2, linestyle="--",
        zorder=2, label="Perfect Prediction (y = x)"
    )

    ann = (f"R²   = {metrics['R2']:.4f}\n"
           f"MAE  = {metrics['MAE']:.4f}\n"
           f"RMSE = {metrics['RMSE']:.4f}")
    ax1.text(
        0.05, 0.95, ann, transform=ax1.transAxes,
        fontsize=10, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.9)
    )

    ax1.set_xlabel("Actual %NaCl",    fontsize=12, labelpad=8)
    ax1.set_ylabel("Predicted %NaCl", fontsize=12, labelpad=8)
    ax1.set_title("Actual vs. Predicted %NaCl",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.set_xlim(lmin, lmax)
    ax1.set_ylim(lmin, lmax)

    # ── กราฟขวา: Feature Importances ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#F7F9FC")

    sorted_imp = importances.sort_values()
    colors = ["#2563EB" if v == sorted_imp.max() else "#93C5FD"
              for v in sorted_imp]

    bars = ax2.barh(sorted_imp.index, sorted_imp.values,
                    color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, sorted_imp.values):
        ax2.text(val + 0.004,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", ha="left",
                 fontsize=9, color="#1E3A8A")

    ax2.set_xlabel("Importance Score", fontsize=11)
    ax2.set_title("Feature Importances",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(0, sorted_imp.max() * 1.22)
    ax2.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Random Forest — NaCl Concentration Predictor 005 ",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()

    plot_path = out_dir / f"rf_results005_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  ✔ กราฟบันทึกที่ : {plot_path}")


# =============================================================================
# PART 4: SAVE MODEL & INFERENCE EXAMPLE
# =============================================================================

def save_model(model, out_dir: Path) -> Path:
    """บันทึกโมเดลด้วย joblib"""
    model_path = out_dir / "rf_nacl_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n  ✔ โมเดลบันทึกที่ : {model_path}")
    return model_path

def inference_example(model_low_path: Path, model_high_path: Path,
                      calib_model_path: Path) -> None:
    """
    ตัวอย่าง Deployment แบบ Temperature-Split Model
    """
    print("\n  --- ตัวอย่าง Deployment (Temperature-Split Model) ---")

    model_low      = joblib.load(model_low_path)
    model_high     = joblib.load(model_high_path)
    loaded_calib   = joblib.load(calib_model_path)

    TEMP_THRESHOLD = 50.0

    # 1. รับค่า RAW จาก sensor
    sensor_ec   = 1201
    sensor_temp = 6214

    # 2. แปลง RAW → °C
    temp_c = sensor_temp / 100

    # 3. คำนวณ EC_25C เบื้องต้น
    ec_25c_approx = sensor_ec / (1 + 0.02 * (temp_c - 25))
    calib_input   = pd.DataFrame([{
        "Temp_M02_C": temp_c,
        "EC_25C_M02": ec_25c_approx
    }])
    predicted_mercury = loaded_calib.predict(calib_input)[0]

    # 4. เตรียม sample_raw
    sample_raw = pd.DataFrame([{
        "EC_M01"      : float("nan"),
        "EC_M02"      : sensor_ec,
        "EC_M03"      : float("nan"),
        "Temp_M01"    : float("nan"),
        "Temp_M02"    : sensor_temp,
        "Temp_M03"    : float("nan"),
        "Mercury_Temp": predicted_mercury,
    }])

    # 5. สร้าง Engineered Features
    sample = engineer_features(sample_raw)
    sample = sample[FEATURE_COLS]

    # 6. เลือกโมเดลตาม predicted_mercury
    if predicted_mercury > TEMP_THRESHOLD:
        prediction = model_high.predict(sample)[0]
        print(f"  ใช้ Model HIGH temp (>{TEMP_THRESHOLD}°C)")
    else:
        prediction = model_low.predict(sample)[0]
        print(f"  ใช้ Model LOW temp (≤{TEMP_THRESHOLD}°C)")

    print(f"  [Calibration] Temp RAW={sensor_temp} → {temp_c:.2f}°C → ชดเชยเป็น {predicted_mercury:.2f}°C")
    print(f"  Input  : EC_RAW={sensor_ec}, Temp_RAW={sensor_temp} ({temp_c:.2f}°C)")
    print(f"  Output : %NaCl Predicted = {prediction:.4f} %")


# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, DATA_ANALYSIS_DIR, df_full = prepare_dataset(INPUT_FILE)
    calib_model_path = train_calibration_model(df_full, DATA_ANALYSIS_DIR)

    model_low, model_high, y_pred, metrics, importances = train_temperature_split_models(
        X_train, X_test, y_train, y_test, df_full
    )

    visualise(y_test, y_pred, metrics, importances, DATA_ANALYSIS_DIR)

    # บันทึกโมเดล
    model_low_path  = DATA_ANALYSIS_DIR / "rf_model_low_temp.joblib"
    model_high_path = DATA_ANALYSIS_DIR / "rf_model_high_temp.joblib"
    joblib.dump(model_low,  model_low_path)
    joblib.dump(model_high, model_high_path)
    print(f"  ✔ บันทึก rf_model_low_temp.joblib")
    print(f"  ✔ บันทึก rf_model_high_temp.joblib")

    print("\n" + "=" * 62)
    print("  💾 PART 4: SAVE MODEL & INFERENCE EXAMPLE")
    print("=" * 62)
    inference_example(model_low_path, model_high_path, calib_model_path)

    print("\n" + "=" * 62)
    print("  ✅ Pipeline สมบูรณ์! ผลลัพธ์ทั้งหมดอยู่ใน:")
    print(f"     {DATA_ANALYSIS_DIR}")
    print("     rf_results.png  |  rf_model_low_temp.joblib  |  rf_model_high_temp.joblib")
    print("=" * 62)

# df = pd.read_csv(r"D:\NaCl_ML_predictor\data\MERGED\data_MERGED_20260317_1355.csv")
# print("📊 จำนวนข้อมูลแยกตาม NaCl × Temp (Mercury_Temp > 50°C):")
# high = df[df["Mercury_Temp"] > 50]
# print(f"  แถวทั้งหมดที่ Temp > 50°C : {len(high):,} แถว")
# print(f"  แถวทั้งหมดที่ Temp ≤ 50°C : {len(df) - len(high):,} แถว")

# print("\n📊 NaCl distribution ใน high temp zone:")
# print(high["NaCl_Percent"].value_counts().sort_index())

# print("\n📊 EC_M02 ที่ NaCl=1.9% และ Temp>50°C:")
# subset = df[(df["NaCl_Percent"] == 1.9) & (df["Mercury_Temp"] > 50)]
# print(subset[["Mercury_Temp", "EC_M02", "NaCl_Percent"]].to_string())

# print("\n📊 EC_M02 ที่ใกล้เคียง sensor_ec=1276 และ Temp>50°C:")
# subset2 = df[(df["EC_M02"].between(1200, 1350)) & (df["Mercury_Temp"] > 50)]
# print(subset2[["Mercury_Temp", "EC_M02", "NaCl_Percent"]].sort_values("Mercury_Temp").to_string())
