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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from sklearn.linear_model import LinearRegression

# =============================================================================
# ⚙️  CONFIGURATION — แก้ไขที่นี่ที่เดียว
# =============================================================================

INPUT_FILE = Path(r"D:\NaCl_ML_predictor\data\RAW_CUT\data_RAW_CUT_20260313_0924.csv")

# Feature columns (ต้องตรงกับที่อยู่ใน CSV ต้นฉบับ + EC_Temp ที่สร้างใหม่)
FEATURE_COLS = [
    "EC_M02",        # Electrical Conductivity (ค่าการนำไฟฟ้าที่ได้จากเซนเซอร์)
    "EC_25C",      # 🔧 Engineered: EC_M02 ที่ชดเชยอุณหภูมิแล้ว (Temperature Compensation) ตามมาตรฐาน IEC 60746
    "Temp_M02_C",      # Solution Temperature — °C (อุณหภูมิสารละลายที่ได้จากเซนเซอร์)
    "Mercury_Temp",  # Mercury thermometer reference — °C (อุณหภูมิที่ได้จากปรอทซึ่งเป็นอุณภูมิจริงของ NaCl_Percent ที่แม่นยำกว่าเซนเซอร์)
    "EC_TrueTemp",   # 🔧 Engineered: EC_M02 × Mercury_Temp (interaction term ที่ใช้ปรอทแทนเซนเซอร์)
    "Temp_Error",    # 🔧 Engineered: Temp_M02_C - Mercury_Temp (เซนเซอร์เพี้ยนจากความจริงแค่ไหน)
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
    """
    Stateless, row-wise transformations only.
    Safe to call before split or at inference time.
    """
    df = df.copy()

    # แปลง RAW → หน่วยจริง
    df["Temp_M02_C"] = df["Temp_M02"] / 100

    # ─────────────────────────────────────────────────────────
    # 🔑 Temperature Compensation (IEC 60746 standard)
    # EC_25C = EC ที่ normalize แล้วให้อ้างอิงที่ 25°C เสมอ
    # ทำให้โมเดลเห็น "EC จริงที่ไม่ขึ้นกับอุณหภูมิ" → ตรงกับ %NaCl โดยตรง
    # α = 0.02 (2% per °C) คือค่ามาตรฐานของ NaCl solution
    ALPHA = 0.02
    df["EC_25C"] = df["EC_M02"] / (1 + ALPHA * (df["Mercury_Temp"] - 25))

    # Physics interaction: ใช้ EC ที่ชดเชยแล้ว × อุณหภูมิจริง
    df["EC_TrueTemp"] = df["EC_25C"] * df["Mercury_Temp"]

    # Sensor bias: เปรียบเทียบใน scale เดียวกัน
    df["Temp_Error"] = df["Temp_M02_C"] - df["Mercury_Temp"]

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
    RAW_REQUIRED = ["EC_M02", "Temp_M02", "Mercury_Temp", TARGET_COL]
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
    nan_counts = df[["EC_TrueTemp", "Temp_Error"]].isna().sum()
    if nan_counts.any():
        raise ValueError(f"พบค่า NaN ในฟีเจอร์ที่สร้างขึ้นใหม่:\n{nan_counts[nan_counts > 0]}")

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

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel(), out_dir


# =============================================================================
# PART 2: MODEL TRAINING & EVALUATION
# =============================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    เทรน Random Forest → คำนวณ R², MAE, RMSE → คืนค่าโมเดลและผลลัพธ์
    """
    print("\n" + "=" * 62)
    print("  🧠 PART 2: MODEL TRAINING & EVALUATION")
    print("=" * 62)

    print(f"\n  เทรน RandomForestRegressor (n_estimators={RF_PARAMS['n_estimators']})...")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    print("  ✔ เทรนเสร็จสิ้น")

    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}

    print(f"\n  {'─'*44}")
    print(f"  {'Metric':<38} {'Value':>6}")
    print(f"  {'─'*44}")
    print(f"  {'🏆  R²  (Coefficient of Determination)':<38} {r2:>6.4f}")
    print(f"  {'🎯  MAE (Mean Absolute Error)  [%NaCl]':<38} {mae:>6.4f}")
    print(f"  {'📉  RMSE (Root Mean Squared Error) [%NaCl]':<38} {rmse:>6.4f}")
    print(f"  {'─'*44}")

    print("\n  Feature Importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    for feat, imp in importances.sort_values(ascending=False).items():
        bar = "█" * int(imp * 45)
        print(f"    {feat:<15} {imp:.4f}  {bar}")

    return model, y_pred, metrics, importances

def train_calibration_model(X_train: pd.DataFrame, out_dir: Path) -> Path:
    """
    STAGE 1: เทรนโมเดลสำหรับชดเชยความเพี้ยนของเซนเซอร์อุณหภูมิ (Calibration)
    เรียนรู้จากข้อมูล Training set เพื่อแปลง (Temp_M02, EC_M02) -> Mercury_Temp
    """
    print("\n" + "=" * 62)
    print("  🌡️  STAGE 1: TRAINING CALIBRATION MODEL (Sensor -> True Temp)")
    print("=" * 62)

    # 1. เลือก Features ที่หน้างานจริงมีให้ใช้ และ Target (อุณหภูมิจริงที่อยากได้)
    # *ต้องแน่ใจว่า X_train ยังมีคอลัมน์ Mercury_Temp อยู่ ณ จุดนี้
    calib_features = ["Temp_M02_C", "EC_25C"]  # ← แทน EC_M02
    calib_target = "Mercury_Temp"

    X_calib = X_train[calib_features]
    y_calib = X_train[calib_target]

    # 2. สร้างและเทรนโมเดล (ใช้ Linear Regression เหมาะสมที่สุดสำหรับ Sensor Drift)
    calib_model = LinearRegression()
    calib_model.fit(X_calib, y_calib)

    # 3. ประเมินความแม่นยำของ Calibration Model
    y_pred = calib_model.predict(X_calib)
    mae = mean_absolute_error(y_calib, y_pred)
    rmse = np.sqrt(mean_squared_error(y_calib, y_pred))

    print(f"  ✔ สมการชดเชย: True_Temp = ({calib_model.coef_[0]:.4f} × Temp_M02_C) + ({calib_model.coef_[1]:.4f} × EC_M02) + {calib_model.intercept_:.4f}")
    print(f"  ✔ โมเดล Calibration เรียนรู้สำเร็จ!")
    print(f"  ✔ MAE (คลาดเคลื่อนเฉลี่ย) : {mae:.3f} °C")
    print(f"  ✔ RMSE                  : {rmse:.3f} °C")
    
    # 4. บันทึกโมเดลไว้ใช้งานตอน Inference
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

    plt.suptitle("Random Forest — NaCl Concentration Predictor",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()

    plot_path = out_dir / "rf_results.png"
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

def inference_example(main_model_path: Path, calib_model_path: Path) -> None:
    """
    ตัวอย่าง Deployment แบบ 2-Stage Model (ชดเชยความเพี้ยนของเซนเซอร์)
    """
    print("\n  --- ตัวอย่าง Deployment (ระบบชดเชยอุณหภูมิ + ทำนาย %NaCl) ---")

    loaded_main_model  = joblib.load(main_model_path)
    loaded_calib_model = joblib.load(calib_model_path)

    # 1. รับค่า RAW จาก sensor
    sensor_ec   = 1269
    sensor_temp = 5274 

    # 2. แปลง RAW → °C
    temp_c = sensor_temp / 100   # 32.00°C

    # 3. คำนวณ EC_25C เบื้องต้น (ใช้ temp_c แทน predicted_mercury ก่อน calib)
    ec_25c_approx = sensor_ec / (1 + 0.02 * (temp_c - 25))

    # 4. STAGE 1: Calibration — ทำนาย Mercury_Temp
    calib_input = pd.DataFrame([{
        "Temp_M02_C": temp_c,
        "EC_25C"    : ec_25c_approx,
    }])
    predicted_mercury = loaded_calib_model.predict(calib_input)[0]

    # 5. เตรียม sample_raw สำหรับ engineer_features
    #    engineer_features จะคำนวณ EC_25C ใหม่โดยใช้ predicted_mercury (แม่นยำกว่า)
    sample_raw = pd.DataFrame([{
        "EC_M02"      : sensor_ec,
        "Temp_M02"    : sensor_temp,
        "Mercury_Temp": predicted_mercury,
    }])

    # 6. สร้าง Engineered Features
    sample = engineer_features(sample_raw)
    sample = sample[FEATURE_COLS]

    # 7. STAGE 2: ทำนาย %NaCl
    prediction = loaded_main_model.predict(sample)[0]

    print(f"  [Calibration] Temp RAW={sensor_temp} → {temp_c:.2f}°C → ชดเชยเป็น {predicted_mercury:.2f}°C")
    print(f"  Input  : EC_RAW={sensor_ec}, Temp_RAW={sensor_temp} ({temp_c:.2f}°C)")
    print(f"  Output : %NaCl Predicted = {prediction:.4f} %")

#=============================================================================

# =============================================================================
# PART 5: DATA AUGMENTATION UTILITIES
# =============================================================================

def calibrate_k_nacl(input_file: Path) -> tuple:
    """หาค่า K_NACL จากข้อมูลจริง และตรวจสอบความสม่ำเสมอ"""
    df = pd.read_csv(input_file)
    df = df.dropna(subset=["EC_M02", "Mercury_Temp", "NaCl_Percent"])
    df = df[df["NaCl_Percent"] > 0].copy()

    ALPHA = 0.02
    df["K_est"] = df["EC_M02"] / (
        df["NaCl_Percent"] * (1 + ALPHA * (df["Mercury_Temp"] - 25))
    )

    k_mean   = df["K_est"].mean()
    k_median = df["K_est"].median()
    k_std    = df["K_est"].std()
    cv       = k_std / k_mean * 100

    print(f"\n  📊 K_NACL Calibration:")
    print(f"  Median : {k_median:.2f}  |  CV : {cv:.1f}%  "
          f"({'✅ ผ่าน' if cv <= 20 else '❌ ไม่ผ่าน — ห้ามสร้างข้อมูลจำลอง'})")

    bins   = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    labels = ["0–0.5%", "0.5–1.0%", "1.0–1.5%", "1.5–2.0%", "2.0–2.5%"]
    df["NaCl_bin"] = pd.cut(df["NaCl_Percent"], bins=bins, labels=labels)
    print(df.groupby("NaCl_bin", observed=True)["K_est"]
            .agg(["count", "mean", "std"]).round(2))

    return k_median, cv


def simulate_nacl_data(k_nacl: float, nacl_range: tuple,
                       temp_range: tuple, n_points: int = 300,
                       random_seed: int = 42) -> pd.DataFrame:
    """สร้างข้อมูลจำลองจากสมการฟิสิกส์เคมี NaCl"""
    ALPHA = 0.02
    np.random.seed(random_seed)

    nacl   = np.random.uniform(nacl_range[0], nacl_range[1], n_points)
    temp_c = np.random.uniform(temp_range[0], temp_range[1], n_points)

    ec_true   = k_nacl * nacl * (1 + ALPHA * (temp_c - 25))
    ec_noise  = np.random.normal(0, ec_true * 0.01)
    ec_raw    = (ec_true + ec_noise).clip(0)

    temp_bias = np.random.normal(2.5, 0.8, n_points)
    temp_raw  = ((temp_c + temp_bias) * 100).round().astype(int)

    df_sim = pd.DataFrame({
        "EC_M02"       : ec_raw.round(1),
        "Temp_M02"     : temp_raw,
        "Mercury_Temp" : temp_c.round(2),
        "NaCl_Percent" : nacl.round(4),
        "is_simulated" : True
    })

    print(f"\n  ✔ สร้างข้อมูลจำลอง {n_points} แถว")
    print(f"  NaCl : {df_sim['NaCl_Percent'].min():.3f}% – "
          f"{df_sim['NaCl_Percent'].max():.3f}%")
    print(f"  Temp : {df_sim['Mercury_Temp'].min():.1f}°C – "
          f"{df_sim['Mercury_Temp'].max():.1f}°C")
    print(f"  EC   : {df_sim['EC_M02'].min():.1f} – "
          f"{df_sim['EC_M02'].max():.1f}")

    return df_sim


def merge_and_prepare(real_file: Path, df_sim: pd.DataFrame) -> pd.DataFrame:
    """รวมข้อมูลจริงกับข้อมูลจำลอง พร้อม flag is_simulated"""
    df_real = pd.read_csv(real_file)
    df_real["is_simulated"] = False

    df_merged = pd.concat([df_real, df_sim], ignore_index=True)

    print(f"\n  📊 ข้อมูลรวม:")
    print(f"  จริง  : {(df_merged['is_simulated'] == False).sum():,} แถว")
    print(f"  จำลอง : {(df_merged['is_simulated'] == True).sum():,} แถว")
    print(f"  รวม   : {len(df_merged):,} แถว")

    return df_merged


# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # PART 1 — เตรียมข้อมูล
    X_train, X_test, y_train, y_test, DATA_ANALYSIS_DIR = prepare_dataset(INPUT_FILE)

    # PART 2a — เทรน Calibration Model (Stage 1)
    calib_model_path = train_calibration_model(X_train, DATA_ANALYSIS_DIR)

    # PART 2b — เทรนโมเดลหลักและประเมิน (Stage 2)
    model, y_pred, metrics, importances = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # PART 3 — Visualisation
    visualise(y_test, y_pred, metrics, importances, DATA_ANALYSIS_DIR)

    # PART 4 — บันทึกโมเดล + ตัวอย่าง Inference
    print("\n" + "=" * 62)
    print("  💾 PART 4: SAVE MODEL & INFERENCE EXAMPLE")
    print("=" * 62)
    model_path = save_model(model, DATA_ANALYSIS_DIR)
    inference_example(model_path, calib_model_path)   # ← 2 arguments

    print("\n" + "=" * 62)
    print("  ✅ Pipeline สมบูรณ์! ผลลัพธ์ทั้งหมดอยู่ใน:")
    print(f"     {DATA_ANALYSIS_DIR}")
    print("     rf_results.png  |  rf_nacl_model.joblib  |  calib_temp_model.joblib")
    print("=" * 62)


