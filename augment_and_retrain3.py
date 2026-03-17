# augment_and_retrain.py
from pathlib import Path
from nacl_pipeline_003 import (
    calibrate_k_nacl_poly,
    simulate_nacl_data,
    merge_and_prepare,
    prepare_dataset,
    train_calibration_model,
    train_and_evaluate,
    visualise,
    save_model,
    INPUT_FILE,
)

AUGMENTED_FILE = Path(r"D:\NaCl_ML_predictor\data\RAW_CUT\data_AUGMENTED.csv")
SIM_FILE       = Path(r"D:\NaCl_ML_predictor\data\DATA_ANALYSIS\simulated_aug.csv")
R2_THRESHOLD   = 0.95

if __name__ == "__main__":

    # 0. Validate poly fit
    poly_model, r2 = calibrate_k_nacl_poly(INPUT_FILE)

    if r2 < R2_THRESHOLD:
        raise ValueError(
            f"Polynomial fit R² = {r2:.4f} < {R2_THRESHOLD}\n"
            f"Sensor มี behavior ซับซ้อนเกินไป — ต้องเก็บข้อมูลจริงในแล็บแทน"
        )

    print(f"\n  ✔ Poly fit R² = {r2:.4f} — ผ่าน ดำเนินต่อได้")

    # 1. สร้างข้อมูลจำลอง
    df_sim = simulate_nacl_data(
        poly_model = poly_model,
        nacl_range = (1.5, 2.0),
        temp_range = (40.0, 80.0),
        n_points   = 300
    )
    df_sim.to_csv(SIM_FILE, index=False)
    print(f"  ✔ บันทึกข้อมูลจำลองที่ : {SIM_FILE.name}")

    # 2. รวมและบันทึก
    df_merged = merge_and_prepare(INPUT_FILE, df_sim)
    df_merged.to_csv(AUGMENTED_FILE, index=False)
    print(f"  ✔ บันทึกข้อมูลรวมที่   : {AUGMENTED_FILE.name}")

    # 3. เทรนใหม่
    X_train, X_test, y_train, y_test, DATA_ANALYSIS_DIR = prepare_dataset(AUGMENTED_FILE)
    calib_model_path = train_calibration_model(X_train, DATA_ANALYSIS_DIR)
    model, y_pred, metrics, importances = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # 4. Visualise + บันทึก
    visualise(y_test, y_pred, metrics, importances, DATA_ANALYSIS_DIR)
    save_model(model, DATA_ANALYSIS_DIR)
