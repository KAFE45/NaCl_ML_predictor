# augment_and_retrain.py
from pathlib import Path
from nacl_pipeline_002 import (    
    calibrate_k_nacl,
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

if __name__ == "__main__":

    # 0. Validate K_NACL
    K_NACL, k_cv = calibrate_k_nacl(INPUT_FILE)
    if k_cv > 20.0:
        raise ValueError(
            f"K_NACL CV = {k_cv:.1f}% > 20%\n"
            f"สมการเส้นตรงไม่ fit กับ sensor — ต้องเก็บข้อมูลจริงแทน"
        )
    print(f"\n  ✔ K_NACL = {K_NACL:.2f} (CV={k_cv:.1f}%) — ผ่าน")

    # 1. สร้างข้อมูลจำลอง
    df_sim = simulate_nacl_data(
        k_nacl     = K_NACL,
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