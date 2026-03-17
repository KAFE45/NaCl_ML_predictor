# find_alpha.py
from nacl_pipeline_003 import find_true_alpha, INPUT_FILE

if __name__ == "__main__":
    alpha = find_true_alpha(INPUT_FILE)
    print(f"\n  ══════════════════════════════════════")
    print(f"  ขั้นตอนถัดไป:")
    print(f"  อัปเดต ALPHA = {alpha:.4f} ใน nacl_pipeline_003.py")
    print(f"  ใน 3 จุด:")
    print(f"    1. engineer_features()")
    print(f"    2. calibrate_k_nacl_poly()")
    print(f"    3. inference_example()")
    print(f"  แล้วรัน augment_and_retrain.py ใหม่")
    print(f"  ══════════════════════════════════════")
