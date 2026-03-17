import numpy as np
import pandas as pd

def simulate_nacl_data(
    nacl_range: tuple,
    temp_range: tuple,
    n_points: int = 200
) -> pd.DataFrame:
    """
    สร้างข้อมูลจำลองจากสมการฟิสิกส์เคมีของ NaCl solution
    
    EC ของ NaCl ≈ k × [NaCl] × (1 + α × (T - 25))
    k    = ~5.5  (conductivity factor ของ NaCl)
    α    = 0.02  (temperature coefficient, 2%/°C)
    """
    ALPHA = 0.02
    K_NACL = 5.5   # calibration constant — ปรับให้ตรงกับ sensor จริง

    np.random.seed(42)
    n = n_points

    # สุ่มค่าในช่วงที่ต้องการ
    nacl   = np.random.uniform(nacl_range[0],  nacl_range[1],  n)
    temp_c = np.random.uniform(temp_range[0],  temp_range[1],  n)

    # สมการ EC จากฟิสิกส์
    ec_true = K_NACL * nacl * (1 + ALPHA * (temp_c - 25))

    # จำลอง sensor noise และ bias
    sensor_noise     = np.random.normal(0, 0.3, n)   # EC noise
    temp_sensor_bias = np.random.normal(2.5, 0.5, n) # sensor เพี้ยน ~2.5°C

    # แปลงกลับเป็น RAW units (× 100)
    ec_raw   = (ec_true + sensor_noise).clip(0) * (1639/9)  # scale ให้ตรง range เดิม
    temp_raw = ((temp_c + temp_sensor_bias) * 100).astype(int)

    df_sim = pd.DataFrame({
        "EC_M02"      : ec_raw.round(1),
        "Temp_M02"    : temp_raw,
        "Mercury_Temp": temp_c.round(1),
        "NaCl_Percent": nacl.round(3),
        "is_simulated": True   # ← flag สำคัญ ต้องเก็บไว้
    })

    return df_sim


# สร้างข้อมูลในโซนที่ขาด
df_aug = simulate_nacl_data(
    nacl_range  = (1.5, 2.0),
    temp_range  = (45.0, 65.0),
    n_points    = 300
)

print(df_aug[["EC_M02", "Temp_M02", "Mercury_Temp", "NaCl_Percent"]].describe())
df_aug.to_csv("simulated_nacl_1.5_2.0_45_65.csv", index=False)

def augment_existing_data(df_real: pd.DataFrame, target_nacl_min=1.5, 
                           target_temp_min=45) -> pd.DataFrame:
    """
    ขยายข้อมูลจากแถวที่ใกล้เคียงกับโซนที่ต้องการ
    """
    # หาแถวที่ NaCl สูงและ Temp สูง (ใกล้โซนที่ขาด)
    near_zone = df_real[
        (df_real["NaCl_Percent"] >= target_nacl_min) &
        (df_real["Mercury_Temp"] >= target_temp_min)
    ]
    
    print(f"  พบ {len(near_zone)} แถวในโซนใกล้เคียง")
    
    # Bootstrap + small noise
    np.random.seed(42)
    augmented = near_zone.sample(n=200, replace=True).copy()
    augmented["EC_M02"]       += np.random.normal(0, 5, 200)
    augmented["Temp_M02"]     += np.random.randint(-50, 50, 200)
    augmented["Mercury_Temp"] += np.random.normal(0, 0.3, 200)
    augmented["is_simulated"]  = True
    
    return augmented