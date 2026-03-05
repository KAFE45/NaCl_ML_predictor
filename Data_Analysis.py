import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

# 1. โหลดข้อมูล
file_path = r"D:\NaCl_ML_predictor\data\data_CLEANED_ML_Ready.csv"
df = pd.read_csv(file_path)

# 🌟 สลับมาโฟกัสที่ M02 เพราะข้อมูลสมบูรณ์กว่า
sensor_features = ['Temp_M02', 'EC_M02']
target_col = 'NaCl_Percent'

# 2. คลีนเฉพาะแถวที่ M02 มีข้อมูลครบ
df_clean = df.dropna(subset=sensor_features + [target_col])

X = df_clean[sensor_features]
y = df_clean[target_col]

# 3. แบ่งชุดข้อมูล (ใช้เทคนิครักษาอัตราส่วนความเข้มข้น)
# เราใช้ y เป็นตัวแบ่งกลุ่ม (Stratify) เพื่อให้ 1.60% กระจายไปทั้งสองฝั่ง
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 ตรวจสอบข้อมูลหลังสลับมาใช้ M02:")
print(f"จำนวนข้อมูลทั้งหมด: {len(df_clean)} แถว (จากเดิม 184 แถวใน M01)")
print(f"✅ ชุดสอน (Train): {len(X_train)} แถว")
print(f"✅ ชุดทดสอบ (Test): {len(X_test)} แถว")

# 4. สร้างและฝึกสอนโมเดล Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. ทำนายผลและวัดความแม่นยำ
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- 🏆 ผลลัพธ์โมเดล (M02 Sensor) ---")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# 6. เซฟผลลัพธ์เปรียบเทียบออกมาดู (เพื่อเช็ก 1.60%)
comparison = X_test.copy()
comparison['Actual'] = y_test.values
comparison['Predicted'] = y_pred
comparison['Error'] = np.abs(y_test.values - y_pred)

# เรียงลำดับตามค่าจริงเพื่อให้ดูง่ายในไฟล์ CSV
comparison = comparison.sort_values('Actual')
comparison.to_csv(r"D:\NaCl_ML_predictor\data\Comparison_M02_Model.csv", index=False, encoding='utf-8-sig')

print(f"\n💾 เซฟไฟล์ 'Comparison_M02_Model.csv' เรียบร้อย! ลองเปิดเช็กค่า 1.60 ได้เลยครับ")

plt.figure(figsize=(10, 6))
# พล็อตความสัมพันธ์ระหว่าง EC และ NaCl โดยแยกสีตามอุณหภูมิ
sns.scatterplot(data=df_clean, x='EC_M02', y='NaCl_Percent', hue='Temp_M02', palette='viridis')
plt.title('Relationship Check: EC_M02 vs NaCl_Percent')
plt.grid(True)
plt.show()