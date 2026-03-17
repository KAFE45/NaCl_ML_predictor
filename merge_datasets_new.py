# merge_datasets.py
from datetime import datetime
from pathlib import Path
import pandas as pd
import re

# =============================================================================
# CONFIG
# =============================================================================
EXCEL_FILE = Path(r"D:\NaCl_ML_predictor\data\data_RAW.xlsx")
CSV_FILE   = Path(r"D:\NaCl_ML_predictor\data\20260211_ChemMeter_phase1_dataset.csv")
OUT_DIR    = Path(r"D:\NaCl_ML_predictor\data\MERGED")

MODEL_MAP = {
    "model 1" : "M01", "model01" : "M01",
    "model 2" : "M02", "model02" : "M02",
    "model 3" : "M03", "model03" : "M03",
}

# =============================================================================
# PART 1: Parse Excel
# =============================================================================
def parse_excel(path: Path) -> pd.DataFrame:
    print(f"  📂 Parse Excel: {path.name}")
    df = pd.read_excel(path, header=[0, 1], engine="openpyxl")

    cols = list(df.columns)
    for i, n in enumerate(["NaCl_Percent", "model_group", "Variable"]):
        cols[i] = (n, "")
    df.columns = pd.MultiIndex.from_tuples(cols)
    df.iloc[:, :3] = df.iloc[:, :3].ffill()

    df = df.melt(id_vars=df.columns[:3].tolist())
    df.columns = ["NaCl_Percent", "model_group", "Variable",
                  "Target_Temp", "Rep", "Value"]

    df["Target_Temp"] = (df["Target_Temp"].astype(str)
                         .str.replace("°C", "", regex=False).str.strip())
    df = df[pd.to_numeric(df["Target_Temp"], errors="coerce").notnull()]
    df = df.dropna(subset=["Value"])

    df["model_group"] = (df["model_group"].astype(str).str.strip().str.lower()
                         .map(lambda x: MODEL_MAP.get(x, x)))
    df["col_name"] = df["Variable"].astype(str).str.strip()

    df = (df.pivot_table(
            index=["NaCl_Percent", "Target_Temp", "Rep"],
            columns="col_name", values="Value", aggfunc="first")
          .reset_index())
    df.columns.name = None
    df = df.apply(pd.to_numeric, errors="coerce")
    df["source"] = "excel"

    print(f"  ✔ {len(df):,} แถว | คอลัมน์: {[c for c in df.columns if c != 'source']}")
    return df


# =============================================================================
# PART 2: Parse CSV
# =============================================================================
def parse_csv(path: Path) -> pd.DataFrame:
    print(f"\n  📂 Parse CSV: {path.name}")
    raw = pd.read_csv(path, header=None, encoding="utf-8-sig")
    raw = raw.dropna(axis=1, how="all")

    temps = raw.iloc[0, 3:].ffill().tolist()
    reps  = raw.iloc[1, 3:].tolist()

    records = []
    for row_idx in range(2, len(raw)):
        row       = raw.iloc[row_idx]
        nacl_raw  = str(row.iloc[0])
        model_raw = str(row.iloc[1]).strip().lower()
        var_raw   = str(row.iloc[2]).strip().lower()

        nacl_match = re.search(r"[\d.]+", nacl_raw)
        if not nacl_match:
            continue
        nacl = float(nacl_match.group())

        model_prefix = MODEL_MAP.get(model_raw)
        if not model_prefix:
            continue

        if "concentration" in var_raw:
            col_name = f"EC_{model_prefix}"
        elif "temperature" in var_raw:
            col_name = f"Temp_{model_prefix}"
        else:
            continue

        for col_idx, (temp, rep) in enumerate(zip(temps, reps)):
            try:
                val = float(row.iloc[3 + col_idx])
                records.append({
                    "NaCl_Percent" : nacl,
                    "Target_Temp"  : float(temp),
                    "Rep"          : int(float(rep)),
                    "col_name"     : col_name,
                    "Value"        : val,
                })
            except (ValueError, TypeError):
                continue

    df = pd.DataFrame(records)
    df = (df.pivot_table(
            index=["NaCl_Percent", "Target_Temp", "Rep"],
            columns="col_name", values="Value", aggfunc="first")
          .reset_index())
    df.columns.name = None

    # Mercury_Temp = Target_Temp (heatplate)
    df["Mercury_Temp"] = df["Target_Temp"]
    df = df.apply(pd.to_numeric, errors="coerce")
    df["source"] = "csv_heatplate"

    print(f"  ✔ {len(df):,} แถว | คอลัมน์: {[c for c in df.columns if c != 'source']}")
    return df


# =============================================================================
# PART 3: Merge + Save
# =============================================================================
def merge_and_save(df_excel: pd.DataFrame, df_csv: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.concat([df_excel, df_csv], ignore_index=True)
    df_merged = df_merged.sort_values(
        ["NaCl_Percent", "Target_Temp", "Rep"]
    ).reset_index(drop=True)

    print(f"\n  📊 ข้อมูลรวม: {len(df_merged):,} แถว")
    print(f"  Excel (Mercury_Temp จริง)     : {(df_merged['source']=='excel').sum():,} แถว")
    print(f"  CSV   (Mercury_Temp=heatplate): {(df_merged['source']=='csv_heatplate').sum():,} แถว")
    print(f"  NaCl  : {df_merged['NaCl_Percent'].min()}% – {df_merged['NaCl_Percent'].max()}%")
    print(f"  Temp  : {df_merged['Target_Temp'].min()}°C – {df_merged['Target_Temp'].max()}°C")

    print(f"\n  📊 Coverage (NaCl × Temp):")
    pivot = df_merged.groupby(["NaCl_Percent", "Target_Temp"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # ไฟล์เดียว — ลำดับคอลัมน์แบบ CUT
    final_cols = [
        "NaCl_Percent", "Target_Temp", "Rep",
        "Mercury_Temp",
        "Temp_M01", "EC_M01",
        "Temp_M02", "EC_M02",
        "Temp_M03", "EC_M03",
        "source"
    ]
    out_path = OUT_DIR / f"data_MERGED_{ts}.csv"
    df_merged[[c for c in final_cols if c in df_merged.columns]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )

    print(f"\n  ✅ บันทึกสำเร็จ: {out_path}")
    print(f"  ⚠  แถวจาก CSV: Mercury_Temp = heatplate (ไม่แม่นยำ)")

    return df_merged


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("  🔀 MERGE DATASETS")
    print("=" * 62)

    df_excel = parse_excel(EXCEL_FILE)
    df_csv   = parse_csv(CSV_FILE)
    df       = merge_and_save(df_excel, df_csv)