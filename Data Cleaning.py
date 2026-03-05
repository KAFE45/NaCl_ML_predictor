from datetime import datetime
from pathlib import Path
import pandas as pd


def save_dataset(df, folder, filename, cols):
    folder.mkdir(exist_ok=True)
    df_out = df[[c for c in cols if c in df.columns]]
    path = folder / filename
    df_out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def clean_experimental_data(input_path):

    df = pd.read_excel(input_path, header=[0, 1], engine="openpyxl")

    cols = list(df.columns)
    for i, n in enumerate(["NaCl_Percent", "model_group", "Variable"]):
        cols[i] = (n, "")
    df.columns = pd.MultiIndex.from_tuples(cols)

    df.iloc[:, :3] = df.iloc[:, :3].ffill()

    df = df.melt(id_vars=df.columns[:3].tolist())
    df.columns = ["NaCl_Percent","model_group","Variable","Target_Temp","Rep","Value"]

    df["Target_Temp"] = df["Target_Temp"].astype(str).str.replace("°C","",regex=False).str.strip()
    df = df[pd.to_numeric(df["Target_Temp"], errors="coerce").notnull()].dropna(subset=["Value"])

    df = (
        df.pivot_table(
            index=["NaCl_Percent","Target_Temp","Rep"],
            columns="Variable",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )

    df.columns.name = None
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values(["NaCl_Percent","Target_Temp","Rep"]).reset_index(drop=True)

    input_path = Path(input_path)
    data_dir = input_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    full_cols = ["NaCl_Percent","Target_Temp","Rep","Mercury_Temp","Temp_M01","EC_M01","Temp_M02","EC_M02"]
    cut_cols  = ["NaCl_Percent","Target_Temp","Rep","Mercury_Temp","Temp_M02","EC_M02"]

    full_path = save_dataset(df, data_dir/"RAW_FULL", f"data_RAW_FULL_{ts}.csv", full_cols)
    cut_path  = save_dataset(df, data_dir/"RAW_CUT",  f"data_RAW_CUT_{ts}.csv",  cut_cols)

    print("✅ Save complete")
    print(full_path)
    print(cut_path)


if __name__ == "__main__":
    INPUT_FILE = r"D:\NaCl_ML_predictor\data\data_RAW.xlsx"
    clean_experimental_data(INPUT_FILE)