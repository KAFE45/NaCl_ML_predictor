from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_dataset(input_file):

    df = pd.read_csv(input_file)

    # -------------------
    # Feature Engineering
    # -------------------
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
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

if __name__ == "__main__":

    INPUT_FILE = r"D:\NaCl_ML_predictor\data\RAW_CUT\data_RAW_CUT_20260305_1553.csv"

    prepare_dataset(INPUT_FILE)