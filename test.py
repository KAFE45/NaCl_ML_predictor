import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------
# DATA PREPARATION
# ---------------------------------
def prepare_dataset(input_file):

    df = pd.read_csv(input_file)

    # Feature Engineering
    df["EC_Temp"] = df["EC_M02"] * df["Temp_M02"]

    # Remove NaN
    initial_rows = len(df)
    df = df.dropna(subset=[
        "EC_M02",
        "Temp_M02",
        "Target_Temp",
        "Mercury_Temp",
        "EC_Temp",
        "NaCl_Percent"
    ])

    if initial_rows > len(df):
        print(f"⚠️ Dropped {initial_rows - len(df)} rows containing NaN")

    # Define X and y
    X = df[
        [
            "EC_M02",
            "Temp_M02",
            "Target_Temp",
            "Mercury_Temp",
            "EC_Temp",
        ]
    ]

    y = df["NaCl_Percent"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Save dataset
    input_path = Path(input_file)
    data_dir = input_path.parent.parent

    out_dir = data_dir / "DATA_ANALYSIS"
    out_dir.mkdir(exist_ok=True)

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)

    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    print("✅ Data prepared")
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))


# ---------------------------------
# MODEL TRAINING
# ---------------------------------
def train_models(data_dir):

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Neural Network
    ann_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    ann_model.fit(X_train_scaled, y_train)
    ann_pred = ann_model.predict(X_test_scaled)

    # Evaluation
    results = pd.DataFrame({
        "Model": ["Random Forest", "Neural Network"],
        "R2 Score": [
            r2_score(y_test, rf_pred),
            r2_score(y_test, ann_pred)
        ],
        "MSE": [
            mean_squared_error(y_test, rf_pred),
            mean_squared_error(y_test, ann_pred)
        ]
    })

    print("\n📊 Model Results")
    print(results)

models = ["Random Forest", "Neural Network"]
r2 = [0.929231, 0.858821]
mse = [0.027614, 0.055088]

plt.bar(models, r2)
plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.show()

# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == "__main__":

    INPUT_FILE = r"D:\NaCl_ML_predictor\data\RAW_CUT\data_RAW_CUT_20260305_1553.csv"

    prepare_dataset(INPUT_FILE)

    DATA_DIR = Path(r"D:\NaCl_ML_predictor\data\DATA_ANALYSIS")

    train_models(DATA_DIR)

