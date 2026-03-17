# =============================================================================
# train_model.py — Model Training, Evaluation & Visualisation
# NaCl Concentration Predictor | Machine Learning + Physical Chemistry
# =============================================================================
# Loads the prepared CSVs, trains a Random Forest Regressor, evaluates it,
# saves performance plots, and persists the model for deployment.
#
# Usage:
#   python train_model.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import all settings from central config
from gg.config import (
    DATA_ANALYSIS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FEATURE_COLS,
    TARGET_COL,
    RF_PARAMS,
    MODEL_FILENAME,
    PLOT_FILENAME,
    IMPORTANCE_FILENAME,
    METRICS_FILENAME,
)


# =============================================================================
# HELPERS
# =============================================================================

def load_data():
    """Load the 4 split CSVs produced by prepare_dataset.py."""
    print(f"\n[1/5] Loading data from:\n  {DATA_ANALYSIS_DIR}\n")

    X_train = pd.read_csv(DATA_ANALYSIS_DIR / "X_train.csv")
    X_test  = pd.read_csv(DATA_ANALYSIS_DIR / "X_test.csv")

    # .values.ravel() → flat 1D array; prevents DataConversionWarning in sklearn
    y_train = pd.read_csv(DATA_ANALYSIS_DIR / "y_train.csv").values.ravel()
    y_test  = pd.read_csv(DATA_ANALYSIS_DIR / "y_test.csv").values.ravel()

    # Guard: verify columns match config exactly
    assert list(X_train.columns) == FEATURE_COLS, (
        f"Column mismatch!\n  Expected : {FEATURE_COLS}\n"
        f"  Got      : {list(X_train.columns)}"
    )

    print(f"  X_train : {X_train.shape}  ({X_train.shape[0]} samples, {X_train.shape[1]} features)")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """Initialise and fit the Random Forest model."""
    print("\n[2/5] Training Random Forest Regressor...")
    print(f"  Hyperparameters: {RF_PARAMS}\n")

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    print("  Training complete.")
    return model


def evaluate(model, X_test, y_test):
    """Compute and display R², MAE, RMSE. Return predictions + metrics dict."""
    print("\n[3/5] Evaluating on test set...\n")

    y_pred = model.predict(X_test)

    metrics = {
        "R2"  : r2_score(y_test, y_pred),
        "MAE" : mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

    # Pretty-print metrics table
    print(f"  {'Metric':<42} {'Value':>10}")
    print(f"  {'-'*52}")
    print(f"  {'R²  (Coefficient of Determination)':<42} {metrics['R2']:>10.4f}")
    print(f"  {'MAE (Mean Absolute Error)       [%NaCl]':<42} {metrics['MAE']:>10.4f}")
    print(f"  {'RMSE (Root Mean Squared Error)  [%NaCl]':<42} {metrics['RMSE']:>10.4f}")
    print(f"  {'-'*52}")

    # Feature importances (text bar chart)
    print("\n  Feature Importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    for feat, imp in importances.sort_values(ascending=False).items():
        bar = "█" * int(imp * 45)
        print(f"    {feat:<15} {imp:.4f}  {bar}")

    return y_pred, metrics, importances


def save_metrics(metrics: dict) -> None:
    """Write metrics to a plain-text file for logging / CI comparison."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / METRICS_FILENAME
    with open(out, "w") as f:
        f.write("NaCl RF Model — Test Set Metrics\n")
        f.write("=" * 35 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<6}: {v:.6f}\n")
    print(f"\n  Metrics saved → {out}")


def plot_actual_vs_predicted(y_test, y_pred, metrics: dict) -> None:
    """Scatter plot: Actual vs Predicted with perfect-prediction reference line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    ax.scatter(
        y_test, y_pred,
        color="#2563EB", edgecolors="#1E3A8A",
        alpha=0.75, s=60, zorder=3, label="Test Predictions"
    )

    # Perfect prediction reference (y = x)
    all_vals = np.concatenate([y_test, y_pred])
    pad = (all_vals.max() - all_vals.min()) * 0.05
    lim_min, lim_max = all_vals.min() - pad, all_vals.max() + pad

    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        color="#EF4444", linewidth=2, linestyle="--",
        zorder=2, label="Perfect Prediction (y = x)"
    )

    # Metrics annotation
    ann = (
        f"R²   = {metrics['R2']:.4f}\n"
        f"MAE  = {metrics['MAE']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}"
    )
    ax.text(
        0.05, 0.95, ann, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.9)
    )

    ax.set_xlabel("Actual %NaCl",    fontsize=12, labelpad=8)
    ax.set_ylabel("Predicted %NaCl", fontsize=12, labelpad=8)
    ax.set_title("Random Forest — Actual vs. Predicted %NaCl",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / PLOT_FILENAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Scatter plot saved → {out}")


def plot_feature_importances(importances: pd.Series) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    sorted_imp = importances.sort_values()
    colors = ["#2563EB" if v == sorted_imp.max() else "#93C5FD" for v in sorted_imp]

    bars = ax.barh(sorted_imp.index, sorted_imp.values, color=colors,
                   edgecolor="white", height=0.6)

    # Value labels on bars
    for bar, val in zip(bars, sorted_imp.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9,
                color="#1E3A8A")

    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importances — Random Forest", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlim(0, sorted_imp.max() * 1.18)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = RESULTS_DIR / IMPORTANCE_FILENAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Importance plot saved → {out}")


def save_model(model) -> None:
    """Persist the trained model to disk with joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / MODEL_FILENAME
    joblib.dump(model, out)
    print(f"  Model saved → {out}")

    print("\n  --- Reload snippet for deployment ---")
    print(f"  import joblib")
    print(f"  model = joblib.load(r'{out}')")
    print(f"  preds = model.predict(X_new)  # X_new: DataFrame with {FEATURE_COLS}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 55)
    print("  NaCl Concentration Prediction — Random Forest")
    print("=" * 55)

    # Step 1 — Load
    X_train, X_test, y_train, y_test = load_data()

    # Step 2 — Train
    model = train(X_train, y_train)

    # Step 3 — Evaluate
    y_pred, metrics, importances = evaluate(model, X_test, y_test)

    # Step 4 — Visualise
    print("\n[4/5] Generating visualisations...")
    plot_actual_vs_predicted(y_test, y_pred, metrics)
    plot_feature_importances(importances)
    save_metrics(metrics)

    # Step 5 — Save model
    print("\n[5/5] Saving model...")
    save_model(model)

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print(f"  Results → {RESULTS_DIR}")
    print(f"  Model   → {MODELS_DIR / MODEL_FILENAME}")
    print("=" * 55)


if __name__ == "__main__":
    main()
