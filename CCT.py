# =============================================================================
# NaCl Concentration Prediction using Random Forest Regressor
# Features: Temperature, Electrical Conductivity → Target: %NaCl
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

DATA_DIR = r"D:\NaCl_ML_predictor\data\DATA_ANALYSIS"

print("=" * 55)
print("  NaCl Concentration Prediction — Random Forest")
print("=" * 55)
print("\n[1/5] Loading data from DATA_ANALYSIS/...\n")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

# .values.ravel() converts the DataFrame column into a flat 1D array,
# preventing DataConversionWarning from scikit-learn about column-vector input.
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")
print(f"  y_train shape : {y_train.shape}")
print(f"  y_test  shape : {y_test.shape}")

# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

print("\n[2/5] Training Random Forest Regressor...\n")

rf_model = RandomForestRegressor(
    n_estimators=200,       # Number of trees in the forest
    max_depth=None,         # Trees grow until leaves are pure (or min_samples_split)
    min_samples_split=2,    # Minimum samples required to split a node
    min_samples_leaf=1,     # Minimum samples required at a leaf node
    random_state=42,        # Seed for reproducibility
    n_jobs=-1               # Use all available CPU cores for speed
)

rf_model.fit(X_train, y_train)
print("  Model training complete.")

# =============================================================================
# STEP 3: EVALUATION
# =============================================================================

print("\n[3/5] Evaluating model on the test set...\n")

y_pred = rf_model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  {'Metric':<35} {'Value':>10}")
print(f"  {'-'*45}")
print(f"  {'R² Score (Coefficient of Determination)':<35} {r2:>10.4f}")
print(f"  {'MAE  (Mean Absolute Error)':<35} {mae:>10.4f}")
print(f"  {'RMSE (Root Mean Squared Error)':<35} {rmse:>10.4f}")
print(f"  {'-'*45}")

# =============================================================================
# STEP 4: VISUALIZATION — Actual vs. Predicted Scatter Plot
# =============================================================================

print("\n[4/5] Generating Actual vs. Predicted scatter plot...")

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

# Scatter: actual vs predicted points
scatter = ax.scatter(
    y_test, y_pred,
    color="#2563EB",
    edgecolors="#1E3A8A",
    alpha=0.75,
    s=60,
    zorder=3,
    label="Predictions"
)

# Perfect prediction reference line (y = x)
all_vals = np.concatenate([y_test, y_pred])
line_min, line_max = all_vals.min() * 0.95, all_vals.max() * 1.05
ax.plot(
    [line_min, line_max], [line_min, line_max],
    color="#EF4444",
    linewidth=2,
    linestyle="--",
    zorder=2,
    label="Perfect Prediction (y = x)"
)

# Metrics annotation box
metrics_text = (
    f"R²   = {r2:.4f}\n"
    f"MAE  = {mae:.4f}\n"
    f"RMSE = {rmse:.4f}"
)
ax.text(
    0.05, 0.95, metrics_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
    family="monospace"
)

ax.set_xlabel("Actual %NaCl", fontsize=12, labelpad=8)
ax.set_ylabel("Predicted %NaCl", fontsize=12, labelpad=8)
ax.set_title("Random Forest — Actual vs. Predicted %NaCl", fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_xlim(line_min, line_max)
ax.set_ylim(line_min, line_max)

plt.tight_layout()

plot_path = os.path.join(DATA_DIR, "actual_vs_predicted.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Plot saved → {plot_path}")

# =============================================================================
# STEP 5: MODEL SAVING
# =============================================================================

print("\n[5/5] Saving trained model...")

model_path = os.path.join(DATA_DIR, "rf_nacl_model.joblib")
joblib.dump(rf_model, model_path)
print(f"  Model saved → {model_path}")

# --- How to reload the model later for deployment ---
# loaded_model = joblib.load("DATA_ANALYSIS/rf_nacl_model.joblib")
# new_predictions = loaded_model.predict(X_new)

print("\n" + "=" * 55)
print("  Pipeline complete. All outputs saved to DATA_ANALYSIS/")
print("=" * 55)