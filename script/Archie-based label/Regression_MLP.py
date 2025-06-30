import pandas as pd
import numpy as np
import warnings
import time
import tracemalloc
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== Suppress All Warnings ==========
warnings.simplefilter("ignore")

# ========== Start Execution Time and Memory Monitoring ==========
tracemalloc.start()
start_time = time.time()

# ========== 1. Load Dataset ==========
file_path = "dataset_0304.xlsx"
df = pd.read_excel(file_path)

# ========== 2. Select Features and Target ==========
features = ["TDEP", "VELP", "VELS"]
target = "Sh"
X = df[features]
y = df[target]

# ========== 3. Standardize Features ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 4. Split into Training and Test Sets ==========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== 5. Define MLP Regressor with Specified Hyperparameters ==========
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    alpha=0.01,
    random_state=42
)

# ========== 6. Perform Manual 5-Fold Cross-Validation (R² score) ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = []

for train_index, val_index in kf.split(X_scaled):
    X_train_cv, X_val_cv = X_scaled[train_index], X_scaled[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

    mlp_regressor.fit(X_train_cv, y_train_cv)
    y_val_pred_cv = mlp_regressor.predict(X_val_cv)
    cv_r2_scores.append(r2_score(y_val_cv, y_val_pred_cv))

print(f"Cross-validation R²: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")

# ========== 7. Train Final Model on Training Set ==========
mlp_regressor.fit(X_train, y_train)
y_pred = mlp_regressor.predict(X_test)

# ========== 8. Evaluate Model Performance ==========
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test Set MAE: {mae:.4f}")
print(f"Test Set RMSE: {rmse:.4f}")
print(f"Test Set R²: {r2:.4f}")

# ========== 9. Report Execution Time and Peak Memory Usage ==========
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\n⏱️ Execution Time: {end_time - start_time:.2f} seconds")
print(f"🧠 Peak Memory Usage: {peak / 1024 / 1024:.2f} MB")
