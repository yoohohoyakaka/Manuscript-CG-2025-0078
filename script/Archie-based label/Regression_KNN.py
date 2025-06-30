import pandas as pd
import numpy as np
import time
import tracemalloc
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# ========== 3. Define KNN Regressor (with specified hyperparameters) ==========
knn_regressor = KNeighborsRegressor(
    n_neighbors=3,
    weights='distance',
    p=1  # Manhattan distance
)

# ========== 4. Perform 5-Fold Cross-Validation (R² score) ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn_regressor, X, y, cv=kf, scoring="r2")
print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ========== 5. Split into Training and Test Sets ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 6. Train and Predict ==========
knn_regressor.fit(X_train, y_train)
y_pred = knn_regressor.predict(X_test)

# ========== 7. Evaluate Model Performance ==========
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test Set MAE: {mae:.4f}")
print(f"Test Set RMSE: {rmse:.4f}")
print(f"Test Set R²: {r2:.4f}")

# ========== 8. Report Execution Time and Peak Memory Usage ==========
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\n⏱️ Execution Time: {end_time - start_time:.2f} seconds")
print(f"🧠 Peak Memory Usage: {peak / 1024 / 1024:.2f} MB")
