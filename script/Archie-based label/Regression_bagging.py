import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ========== 1. Load Dataset ==========
file_path = "dataset_0304.xlsx"  # Update with the actual path if needed
df = pd.read_excel(file_path)

# ========== 2. Feature and Target Selection ==========
features = ["TDEP", "VELP", "VELS"]
target = "Sh"

X = df[features]
y = df[target]

# ========== 3. Initialize Bagging Regressor ==========
base_estimator = DecisionTreeRegressor(max_depth=None)
bagging_regressor = BaggingRegressor(
    estimator=base_estimator,
    n_estimators=1000,
    max_features=3,
    max_samples=1.0,
    bootstrap=True,
    random_state=42
)

# ========== 4. Start Performance Tracking ==========
tracemalloc.start()
start_time = time.time()

# ========== 5. Cross-Validation (5-fold, R² metric) ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(bagging_regressor, X, y, cv=kf, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ========== 6. Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 7. Model Training ==========
bagging_regressor.fit(X_train, y_train)

# ========== 8. Prediction and Evaluation ==========
y_pred = bagging_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# ========== 9. Performance Summary ==========
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\n⏱️ Execution Time: {end_time - start_time:.2f} seconds")
print(f"🧠 Peak Memory Usage: {peak / 1024 / 1024:.2f} MB")
