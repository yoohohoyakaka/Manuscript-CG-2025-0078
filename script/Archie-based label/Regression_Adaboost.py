import pandas as pd
import numpy as np
import time
import tracemalloc
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

# ========== Runtime and Memory Monitoring ==========
start_time = time.time()
tracemalloc.start()

# ========== 1. Load Dataset ==========
file_path = "dataset_0304.xlsx"  # Replace with your actual path
df = pd.read_excel(file_path)

# ========== 2. Feature and Target Selection ==========
features = ["TDEP", "VELP", "VELS"]
target = "Sh"

X = df[features]
y = df[target]

# ========== 3. Feature Scaling ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 4. Initialize AdaBoost Regressor ==========
base_estimator = DecisionTreeRegressor(max_depth=None)
ada_regressor = AdaBoostRegressor(
    estimator=base_estimator,
    learning_rate=0.1,
    n_estimators=500,
    random_state=42
)

# ========== 5. Cross-Validation (5-fold, R² scoring) ==========
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ada_regressor, X_scaled, y, cv=cv, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ========== 6. Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== 7. Model Training ==========
ada_regressor.fit(X_train, y_train)

# ========== 8. Prediction and Evaluation ==========
y_pred = ada_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# ========== 9. Performance Summary ==========
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
print(f"\n⏱️ Execution Time: {end_time - start_time:.2f} seconds")
print(f"🧠 Peak Memory Usage: {peak / 1024 / 1024:.2f} MB")
tracemalloc.stop()
