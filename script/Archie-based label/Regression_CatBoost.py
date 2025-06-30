import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import tracemalloc

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== Suppress All Warnings ==========
warnings.simplefilter("ignore")

# ========== Start Time and Memory Monitoring ==========
tracemalloc.start()
start_time = time.time()

# ========== 1. Load Dataset ==========
file_path = "dataset_0304.xlsx"  # Update with your actual path
df = pd.read_excel(file_path)

# ========== 2. Feature and Target Selection ==========
features = ["TDEP", "VELP", "VELS"]
target = "Sh"

X = df[features]
y = df[target]

# ========== 3. Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 4. Initialize CatBoost Regressor ==========
cat_regressor = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.2,
    depth=6,
    random_seed=42,
    verbose=False  # Suppress training output
)

# ========== 5. Cross-Validation (5-fold, R² metric) ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cat_regressor, X, y, cv=kf, scoring="r2")
print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ========== 6. Model Training ==========
cat_regressor.fit(X_train, y_train)

# ========== 7. Prediction and Evaluation ==========
y_pred = cat_regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test Set MAE: {mae:.4f}")
print(f"Test Set RMSE: {rmse:.4f}")
print(f"Test Set R²: {r2:.4f}")

# ========== 8. Execution Summary ==========
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\n⏱️ Execution Time: {end_time - start_time:.2f} seconds")
print(f"🧠 Peak Memory Usage: {peak / 1024 / 1024:.2f} MB")
