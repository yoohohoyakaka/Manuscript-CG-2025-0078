# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

print("==========🚩 Biot-Based Hydrate Regression (GBDT) ==========")

# Step 1: Load the dataset
file_path = "dataset_1.xlsx"  # Three-phase Biot dataset
df = pd.read_excel(file_path)

# Step 2: Define features and target
features = ["ILD", "Cl", "Sw"]
target = "Sh_interpolated"

X = df[features]
y = df[target]

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Define the GBDT regressor with specified hyperparameters
gbdt_regressor = GradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=1000,
    random_state=42
)

# Step 5: Perform 5-fold cross-validation using R² as the evaluation metric
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gbdt_regressor, X_scaled, y, cv=cv, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Step 6: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 7: Train the model
gbdt_regressor.fit(X_train, y_train)

# Step 8: Make predictions and evaluate performance
y_pred = gbdt_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nTest RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

