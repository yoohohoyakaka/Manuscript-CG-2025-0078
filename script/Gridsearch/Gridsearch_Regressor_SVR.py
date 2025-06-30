import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🚫 Suppress all warnings
warnings.simplefilter("ignore")

# Step 1️⃣: Load dataset
file_path = "dataset_0304.xlsx"
df = pd.read_excel(file_path)

# Step 2️⃣: Select features and target
features = ["TDEP", "VELP", "VELS"]
target = "Sh"

X = df[features]
y = df[target]

# Step 3️⃣: Build pipeline with standardization and SVR
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature standardization
    ('svr', SVR())                 # Support Vector Regressor
])

# Step 4️⃣: Define hyperparameter grid
param_grid = {
    'svr__C': [1, 10, 100],
    'svr__epsilon': [0.01, 0.05, 0.1],
    'svr__gamma': ['scale', 'auto']
}

# Step 5️⃣: Configure GridSearchCV with 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Step 6️⃣: Train model using grid search
grid_search.fit(X, y)

# Step 7️⃣: Output best parameters and CV performance
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV R² Score: {grid_search.best_score_:.4f}")

# Step 8️⃣: Evaluate on test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Step 9️⃣: Report evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Test Set Evaluation:")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")

