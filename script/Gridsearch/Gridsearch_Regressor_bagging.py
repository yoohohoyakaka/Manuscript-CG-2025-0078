import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1️⃣ 加载数据
file_path = "dataset_0304.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ 特征与目标
features = ["TDEP", "VELP", "VELS"]
target = "Sh"

X = df[features]
y = df[target]

# 3️⃣ 训练集与测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ 基础模型与 BaggingRegressor（新版使用 estimator）
base_estimator = DecisionTreeRegressor(random_state=42)
bagging = BaggingRegressor(estimator=base_estimator, random_state=42)

# 5️⃣ 超参数搜索空间
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_samples': [0.5, 0.8, 1.0],
    'max_features': [0.5, 1, 2],
    'estimator__max_depth': [5, 7, 9, None]
}

# 6️⃣ 网格搜索交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=bagging,
    param_grid=param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# 7️⃣ 开始训练
grid_search.fit(X_train, y_train)

# 8️⃣ 最佳结果输出
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV R² Score: {grid_search.best_score_:.4f}")

# 9️⃣ 测试集评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📊 Test Set Evaluation:")
print(f"R² Score:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE:       {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE:      {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
