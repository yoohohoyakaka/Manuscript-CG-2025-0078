import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ 加载数据
file_path = "dataset_0304.xlsx"  # ← 替换为你的路径
df = pd.read_excel(file_path)

# 2️⃣ 特征与目标
features = ["TDEP", "VELP", "VELS"]
target = "Sh"

X = df[features]
y = df[target]

# 3️⃣ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ 参数网格
param_grid = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf": [1, 2, 4, 6],
    "criterion": ["squared_error", "friedman_mse"]
}

# 5️⃣ 构造 GridSearchCV
dt_model = DecisionTreeRegressor(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=kf,
    scoring='r2',
    n_jobs=-1
)

# 6️⃣ 执行超参搜索
grid_search.fit(X_train, y_train)

# 7️⃣ 输出最佳参数和交叉验证得分
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV R² Score: {grid_search.best_score_:.4f}")

# 8️⃣ 测试集预测与评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Test Set Evaluation:")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")

