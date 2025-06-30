import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1️⃣ 加载数据
file_path = "dataset_0304.xlsx"  # 修改为你的路径
df = pd.read_excel(file_path)

# 2️⃣ 特征与目标
features = ["TDEP", "VELP", "VELS"]  # 可根据需要调整
target = "Sh"  # 连续值标签

X = df[features]
y = df[target]

# 3️⃣ 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ 定义基础模型
base_estimator = DecisionTreeRegressor(random_state=42)

adaboost = AdaBoostRegressor(
    estimator=base_estimator,
    random_state=42
)

# 5️⃣ 构造参数搜索空间
param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'learning_rate': [0.1, 0.3, 0.5, 1.0],
    'estimator__max_depth': [2, 3, 4, 5]
}

# 6️⃣ 构造 GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=adaboost,
    param_grid=param_grid,
    cv=cv,
    scoring='r2',  # 回归常用指标
    n_jobs=-1,
    verbose=1
)

# 7️⃣ 执行搜索
grid_search.fit(X_train, y_train)

# 8️⃣ 打印最优参数与得分
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV R² Score: {grid_search.best_score_:.4f}")

# 9️⃣ 使用最优模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 🔟 测试集评估指标
print("\n📊 Test Set Evaluation:")
print(f"R² Score:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE:       {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE:      {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")

