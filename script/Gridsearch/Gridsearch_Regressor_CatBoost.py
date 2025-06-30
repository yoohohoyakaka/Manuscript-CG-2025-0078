import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 屏蔽所有警告
warnings.simplefilter("ignore")

# 1️⃣ 读取数据
file_path = "dataset_0304.xlsx"  # ← 根据实际路径修改
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
    "iterations": [300, 500, 700, 1000],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "depth": [4, 6, 8, 10]
}

# 5️⃣ 构造基础模型
cat_model = CatBoostRegressor(
    verbose=0,
    random_seed=42
)

# 6️⃣ 构造 GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    cv=kf,
    scoring='r2',
    n_jobs=-1
)

# 7️⃣ 执行超参搜索
grid_search.fit(X_train, y_train)

# 8️⃣ 最佳参数和得分
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV R² Score: {grid_search.best_score_:.4f}")

# 9️⃣ 用最佳模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 🔟 测试集评估
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Test Set Evaluation:")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
