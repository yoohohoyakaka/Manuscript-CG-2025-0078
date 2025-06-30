import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🚫 屏蔽所有警告
warnings.simplefilter("ignore")

# 1️⃣ 加载数据
file_path = "dataset_0304.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ 特征和目标变量
features = ["TDEP", "VELP", "VELS"]
target = "Sh"
X = df[features]
y = df[target]

# 3️⃣ 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 构建 XGB 模型
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)

# 5️⃣ 设置参数网格（如需更快运行可缩小）
param_grid = {
    "n_estimators": [100, 300, 500, 1000],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}

# 6️⃣ 交叉验证与网格搜索
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

# 7️⃣ 训练
grid_search.fit(X_train, y_train)

# 8️⃣ 输出结果
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV R² Score: {grid_search.best_score_:.4f}")

# 9️⃣ 使用最优模型测试
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
