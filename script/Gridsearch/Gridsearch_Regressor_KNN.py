import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ 加载数据
file_path = "dataset_0304.xlsx"
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

# 4️⃣ 构建 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),           # Z-score 标准化
    ('knn', KNeighborsRegressor())          # KNN 模型
])

# 5️⃣ 参数网格（注意 knn__ 前缀）
param_grid = {
    "knn__n_neighbors": [2, 3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2]  # p=1 曼哈顿距离, p=2 欧几里得距离
}

# 6️⃣ 网格搜索 + 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

# 7️⃣ 执行搜索
grid_search.fit(X_train, y_train)

# 8️⃣ 输出最佳参数与交叉验证得分
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV R² Score: {grid_search.best_score_:.4f}")

# 9️⃣ 测试集评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Test Set Evaluation:")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
