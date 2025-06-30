import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"  # 修改为你的路径
df = pd.read_excel(file_path)

# 2️⃣ 特征与标签
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"

X = df[features]
y = df[target]

# 3️⃣ 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ 划分训练集 & 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ 定义 CatBoost 基础模型
cat_model = CatBoostClassifier(
    verbose=0,
    random_seed=42
)

# 6️⃣ 构造参数网格
param_grid = {
    'iterations': [100, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'depth': [4, 6, 8, 10]
}

# 7️⃣ 设置 GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# 8️⃣ 执行搜索
grid_search.fit(X_train, y_train)

# 9️⃣ 输出结果
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 🔟 最优模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 🔁 模型评估
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


