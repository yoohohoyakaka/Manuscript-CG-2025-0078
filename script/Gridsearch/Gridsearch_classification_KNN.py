import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ 特征与标签
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"

# 标准化特征（KNN对尺度敏感）
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target]

# 3️⃣ 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ 构建基础模型
knn = KNeighborsClassifier()

# 5️⃣ 构造超参数搜索空间
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # p=1: 曼哈顿距离, p=2: 欧氏距离
}

# 6️⃣ GridSearchCV + StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# 7️⃣ 运行 Grid Search
grid_search.fit(X_train, y_train)


# 8️⃣ 输出调参结果
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")


# 9️⃣ 使用最优模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 🔟 模型评估
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

