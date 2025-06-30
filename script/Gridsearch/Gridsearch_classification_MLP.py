import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc
)

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ 特征与标签
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"
X = df[features]
y = df[target]

# 3️⃣ 标准化（MLP对数值范围敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 5️⃣ 设置参数搜索空间
param_grid = {
    'hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'tanh'],
}

mlp = MLPClassifier(max_iter=500, solver='adam', random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    mlp, param_grid, scoring='accuracy', cv=cv,
    n_jobs=-1, verbose=1
)

# 7️⃣ 运行 Grid Search
grid_search.fit(X_train, y_train)


# 7️⃣ 输出结果
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 8️⃣ 最佳模型预测
best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test)
y_proba = best_mlp.predict_proba(X_test)[:, 1]

# 9️⃣ 评估
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

