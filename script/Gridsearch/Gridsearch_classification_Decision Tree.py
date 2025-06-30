import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 2️⃣ 特征与目标变量
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"

X = df[features]
y = df[target]

# 3️⃣ 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ 定义基础模型
tree_model = DecisionTreeClassifier(random_state=42)

# 5️⃣ 构造参数网格
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6]
}

# 6️⃣ Grid SearchCV + KFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=tree_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# 7️⃣ 执行调参
grid_search.fit(X_train, y_train)

# 8️⃣ 输出最佳结果
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 🔟 使用最优模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 🔁 模型评估
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




