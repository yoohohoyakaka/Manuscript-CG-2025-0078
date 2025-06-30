import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc
)

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"  # 替换为你的 Excel 路径
df = pd.read_excel(file_path)

# 2️⃣ 特征与目标变量
features = ["TDEP", "VELP", "VELS"]  # 可根据需要调整
target = "hydrate"

X = df[features]
y = df[target]

# 3️⃣ 划分训练集与测试集（保持分布）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ 定义基础决策树
base_tree = DecisionTreeClassifier(random_state=42)

# 5️⃣ 定义 Bagging 分类器
bagging = BaggingClassifier(
    estimator=base_tree,
    bootstrap=True,
    random_state=42
)

# 6️⃣ 构造参数网格
param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [1, 2, 3],
    'estimator__max_depth': [3, 5, 7]
}

# 7️⃣ Grid Search + 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=bagging,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# 8️⃣ 执行搜索
grid_search.fit(X_train, y_train)

# 9️⃣ 输出最佳参数和得分
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 🔟 最优模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 🔁 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

