import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc
)

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"  # 修改为你的路径
df = pd.read_excel(file_path)

# 2️⃣ 特征与标签选择
features = ["TDEP", "VELP", "VELS"]  # 可根据需要调整
target = "hydrate"

X = df[features]
y = df[target]

# 3️⃣ 数据划分（Stratified 保持类别平衡）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ 定义基础模型
base_estimator = DecisionTreeClassifier(random_state=42)

adaboost = AdaBoostClassifier(
    estimator=base_estimator,
    random_state=42
)

# 5️⃣ 定义超参数搜索空间
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.1, 0.3, 0.5, 1.0],
    'estimator__max_depth': [2, 3, 4, 5]
}

# 6️⃣ 构建 GridSearchCV 对象
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=adaboost,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 7️⃣ 执行搜索
grid_search.fit(X_train, y_train)

# 8️⃣ 打印最优参数与得分
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 9️⃣ 用最优模型做预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 🔟 测试集评估指标
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

