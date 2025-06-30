import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
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

# 3️⃣ 标准化
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target]

# 4️⃣ 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ 构造基础 GBDT 模型
gbdt = GradientBoostingClassifier(random_state=42)

# 6️⃣ 设置参数搜索空间
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}

# 7️⃣ Grid Search + CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=gbdt,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# 8️⃣ 执行搜索
grid_search.fit(X_train, y_train)

# 9️⃣ 输出最优参数
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 🔟 用最优模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 🔁 模型评估
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




