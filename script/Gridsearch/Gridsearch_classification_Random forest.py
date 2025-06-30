import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ 数据预处理
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"
X = df[features]
y = df[target]

# 3️⃣ 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 定义模型和参数网格
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 3, 5,10],
    'min_samples_leaf': [1, 2, 5,10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6️⃣ 网格搜索交叉验证
grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)


# 8️⃣ 输出最优参数与CV准确率
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 9️⃣ 测试集评估
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))