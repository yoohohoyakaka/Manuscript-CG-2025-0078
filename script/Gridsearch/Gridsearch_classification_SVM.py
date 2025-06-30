import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import time
import psutil
import os

# 1️⃣ 加载数据
file_path = "dataset_0227.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ 特征与目标
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"
X = df[features]
y = df[target]

# 3️⃣ 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5️⃣ 定义 SVC 与参数网格
svc = SVC(probability=True, class_weight='balanced', random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 7️⃣ GridSearchCV 调参
grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 8️⃣ 输出最优结果
end_time = time.time()
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")

# 9️⃣ 测试集评估
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)
y_proba = best_svm.predict_proba(X_test)[:, 1]

print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))