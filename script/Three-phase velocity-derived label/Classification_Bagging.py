# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GroupShuffleSplit, StratifiedGroupKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

print("==========🚩 Biot-Based Hydrate Classification (Bagging) ==========")

# 1️⃣ 加载数据
file_path = "dataset_1.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ 特征与标签
features = ["ILD", "Cl", "Sw"]
target = "hydrate"
X = df[features]
y = df[target]

# ==========🚩 非分组模式 ==========
print("\n==========🚩 非分组模式 ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3️⃣ 初始化 Bagging 模型（指定超参数）
base_estimator = DecisionTreeClassifier(max_depth=None)
bagging_classifier = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=500,
    max_features=3,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)

# 4️⃣ 非分组交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(bagging_classifier, X, y, cv=skf, scoring="accuracy")
print(f"Cross-validation Accuracy (non-group): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 5️⃣ 训练 + 预测
bagging_classifier.fit(X_train, y_train)
y_pred = bagging_classifier.predict(X_test)
y_proba = bagging_classifier.predict_proba(X_test)[:, 1]

# 6️⃣ 模型评估
print(f"\nTest Set Accuracy (non-group): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report (non-group):")
print(classification_report(y_test, y_pred))
cm_non_group = confusion_matrix(y_test, y_pred)
fpr_non_group, tpr_non_group, _ = roc_curve(y_test, y_proba)
auc_non_group = auc(fpr_non_group, tpr_non_group)
print(f"AUC (non-group): {auc_non_group:.4f}")

# ==========🚩 分组模式 ==========
print("\n==========🚩 分组模式 ==========")

groups = (df["TDEP"] // 3).astype(int)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train_g, X_test_g = X.iloc[train_idx], X.iloc[test_idx]
y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]
group_train = groups.iloc[train_idx]

# 7️⃣ 分组交叉验证
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_group = cross_val_score(
    bagging_classifier, X_train_g, y_train_g,
    cv=sgkf.split(X_train_g, y_train_g, group_train),
    scoring="accuracy"
)
print(f"Cross-validation Accuracy (group): {cv_scores_group.mean():.4f} ± {cv_scores_group.std():.4f}")

# 8️⃣ 模型训练 + 预测
bagging_classifier.fit(X_train_g, y_train_g)
y_pred_group = bagging_classifier.predict(X_test_g)
y_proba_group = bagging_classifier.predict_proba(X_test_g)[:, 1]

# 9️⃣ 模型评估
print(f"\nTest Set Accuracy (group): {accuracy_score(y_test_g, y_pred_group):.4f}")
print("\nClassification Report (group):")
print(classification_report(y_test_g, y_pred_group))
cm_group = confusion_matrix(y_test_g, y_pred_group)
fpr_group, tpr_group, _ = roc_curve(y_test_g, y_proba_group)
auc_group = auc(fpr_group, tpr_group)
print(f"AUC (group): {auc_group:.4f}")

