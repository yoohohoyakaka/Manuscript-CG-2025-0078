import pandas as pd
import numpy as np
import time
import tracemalloc
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GroupShuffleSplit, StratifiedGroupKFold
)
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# ========= Non-grouped mode =========
print("========== Non-grouped Mode ==========")

# ⏱️ Start timing and memory tracking
start_time = time.time()
tracemalloc.start()

# Load dataset
file_path = "dataset_0227.xlsx"
df = pd.read_excel(file_path)

# Define features and target
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"
X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define XGBoost classifier
xgb_classifier = XGBClassifier(
    learning_rate=0.05,
    max_depth=7,
    n_estimators=500,
    random_state=42
)

# Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=skf, scoring="accuracy")
print(f"Cross-validation Accuracy (non-group): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train and evaluate model
xgb_classifier.fit(X_train, y_train)
y_pred_non_group = xgb_classifier.predict(X_test)
y_proba_non_group = xgb_classifier.predict_proba(X_test)[:, 1]

print(f"Test Accuracy (non-group): {accuracy_score(y_test, y_pred_non_group):.4f}")
print("\nClassification Report (non-group):")
print(classification_report(y_test, y_pred_non_group))

# Compute AUC
fpr_non_group, tpr_non_group, _ = roc_curve(y_test, y_proba_non_group)
auc_non_group = auc(fpr_non_group, tpr_non_group)
print(f"AUC (non-group): {auc_non_group:.4f}")

# Report execution time and peak memory usage
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nExecution Time (non-group): {end_time - start_time:.2f} seconds")
print(f"Peak Memory Usage (non-group): {peak / 1024 / 1024:.2f} MB")

# ========= Grouped mode =========
print("\n========== Grouped Mode ==========")

# Define groups (e.g., every 3 meters as a group)
groups = (df["TDEP"] // 3).astype(int)

# Group-aware train-test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y, groups))
X_train_g, X_test_g = X_scaled[train_idx], X_scaled[test_idx]
y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]
group_train = groups.iloc[train_idx]

# Group-aware cross-validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_group = cross_val_score(
    xgb_classifier, X_train_g, y_train_g,
    cv=sgkf.split(X_train_g, y_train_g, group_train),
    scoring="accuracy"
)
print(f"Cross-validation Accuracy (group): {cv_scores_group.mean():.4f} ± {cv_scores_group.std():.4f}")

# Train and evaluate model on grouped test set
xgb_classifier.fit(X_train_g, y_train_g)
y_pred_group = xgb_classifier.predict(X_test_g)
y_proba_group = xgb_classifier.predict_proba(X_test_g)[:, 1]

print(f"Test Accuracy (group): {accuracy_score(y_test_g, y_pred_group):.4f}")
print("\nClassification Report (group):")
print(classification_report(y_test_g, y_pred_group))

# Compute AUC for grouped mode
fpr_group, tpr_group, _ = roc_curve(y_test_g, y_proba_group)
auc_group = auc(fpr_group, tpr_group)
print(f"AUC (group): {auc_group:.4f}")
