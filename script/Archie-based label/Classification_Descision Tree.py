import pandas as pd
import numpy as np
import time
import tracemalloc
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GroupShuffleSplit, StratifiedGroupKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)

# ==========🚩 Standard (non-grouped) validation ==========
print("==========🚩 Standard (non-grouped) validation ==========")

# Start tracking time and memory
start_time = time.time()
tracemalloc.start()

# Load data
file_path = "dataset_0227.xlsx"
df = pd.read_excel(file_path)

# Select features and target
features = ["TDEP", "VELP", "VELS"]
target = "hydrate"
X = df[features]
y = df[target]

# Define decision tree model
dt_classifier = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=10,
    random_state=42
)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt_classifier, X, y, cv=skf, scoring='accuracy')
print(f"Cross-validation Accuracy (non-group): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model training and evaluation
dt_classifier.fit(X_train, y_train)
y_pred_non_group = dt_classifier.predict(X_test)
y_proba_non_group = dt_classifier.predict_proba(X_test)[:, 1]
auc_non_group = roc_auc_score(y_test, y_proba_non_group)

print(f"Test Set Accuracy (non-group): {accuracy_score(y_test, y_pred_non_group):.4f}")
print(f"AUC (non-group): {auc_non_group:.4f}")
print("\nClassification Report (non-group):")
print(classification_report(y_test, y_pred_non_group))

# Runtime and memory usage
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\n⏱️ Execution Time (non-group): {end_time - start_time:.2f} seconds")
print(f"🧠 Peak Memory Usage (non-group): {peak / 1024 / 1024:.2f} MB")

# ==========🚩 Group-based validation ==========
print("\n==========🚩 Group-based validation ==========")

# Define depth-based groups (every 3 meters)
groups = (df["TDEP"] // 3).astype(int)

# Group-based train-test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train_g, X_test_g = X.iloc[train_idx], X.iloc[test_idx]
y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]
group_train = groups.iloc[train_idx]

# Group-aware cross-validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_group = cross_val_score(
    dt_classifier, X_train_g, y_train_g,
    cv=sgkf.split(X_train_g, y_train_g, group_train),
    scoring="accuracy"
)
print(f"Cross-validation Accuracy (group): {cv_scores_group.mean():.4f} ± {cv_scores_group.std():.4f}")

# Train and evaluate
dt_classifier.fit(X_train_g, y_train_g)
y_pred_group = dt_classifier.predict(X_test_g)
y_proba_group = dt_classifier.predict_proba(X_test_g)[:, 1]
auc_group = roc_auc_score(y_test_g, y_proba_group)

print(f"Test Set Accuracy (group): {accuracy_score(y_test_g, y_pred_group):.4f}")
print(f"AUC (group): {auc_group:.4f}")
print("\nClassification Report (group):")
print(classification_report(y_test_g, y_pred_group))
