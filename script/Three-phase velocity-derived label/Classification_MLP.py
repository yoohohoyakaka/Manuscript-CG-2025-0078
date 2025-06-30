# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

print("==========🚩 Biot-Based Hydrate Classification (MLP) ==========")

# 1. Load dataset
file_path = "dataset_1.xlsx"
df = pd.read_excel(file_path)

# 2. Define features and target
features = ["ILD", "Cl", "Sw"]
target = "hydrate"
X = df[features]
y = df[target]

# 3. Train-test split (stratified)
print("\n==========🚩 Non-group mode (SMOTE) ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Define MLP model
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-5,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

# 6. Cross-validation on original (non-SMOTE) data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(mlp_classifier, X, y, cv=skf, scoring="accuracy")
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 7. Train with SMOTE-resampled data
mlp_classifier.fit(X_train_resampled, y_train_resampled)

# 8. Predict on test set
y_pred = mlp_classifier.predict(X_test)
y_proba = mlp_classifier.predict_proba(X_test)[:, 1]

# 9. Evaluation
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = auc(fpr, tpr)
print(f"AUC: {auc_score:.4f}")

