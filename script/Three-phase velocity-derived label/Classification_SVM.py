# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

print("==========🚩 Biot-Based Hydrate Classification (SVM) ==========")

# Step 1: Load dataset
file_path = "dataset_1.xlsx"
df = pd.read_excel(file_path)

# Step 2: Define input features and target label
features = ["ILD", "Cl", "Sw"]
target = "hydrate"
X = df[features]
y = df[target]

# Step 3: Stratified train-test split to maintain class distribution
print("\n==========🚩 Non-group mode ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Define SVM model with RBF kernel
svm_classifier = SVC(
    kernel="rbf",
    C=50,
    gamma=1,
    probability=True,
    random_state=42
)

# Step 5: Stratified 5-fold cross-validation for performance estimation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_classifier, X, y, cv=skf, scoring="accuracy")
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Step 6: Train model and make predictions
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
y_proba = svm_classifier.predict_proba(X_test)[:, 1]

# Step 7: Evaluation metrics
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = auc(fpr, tpr)
print(f"AUC: {auc_score:.4f}")
