import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

print("==========🚩 Biot-Based Hydrate Classification using GBDT ==========")

# 1. Load dataset
file_path = "dataset_1.xlsx"
df = pd.read_excel(file_path)

# 2. Define features and target
features = ["ILD", "Cl", "Sw"]
target = "hydrate"
X = df[features]
y = df[target]

# 3. Stratified train-test split
print("\n==========🚩 Non-group mode ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Initialize GBDT classifier
gbdt_classifier = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=7,
    n_estimators=500,
    random_state=42
)

# 5. Stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gbdt_classifier, X, y, cv=skf, scoring="accuracy")
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 6. Train and predict
gbdt_classifier.fit(X_train, y_train)
y_pred = gbdt_classifier.predict(X_test)
y_proba = gbdt_classifier.predict_proba(X_test)[:, 1]

# 7. Evaluation
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = auc(fpr, tpr)
print(f"AUC: {auc_score:.4f}")
