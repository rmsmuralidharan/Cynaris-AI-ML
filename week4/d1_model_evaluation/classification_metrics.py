# ============================================================
# CLASSIFICATION MODEL EVALUATION
# Dataset: Breast Cancer Wisconsin
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import pandas as pd
import numpy as np


# ============================================================
# 2. Load Dataset
# ============================================================

data = load_breast_cancer()

x = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)

print("Dataset Shape:")
print(x.shape)

print("\nFirst 5 Rows:")
print(x.head())

print("\nTarget Classes:")
print(data.target_names)

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# 3. Basic Dataset Inspection
# ============================================================

print("\nDataset Information:")
print(x.info())

print("\nMissing Values:")
print(x.isnull().sum().sum())

print("\nDuplicate Rows:")
print(x.duplicated().sum())


# ============================================================
# 4. Train-Test Split
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(x_train.shape)

print("\nTesting Data Shape:")
print(x_test.shape)


# ============================================================
# 5. Feature Scaling
# ============================================================

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)


# ============================================================
# 6. Create Models
# ============================================================

logistic = LogisticRegression(
    max_iter=1000,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=5
)

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

svc = SVC(
    probability=True,
    random_state=42
)

gradient_boosting = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 7. Train Models
# ============================================================

logistic.fit(x_train_scaled, y_train)

knn.fit(x_train_scaled, y_train)

decision_tree.fit(x_train, y_train)

random_forest.fit(x_train, y_train)

svc.fit(x_train_scaled, y_train)

gradient_boosting.fit(x_train_scaled, y_train)


# ============================================================
# 8. Predictions
# ============================================================

y_pred_logistic = logistic.predict(x_test_scaled)

y_pred_knn = knn.predict(x_test_scaled)

y_pred_dt = decision_tree.predict(x_test)

y_pred_rf = random_forest.predict(x_test)

y_pred_svc = svc.predict(x_test_scaled)

y_pred_gb = gradient_boosting.predict(x_test_scaled)


# ============================================================
# 9. Prediction Probabilities
# ============================================================

y_prob_logistic = logistic.predict_proba(x_test_scaled)[:, 1]

y_prob_knn = knn.predict_proba(x_test_scaled)[:, 1]

y_prob_dt = decision_tree.predict_proba(x_test)[:, 1]

y_prob_rf = random_forest.predict_proba(x_test)[:, 1]

y_prob_svc = svc.predict_proba(x_test_scaled)[:, 1]

y_prob_gb = gradient_boosting.predict_proba(x_test_scaled)[:, 1]


# ============================================================
# 10. Store Predictions
# ============================================================

predictions = {
    "Logistic Regression": (
        y_pred_logistic,
        y_prob_logistic
    ),

    "KNN": (
        y_pred_knn,
        y_prob_knn
    ),

    "Decision Tree": (
        y_pred_dt,
        y_prob_dt
    ),

    "Random Forest": (
        y_pred_rf,
        y_prob_rf
    ),

    "SVC": (
        y_pred_svc,
        y_prob_svc
    ),

    "Gradient Boosting": (
        y_pred_gb,
        y_prob_gb
    )
}


# ============================================================
# 11. Model Evaluation
# ============================================================

results = []

for model_name, (y_pred, y_prob) in predictions.items():

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })


# ============================================================
# 12. Comparison Table
# ============================================================

classification_results = pd.DataFrame(results)

classification_results = classification_results.round(4)

print("\n\nClassification Model Evaluation")
print("=" * 100)

print(
    classification_results.to_string(
        index=False
    )
)


# ============================================================
# 13. Confusion Matrix
# ============================================================

print("\n\nConfusion Matrices")
print("=" * 100)

for model_name, (y_pred, y_prob) in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"\n{model_name}")
    print("-" * 50)

    print(cm)


# ============================================================
# 14. Classification Reports
# ============================================================

print("\n\nClassification Reports")
print("=" * 100)

for model_name, (y_pred, y_prob) in predictions.items():

    print(f"\n{model_name}")
    print("-" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=data.target_names
        )
    )