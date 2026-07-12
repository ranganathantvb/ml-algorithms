import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------
# 1. Sample dataset
# Use case: Predict release risk
# 0 = Low Risk, 1 = High Risk
# ---------------------------------------------------

data = {
    "code_churn": [
        120, 300, 250, 500, 700, 150, 400, 650, 220, 800,
        180, 520, 610, 90, 760, 340, 280, 690, 130, 570
    ],
    "failed_tests": [
        2, 5, 4, 8, 10, 1, 6, 9, 3, 12,
        2, 7, 9, 1, 11, 5, 4, 10, 1, 8
    ],
    "test_coverage": [
        85, 75, 78, 65, 60, 90, 70, 62, 82, 55,
        88, 67, 63, 92, 58, 74, 80, 61, 89, 66
    ],
    "previous_incidents": [
        1, 2, 1, 4, 5, 0, 3, 4, 1, 6,
        0, 3, 4, 0, 5, 2, 1, 5, 0, 4
    ],
    "release_risk": [
        0, 0, 0, 1, 1, 0, 1, 1, 0, 1,
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1
    ]
}
df = pd.DataFrame(data)

# ---------------------------------------------------
# 2. Define input features and target
# ---------------------------------------------------

X = df[["code_churn", "failed_tests", "test_coverage", "previous_incidents"]]
y = df["release_risk"]

# ---------------------------------------------------
# 3. Train-test split
# stratify=y keeps class balance in train and test
# -----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------
# 4. Define models
# ---------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced"))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        max_depth=5,
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()), ("model", SVC(probability=True, class_weight="balanced"))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=3))
    ]),
    "Naive Bayes": GaussianNB(),
    "XGBoost": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ),
}

# ---------------------------------------------------
# 5. Train, predict, and evaluate
# ---------------------------------------------------
results = []
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = (
        roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        if hasattr(model, "predict_proba")
        else None
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "roc_auc": roc_auc
    })

results_df = pd.DataFrame(results)
print("Model comparison:")
print(results_df)

model = models["Random Forest"]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
