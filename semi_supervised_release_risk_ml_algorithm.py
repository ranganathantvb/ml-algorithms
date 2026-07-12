"""
Semi-Supervised Learning Example: Release Risk Classification

Use case
--------
In enterprise quality engineering, many Jira/release records may not be labeled as
low risk or high risk. Semi-supervised learning helps when we have:

    small trusted labeled data + larger unlabeled data

This script demonstrates:
1. A supervised baseline using only labeled records.
2. A semi-supervised SelfTrainingClassifier using labeled + unlabeled records.
3. Confidence-threshold tuning for pseudo-label quality.
4. LabelSpreading as a similarity-based semi-supervised alternative.
5. Evaluation on a trusted manually labeled holdout test set.

Labels:
    0  = low_risk
    1  = high_risk
    -1 = unlabeled record

Run:
    python semi_supervised_release_risk.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier


FEATURE_COLUMNS = [
    "code_churn",
    "failed_tests",
    "test_coverage",
    "previous_incidents",
    "deployment_frequency",
    "new_relic_alerts",
]


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None


def build_release_risk_dataset() -> pd.DataFrame:
    """Create a small but explainable release-risk dataset.

    The final_test_label column is kept only for trusted evaluation.
    The semi_supervised_label column simulates the real-world case where many
    historical records are not labeled yet and therefore appear as -1.
    """

    records = [
        # Low-risk style releases
        [120, 2, 88, 0, 1, 0, 0],
        [180, 2, 86, 1, 1, 0, 0],
        [220, 3, 82, 1, 2, 1, 0],
        [260, 4, 80, 1, 2, 1, 0],
        [310, 4, 78, 1, 2, 1, 0],
        [150, 1, 91, 0, 1, 0, 0],
        [200, 2, 87, 0, 2, 0, 0],
        [280, 3, 83, 1, 2, 1, 0],
        [330, 4, 79, 1, 3, 1, 0],
        [240, 3, 84, 1, 1, 0, 0],
        # High-risk style releases
        [500, 7, 68, 3, 4, 3, 1],
        [560, 8, 65, 3, 5, 3, 1],
        [610, 9, 63, 4, 5, 4, 1],
        [680, 10, 60, 4, 6, 5, 1],
        [740, 11, 57, 5, 6, 5, 1],
        [820, 13, 54, 6, 7, 6, 1],
        [530, 8, 66, 3, 4, 3, 1],
        [650, 9, 62, 4, 5, 4, 1],
        [790, 12, 56, 5, 7, 6, 1],
        [700, 10, 59, 5, 6, 5, 1],
        # Borderline releases
        [420, 5, 73, 2, 3, 2, 0],
        [450, 6, 71, 2, 4, 2, 0],
        [470, 6, 69, 3, 4, 2, 1],
        [490, 7, 67, 3, 4, 3, 1],
        [390, 5, 74, 2, 3, 1, 0],
        [515, 7, 66, 3, 5, 3, 1],
    ]

    df = pd.DataFrame(records, columns=[*FEATURE_COLUMNS, "final_test_label"])

    # Simulate partial labeling: some labels are known, many are missing.
    known_label_indexes = {0, 1, 5, 8, 10, 12, 15, 18, 20, 23}
    df["semi_supervised_label"] = -1
    for idx in known_label_indexes:
        df.loc[idx, "semi_supervised_label"] = df.loc[idx, "final_test_label"]

    return df


def evaluate_classifier(
    name: str,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Iterable[float] | None = None,
) -> Metrics:
    """Evaluate a binary classifier and print business-friendly results."""

    y_true_array = np.asarray(list(y_true))
    y_pred_array = np.asarray(list(y_pred))

    metrics = Metrics(
        accuracy=accuracy_score(y_true_array, y_pred_array),
        precision=precision_score(y_true_array, y_pred_array, zero_division=0),
        recall=recall_score(y_true_array, y_pred_array, zero_division=0),
        f1=f1_score(y_true_array, y_pred_array, zero_division=0),
        roc_auc=None,
    )

    if y_prob is not None and len(set(y_true_array)) > 1:
        metrics = Metrics(
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            roc_auc=roc_auc_score(y_true_array, np.asarray(list(y_prob))),
        )

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {metrics.accuracy:.2f}")
    print(f"Precision: {metrics.precision:.2f}")
    print(f"Recall   : {metrics.recall:.2f}")
    print(f"F1 Score : {metrics.f1:.2f}")
    if metrics.roc_auc is not None:
        print(f"ROC-AUC  : {metrics.roc_auc:.2f}")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(confusion_matrix(y_true_array, y_pred_array))

    return metrics


def create_self_training_classifier(threshold: float) -> SelfTrainingClassifier:
    """Create a semi-supervised model using RandomForest as the base estimator."""

    base_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    # sklearn 1.6+ uses estimator. Older versions used base_estimator.
    try:
        return SelfTrainingClassifier(
            estimator=base_model,
            threshold=threshold,
            max_iter=10,
            verbose=False,
        )
    except TypeError:
        return SelfTrainingClassifier(
            base_estimator=base_model,  # type: ignore[call-arg]
            threshold=threshold,
            max_iter=10,
            verbose=False,
        )


def run_supervised_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Train a supervised model using only manually labeled training records."""

    labeled_train = train_df[train_df["semi_supervised_label"] != -1]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(labeled_train[FEATURE_COLUMNS], labeled_train["semi_supervised_label"])
    predictions = model.predict(test_df[FEATURE_COLUMNS])
    probabilities = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    evaluate_classifier(
        "Supervised Baseline: Logistic Regression on labeled records only",
        test_df["final_test_label"],
        predictions,
        probabilities,
    )


def tune_self_training_thresholds(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Compare confidence thresholds for pseudo-labeling."""

    print("\nSelf-training threshold comparison")
    print("----------------------------------")
    print("Threshold | Accuracy | Precision | Recall | F1")

    for threshold in [0.60, 0.70, 0.80, 0.90]:
        model = create_self_training_classifier(threshold=threshold)
        model.fit(train_df[FEATURE_COLUMNS], train_df["semi_supervised_label"])

        predictions = model.predict(test_df[FEATURE_COLUMNS])

        print(
            f"{threshold:>9.2f} | "
            f"{accuracy_score(test_df['final_test_label'], predictions):>8.2f} | "
            f"{precision_score(test_df['final_test_label'], predictions, zero_division=0):>9.2f} | "
            f"{recall_score(test_df['final_test_label'], predictions, zero_division=0):>6.2f} | "
            f"{f1_score(test_df['final_test_label'], predictions, zero_division=0):>4.2f}"
        )


def run_best_self_training_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Train and evaluate the selected self-training model."""

    model = create_self_training_classifier(threshold=0.80)
    model.fit(train_df[FEATURE_COLUMNS], train_df["semi_supervised_label"])

    predictions = model.predict(test_df[FEATURE_COLUMNS])
    probabilities = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    evaluate_classifier(
        "Semi-Supervised: Self-Training Random Forest, threshold=0.80",
        test_df["final_test_label"],
        predictions,
        probabilities,
    )

    comparison = test_df[[*FEATURE_COLUMNS, "final_test_label"]].copy()
    comparison["predicted_label"] = predictions
    comparison["predicted_probability_high_risk"] = np.round(probabilities, 3)
    print("\nActual vs predicted release risk")
    print("--------------------------------")
    print(comparison.to_string(index=False))


def run_label_spreading(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Train and evaluate LabelSpreading.

    LabelSpreading is useful when similar release records should share similar
    risk labels. Scaling is important because it uses distance/similarity.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LabelSpreading(
                    kernel="rbf",
                    gamma=0.35,
                    alpha=0.2,
                    max_iter=50,
                ),
            ),
        ]
    )

    model.fit(train_df[FEATURE_COLUMNS], train_df["semi_supervised_label"])
    predictions = model.predict(test_df[FEATURE_COLUMNS])

    evaluate_classifier(
        "Semi-Supervised: LabelSpreading similarity-based model",
        test_df["final_test_label"],
        predictions,
    )


def main() -> None:
    df = build_release_risk_dataset()

    # Keep a trusted holdout set with true labels for evaluation.
    # In real projects, this should be manually verified data.
    test_indexes = [2, 3, 6, 9, 11, 14, 17, 21, 22, 24, 25]
    test_df = df.loc[test_indexes].copy()
    train_df = df.drop(index=test_indexes).copy()

    print("Semi-Supervised Learning: Release Risk Classification")
    print("====================================================")
    print(f"Total records             : {len(df)}")
    print(f"Training records          : {len(train_df)}")
    print(f"Trusted test records      : {len(test_df)}")
    print(f"Labeled training records  : {(train_df['semi_supervised_label'] != -1).sum()}")
    print(f"Unlabeled training records: {(train_df['semi_supervised_label'] == -1).sum()}")

    run_supervised_baseline(train_df, test_df)
    tune_self_training_thresholds(train_df, test_df)
    run_best_self_training_model(train_df, test_df)
    run_label_spreading(train_df, test_df)

    print("\nInterview takeaway")
    print("------------------")
    print(
        "Use semi-supervised learning when labeled release/defect/RCA data is limited "
        "but many unlabeled engineering records are available. Always evaluate on "
        "trusted manually labeled test data, tune the pseudo-label confidence threshold, "
        "and add human review for high-risk enterprise decisions."
    )


if __name__ == "__main__":
    main()