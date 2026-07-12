"""
Unsupervised Learning Example: Release Pattern Discovery and Anomaly Detection

Use case
--------
In enterprise quality engineering, many engineering records may not have labels.
Unsupervised learning helps discover hidden groups and unusual releases from
signals such as code churn, failed tests, coverage, prior incidents, deployment
frequency, and New Relic alerts.

This script demonstrates:
1. K-Means clustering for release pattern grouping.
2. Tuning the number of clusters using silhouette and Davies-Bouldin scores.
3. PCA for two-dimensional visualization-ready components.
4. DBSCAN for density-based clusters and outlier discovery.
5. Isolation Forest for anomaly detection.
6. Business interpretation of clusters for interview preparation.

Run:
    python unsupervised_release_patterns.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "code_churn",
    "failed_tests",
    "test_coverage",
    "previous_incidents",
    "deployment_frequency",
    "new_relic_alerts",
]


@dataclass(frozen=True)
class ClusteringScore:
    cluster_count: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    inertia: float


def build_release_dataset() -> pd.DataFrame:
    """Create unlabeled release-quality data for clustering and anomaly detection."""

    records = [
        # Low-risk style releases: low churn, few failed tests, high coverage
        ["REL-001", 120, 2, 88, 0, 1, 0],
        ["REL-002", 180, 2, 86, 1, 1, 0],
        ["REL-003", 220, 3, 82, 1, 2, 1],
        ["REL-004", 260, 4, 80, 1, 2, 1],
        ["REL-005", 150, 1, 91, 0, 1, 0],
        ["REL-006", 200, 2, 87, 0, 2, 0],
        # Medium-risk style releases: moderate churn and test failures
        ["REL-007", 360, 5, 76, 2, 3, 1],
        ["REL-008", 410, 6, 73, 2, 3, 2],
        ["REL-009", 450, 6, 71, 2, 4, 2],
        ["REL-010", 390, 5, 74, 2, 3, 1],
        ["REL-011", 480, 7, 69, 3, 4, 2],
        ["REL-012", 430, 6, 72, 2, 4, 2],
        # High-risk style releases: high churn, many failed tests, lower coverage
        ["REL-013", 560, 8, 65, 3, 5, 3],
        ["REL-014", 610, 9, 63, 4, 5, 4],
        ["REL-015", 680, 10, 60, 4, 6, 5],
        ["REL-016", 740, 11, 57, 5, 6, 5],
        ["REL-017", 650, 9, 62, 4, 5, 4],
        ["REL-018", 700, 10, 59, 5, 6, 5],
        # Extreme / unusual release patterns
        ["REL-019", 950, 15, 48, 8, 8, 9],
        ["REL-020", 80, 0, 94, 0, 1, 0],
        ["REL-021", 870, 13, 52, 7, 7, 8],
        ["REL-022", 300, 9, 88, 0, 1, 6],  # unusual: high alerts but high coverage
    ]

    return pd.DataFrame(records, columns=["release_id", *FEATURE_COLUMNS])


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """Scale features so distance-based algorithms behave correctly."""

    scaler = StandardScaler()
    return scaler.fit_transform(df[FEATURE_COLUMNS])


def tune_kmeans_cluster_count(x_scaled: np.ndarray) -> list[ClusteringScore]:
    """Compare K-Means cluster counts using common unsupervised metrics."""

    scores: list[ClusteringScore] = []

    print("K-Means cluster-count tuning")
    print("----------------------------")
    print("K | Silhouette | Davies-Bouldin | Calinski-Harabasz | Inertia")

    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x_scaled)

        score = ClusteringScore(
            cluster_count=k,
            silhouette=silhouette_score(x_scaled, labels),
            davies_bouldin=davies_bouldin_score(x_scaled, labels),
            calinski_harabasz=calinski_harabasz_score(x_scaled, labels),
            inertia=model.inertia_,
        )
        scores.append(score)

        print(
            f"{k} | {score.silhouette:>10.3f} | {score.davies_bouldin:>14.3f} | "
            f"{score.calinski_harabasz:>17.2f} | {score.inertia:>7.2f}"
        )

    return scores


def run_kmeans(df: pd.DataFrame, x_scaled: np.ndarray, n_clusters: int = 3) -> pd.DataFrame:
    """Run K-Means and summarize cluster profiles."""

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    result = df.copy()
    result["kmeans_cluster"] = model.fit_predict(x_scaled)

    print(f"\nK-Means clustering with k={n_clusters}")
    print("--------------------------------")
    print(result[["release_id", *FEATURE_COLUMNS, "kmeans_cluster"]].to_string(index=False))

    print("\nCluster profile averages")
    print("------------------------")
    profile = result.groupby("kmeans_cluster")[FEATURE_COLUMNS].mean().round(2)
    print(profile.to_string())

    print("\nBusiness interpretation guide")
    print("-----------------------------")
    for cluster_id, row in profile.iterrows():
        if row["code_churn"] > 600 or row["failed_tests"] > 9 or row["new_relic_alerts"] > 5:
            meaning = "High-risk or unstable release pattern"
        elif row["code_churn"] < 300 and row["failed_tests"] <= 4 and row["test_coverage"] >= 80:
            meaning = "Low-risk release pattern"
        else:
            meaning = "Medium-risk release pattern that may need targeted review"
        print(f"Cluster {cluster_id}: {meaning}")

    return result


def run_pca(result: pd.DataFrame, x_scaled: np.ndarray) -> pd.DataFrame:
    """Reduce features to two principal components for visualization-ready output."""

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(x_scaled)

    result = result.copy()
    result["pca_1"] = components[:, 0]
    result["pca_2"] = components[:, 1]

    explained = pca.explained_variance_ratio_

    print("\nPCA: two-dimensional representation")
    print("-----------------------------------")
    print(f"Explained variance by PCA-1: {explained[0]:.2%}")
    print(f"Explained variance by PCA-2: {explained[1]:.2%}")
    print(f"Total explained variance   : {explained.sum():.2%}")
    print(result[["release_id", "kmeans_cluster", "pca_1", "pca_2"]].round(3).to_string(index=False))

    return result


def run_dbscan(result: pd.DataFrame, x_scaled: np.ndarray) -> pd.DataFrame:
    """Run DBSCAN for natural clusters and noise/outlier detection."""

    dbscan = DBSCAN(eps=1.45, min_samples=3)
    result = result.copy()
    result["dbscan_cluster"] = dbscan.fit_predict(x_scaled)

    print("\nDBSCAN density-based clustering")
    print("--------------------------------")
    print("DBSCAN labels: -1 means noise/outlier")
    print(result[["release_id", "kmeans_cluster", "dbscan_cluster"]].to_string(index=False))

    outliers = result[result["dbscan_cluster"] == -1]
    print("\nDBSCAN outlier candidates")
    print("-------------------------")
    if outliers.empty:
        print("No DBSCAN outliers detected with the current eps/min_samples settings.")
    else:
        print(outliers[["release_id", *FEATURE_COLUMNS]].to_string(index=False))

    return result


def run_isolation_forest(result: pd.DataFrame, x_scaled: np.ndarray) -> pd.DataFrame:
    """Run Isolation Forest for anomaly detection."""

    model = IsolationForest(
        n_estimators=150,
        contamination=0.14,  # expected anomaly proportion, tune in real data
        n_jobs=1,
        random_state=42,
    )

    result = result.copy()
    result["isolation_forest_label"] = model.fit_predict(x_scaled)
    result["anomaly_score"] = model.decision_function(x_scaled)
    result["is_anomaly"] = result["isolation_forest_label"].map({1: "normal", -1: "anomaly"})

    print("\nIsolation Forest anomaly detection")
    print("----------------------------------")
    print(
        result[["release_id", *FEATURE_COLUMNS, "anomaly_score", "is_anomaly"]]
        .sort_values("anomaly_score")
        .round(3)
        .to_string(index=False)
    )

    return result


def print_interview_summary() -> None:
    print("\nInterview takeaway")
    print("------------------")
    print(
        "Use unsupervised learning when labels are unavailable but engineering data is rich. "
        "K-Means is a strong starting point for release or defect grouping, DBSCAN helps "
        "detect noisy/outlier patterns, PCA reduces many metrics into visualization-ready "
        "components, and Isolation Forest is useful for anomaly detection. Always validate "
        "clusters with domain experts before using them in release governance."
    )


def main() -> None:
    df = build_release_dataset()
    x_scaled = scale_features(df)

    print("Unsupervised Learning: Release Pattern Discovery")
    print("================================================")
    print(f"Total unlabeled release records: {len(df)}")
    print("Features used:", ", ".join(FEATURE_COLUMNS))
    print()

    tune_kmeans_cluster_count(x_scaled)
    clustered = run_kmeans(df, x_scaled, n_clusters=3)
    with_pca = run_pca(clustered, x_scaled)
    with_dbscan = run_dbscan(with_pca, x_scaled)
    run_isolation_forest(with_dbscan, x_scaled)
    print_interview_summary()


if __name__ == "__main__":
    main()
