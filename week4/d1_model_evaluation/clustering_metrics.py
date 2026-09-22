# ============================================================
# CLUSTERING MODEL EVALUATION
# Dataset: Iris
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import pandas as pd
import numpy as np


# ============================================================
# 2. Load Dataset
# ============================================================

data = load_iris()

x = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

print("Dataset Shape:")
print(x.shape)

print("\nFirst 5 Rows:")
print(x.head())


# ============================================================
# 3. Dataset Information
# ============================================================

print("\nDataset Information:")
print(x.info())

print("\nMissing Values:")
print(x.isnull().sum().sum())

print("\nDuplicate Rows:")
print(x.duplicated().sum())


# ============================================================
# 4. Feature Scaling
# ============================================================

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)


# ============================================================
# 5. K-Means Clustering
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(x_scaled)


# ============================================================
# 6. Agglomerative Clustering
# ============================================================

agglomerative = AgglomerativeClustering(
    n_clusters=3
)

agglomerative_labels = agglomerative.fit_predict(
    x_scaled
)


# ============================================================
# 7. DBSCAN Clustering
# ============================================================

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(
    x_scaled
)


# ============================================================
# 8. Store Cluster Labels
# ============================================================

cluster_predictions = {

    "K-Means": kmeans_labels,

    "Agglomerative Clustering": agglomerative_labels,

    "DBSCAN": dbscan_labels
}


# ============================================================
# 9. Display Number of Clusters
# ============================================================

print("\nNumber of Clusters")
print("=" * 70)

for model_name, labels in cluster_predictions.items():

    unique_labels = set(labels)

    # DBSCAN uses -1 for noise
    if -1 in unique_labels:
        number_of_clusters = len(
            unique_labels - {-1}
        )

        number_of_noise = list(labels).count(-1)

        print(
            f"{model_name}: "
            f"{number_of_clusters} clusters, "
            f"{number_of_noise} noise points"
        )

    else:
        number_of_clusters = len(unique_labels)

        print(
            f"{model_name}: "
            f"{number_of_clusters} clusters"
        )


# ============================================================
# 10. Model Evaluation
# ============================================================

results = []


for model_name, labels in cluster_predictions.items():

    # DBSCAN may produce noise points (-1)
    if -1 in labels:

        mask = labels != -1

        x_evaluation = x_scaled[mask]

        labels_evaluation = labels[mask]

    else:

        x_evaluation = x_scaled

        labels_evaluation = labels


    # Need at least 2 clusters for these metrics
    number_of_clusters = len(
        np.unique(labels_evaluation)
    )


    if number_of_clusters >= 2:

        silhouette = silhouette_score(
            x_evaluation,
            labels_evaluation
        )

        davies_bouldin = davies_bouldin_score(
            x_evaluation,
            labels_evaluation
        )

        calinski_harabasz = calinski_harabasz_score(
            x_evaluation,
            labels_evaluation
        )

    else:

        silhouette = np.nan

        davies_bouldin = np.nan

        calinski_harabasz = np.nan


    results.append({

        "Model": model_name,

        "Silhouette Score": silhouette,

        "Davies-Bouldin Index": davies_bouldin,

        "Calinski-Harabasz Index": calinski_harabasz

    })


# ============================================================
# 11. Comparison Table
# ============================================================

clustering_results = pd.DataFrame(
    results
)

clustering_results = clustering_results.round(4)


print("\n\nClustering Model Evaluation")
print("=" * 100)

print(
    clustering_results.to_string(
        index=False
    )
)


# ============================================================
# 12. Cluster Distribution
# ============================================================

print("\n\nCluster Distribution")
print("=" * 100)

for model_name, labels in cluster_predictions.items():

    print(f"\n{model_name}")

    print(
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )