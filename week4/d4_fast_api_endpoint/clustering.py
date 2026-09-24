import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "d4_fast_api_endpoint",
    "data",
    "Telco-Customer-Churn.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after cleaning:")
print(df.shape)


# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[features].copy()

print("\nSelected Features:")
print(X.head())

print("\nFeature Statistics:")
print(X.describe())


# ============================================================
# 5. EDA
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# Distribution plots
for column in features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")

    plt.tight_layout()

    filename = column.replace(" ", "_") + "_distribution.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename)
    )

    plt.close()


# Relationship: Tenure vs MonthlyCharges
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges"
)

plt.title("Tenure vs Monthly Charges")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "tenure_vs_monthly_charges.png"
    )
)

plt.close()


# MonthlyCharges vs TotalCharges
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="MonthlyCharges",
    y="TotalCharges"
)

plt.title("Monthly Charges vs Total Charges")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "monthly_vs_total_charges.png"
    )
)

plt.close()


# Correlation matrix
plt.figure(figsize=(8, 6))

sns.heatmap(
    X.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_matrix.png"
    )
)

plt.close()


# ============================================================
# 6. FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)

print("\nScaled Features:")
print(X_scaled.head())


# ============================================================
# 7. ELBOW METHOD
# ============================================================

print("\n" + "=" * 60)
print("ELBOW METHOD")
print("=" * 60)

inertias = []

K_range = range(2, 11)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

    print(
        f"K = {k} | Inertia = {model.inertia_:.2f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    list(K_range),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.xticks(list(K_range))

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "elbow_method.png"
    )
)

plt.close()


# ============================================================
# 8. SILHOUETTE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SILHOUETTE ANALYSIS")
print("=" * 60)

silhouette_scores = {}

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k} | Silhouette Score = {score:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    list(silhouette_scores.keys()),
    list(silhouette_scores.values()),
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Analysis")

plt.xticks(list(K_range))

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "silhouette_scores.png"
    )
)

plt.close()


# ============================================================
# 9. SELECT K
# ============================================================

# Select K with the highest silhouette score
best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

print("\nBest K based on Silhouette Score:")
print(best_k)


# ============================================================
# 10. FINAL K-MEANS MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL K-MEANS MODEL")
print("=" * 60)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)


# Add cluster to dataframe
df["Cluster"] = clusters


print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())


# ============================================================
# 11. FINAL SILHOUETTE SCORE
# ============================================================

final_score = silhouette_score(
    X_scaled,
    clusters
)

print(
    f"\nFinal Silhouette Score: {final_score:.4f}"
)


# ============================================================
# 12. CLUSTER PROFILING
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER PROFILING")
print("=" * 60)

cluster_profile = df.groupby("Cluster")[features].mean()

print("\nCluster Profile:")
print(cluster_profile)


# Save cluster profile
cluster_profile.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_profile.csv"
    )
)


# ============================================================
# 13. PCA VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("PCA VISUALIZATION")
print("=" * 60)

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Cluster"] = clusters


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1",
    s=70
)

plt.title("Customer Segmentation using K-Means")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "customer_clusters_pca.png"
    )
)

plt.close()


# ============================================================
# 14. CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges",
    hue="Cluster",
    palette="Set1",
    s=70
)

plt.title(
    "Customer Clusters: Tenure vs Monthly Charges"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "clusters_tenure_monthly.png"
    )
)

plt.close()


# ============================================================
# 15. SAVE CLUSTERED DATA
# ============================================================

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "clustered_customers.csv"
    ),
    index=False
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "kmeans.pkl"
)

scaler_path = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

joblib.dump(
    kmeans,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

print("\nModel saved:")
print(model_path)

print("\nScaler saved:")
print(scaler_path)


# ============================================================
# 17. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nNumber of customers: {len(df)}")
print(f"Number of clusters: {best_k}")
print(f"Final silhouette score: {final_score:.4f}")