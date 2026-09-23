import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import joblib
import os

## loading the data
df = pd.read_csv('week4/d3_model_serialization/data/Mall_Customers.csv')

### data validation
print('\nshape:')
print(df.shape)

print('\nfirst 5 columns:')
print(df.head(5))

print('\ndata info:')
print(df.info())


print('\nmissing values:')
print(df.isnull().sum())

print('\nduplicate values:')
print(df.duplicated().sum())

print('\ndataset description:')
print(df.describe())


### one hot encoding for gender column
encoding = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

encoded = encoding.fit_transform(df[['Gender']])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoding.get_feature_names_out()
)

print('\nonehot encoding for gender column:')
print(encoded_df.shape)
print(encoded_df.head(5))

### removing the gender column from the origin dataset
df = df.drop(columns=['Gender'])

### scaling the data using standard scaler

scaler = StandardScaler()

scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled,
    columns=df.columns
)

print(scaled_df.shape)
print(scaled_df.head(5))


### merging the data frames
df_merged = pd.concat([scaled_df, encoded_df], axis=1)

print('\nfinal dataset for the model:')
print(df_merged.shape)
print(df_merged.head(5))


### kmeans clustering algorithm

### finding the number of clusters

inhertia = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(df_merged)
    inhertia.append(model.inertia_)

plt.plot(range(2,11), inhertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()


### calculating the scorres for the clusters

silhouette_scores = []

for k in range(2,11):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(df_merged)

    score = silhouette_score(df_merged, labels=labels)

    silhouette_scores.append(score)

    print(f"K={k}, Silhouette Score={score:.4f}")


plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.show()


### training the final model with the optimal number of clusters

model = KMeans(
    n_clusters=5,
    n_init='auto',
    random_state=42
)

clusters = model.fit_predict(df_merged)

### add the cluster labels
df['clusters'] = clusters

print('\nfinal dataset with cluster labels:')
print(df.shape)
print(df.head(5))

### checking the cluster sizes
print('\ncluster sizes:')
print(df['clusters'].value_counts().sort_index())

## profile each clusters
cluster_profile = df.groupby("clusters")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print(cluster_profile)



## visualizing the clusters using a scatter plot
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["clusters"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()


### Serialize the K-Means model


model_path = r"week4/d3_model_serialization/models"

os.makedirs(model_path, exist_ok=True)

# Save K-Means model
joblib.dump(model, os.path.join(model_path, "customer_segmentation_model.pkl"))

# Save the scaler
joblib.dump(scaler, os.path.join(model_path, "customer_scaler.pkl"))

print("Model and scaler saved successfully.")


