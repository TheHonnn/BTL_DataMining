# src/mining/clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_kmeans_clustering(df, n_clusters=4):

    # Chọn feature số
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster"] = clusters

    # Evaluation
    sil_score = silhouette_score(X_scaled, clusters)
    dbi_score = davies_bouldin_score(X_scaled, clusters)

    return df_clustered, X_scaled, sil_score, dbi_score