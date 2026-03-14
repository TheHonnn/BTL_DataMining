from src.data.loader import load_raw_data
from src.features.builder import build_features
from src.mining.clustering import (
    prepare_clustering_data,
    run_kmeans,
    evaluate_clustering,
    attach_clusters,
    name_clusters
)

df = load_raw_data()

df = build_features(df)

df_scaled = prepare_clustering_data(df)

labels, model = run_kmeans(df_scaled, k=3)

sil, dbi = evaluate_clustering(df_scaled, labels)

print("\nClustering Evaluation")
print("Silhouette Score:", sil)
print("Davies Bouldin Index:", dbi)

df_clustered = attach_clusters(df.iloc[:len(labels)], labels)

df_clustered, names = name_clusters(df_clustered)

print("\nCluster names:")
print(names)

print("\nCluster distribution:")
print(df_clustered["cluster_name"].value_counts())