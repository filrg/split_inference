import yaml
from K_means import KMeansEdgeClustering

if __name__ == "__main__":
    with open('../../cfg/config.yaml') as file:
        cfg_cluster = yaml.safe_load(file)["clustering"]

    ws = cfg_cluster["weights"]
    FEATURE_INDEX = {
        "Total Ram": 0,
        "Storage": 1,
        "Internet": 2,
        "Core": 3,
    }

    WEIGHTS = {
        "Core": ws[0],
        "Total Ram": ws[1],
        "Internet": ws[2],
        "Storage": ws[3],
    }

    num_clusters = cfg_cluster["num_clusters"]
    if num_clusters == 0 :
        print("!!! We haven't developed this function yet :(((")
        print("So , K will be 2 >.<")
        num_clusters = 2
    kmeans = KMeansEdgeClustering(
        num_clusters= num_clusters,
        max_iterations=cfg_cluster["max_iterations"],
        feature_index=FEATURE_INDEX,
        weights=WEIGHTS,
    )

    best_cluster = None
    best_silhouette = 0
    num_rums = cfg_cluster["num_runs"]

    data_path = "../../data/data_device.csv"

    for _ in range(num_rums):
        final_centers, clusters, init_centers = kmeans.run_from_csv(data_path)
        sil_score = kmeans.estimate_silhouette()
        if sil_score > best_silhouette:
            best_cluster = clusters
            best_silhouette = sil_score

    device_names = kmeans.get_device_names()

    print(f"\nBest silhouette score: {best_silhouette:.4f}")
    print("\nCluster Sizes:")
    for i, c in enumerate(best_cluster):
        print(f"Cluster {i}: {len(c)} samples :{[device_names[idx] for idx , _ in c]} ")
