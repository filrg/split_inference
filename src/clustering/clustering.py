import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Clustering:
    def __init__(self , features , device_names , n_clusters = 3 ):
        self.features = features
        self.device_names = device_names
        self.n_clusters = n_clusters

    def k_means(self):
        """
        features     : shape (N, F)
        device_names : list of length N
        n_clusters   : number of power levels

        return:
        {
            "level 1": [device weakest → strongest],
            "level 2": [...],
            ...
        }
        """

        assert len(self.features) == len(self.device_names), "Mismatch input size"

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)

        # KMeans clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=42
        )
        labels = kmeans.fit_predict(X_scaled)

        # Compute power score
        power_score = X_scaled.sum(axis=1)

        # Group devices by cluster-
        cluster_map = {}
        for name, label, score in zip(self.device_names, labels, power_score):
            cluster_map.setdefault(label, []).append((name, score))

        # 5. Sort clusters by average power
        cluster_order = sorted(
            cluster_map.keys(),
            key=lambda c: np.mean([s for _, s in cluster_map[c]])
        )

        # Build result dict
        result = {}
        for i, c in enumerate(cluster_order, start=1):
            # sort devices inside cluster
            devices_sorted = sorted(
                cluster_map[c],
                key=lambda x: x[1]
            )
            result[f"level {i}"] = [d[0] for d in devices_sorted]


        return result

    def run(self):
        if self.n_clusters == 1:
            return self.k_means()
            # print("The number of cluster is 1")
        elif self.n_clusters > len(self.features):
            print("[Ops the number of cluster > length of features ")
        else :
            return self.k_means()
