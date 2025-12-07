import math
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


class KMeansEdgeClustering:
    """
    Lightweight K-Means clustering optimized for edge-device use.

    Features:
    - Manual Min-Max normalization
    - Optional feature weighting
    - Farthest-point centroid initialization
    - Low-memory Euclidean distance computation
    - Mapping centroids to nearest raw data points
    """
    def __init__(
        self,
        num_clusters: int = 2,
        max_iterations: int = 500,
        feature_index: Dict[str, int] | None = None,
        weights: Dict[str, float] | None = None,
    ):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.device_names = None

        self.feature_index = feature_index or {}
        self.weights = weights or {}

        self.centroids: List[List[float]] = []
        self.initial_centroids: List[List[float]] = []
        self.clusters: List[List[List[float]]] = []

    # Normalization
    @staticmethod
    def min_max_normalize(data: List[List[float]],) -> Tuple[List[List[float]], List[float], List[float]]:
        """
        Perform Min-Max normalization without numpy heavy ops.
        """

        num_features = len(data[0])
        min_vals = [float("inf")] * num_features
        max_vals = [float("-inf")] * num_features

        # Find min/max
        for row in data:
            for i, val in enumerate(row):
                min_vals[i] = min(min_vals[i], val)
                max_vals[i] = max(max_vals[i], val)

        # Normalize
        normalized = []
        for row in data:
            nrow = []
            for i, val in enumerate(row):
                if min_vals[i] == max_vals[i]:
                    nrow.append(0.0)
                else:
                    nrow.append((val - min_vals[i]) / (max_vals[i] - min_vals[i]))
            normalized.append(nrow)

        return normalized, min_vals, max_vals

    @staticmethod
    def euclidean_distance(p1: List[float], p2: List[float]) -> float:
        """
        Lightweight Euclidean distance.
        """
        s = 0.0
        for x, y in zip(p1, p2):
            s += (x - y) ** 2
        return math.sqrt(s)

    def min_distance_to_centroids(
        self,
        point: List[float],
        centroids: List[List[float]],) -> float:
        return min(self.euclidean_distance(point, c) for c in centroids)

    # Centroid init
    def select_next_centroid(
        self,
        centroids: List[List[float]],
        points: List[List[float]],) -> List[float]:
        """
        Farthest-point heuristic for selecting centroids.
        """
        best = None
        max_dist = -1

        for p in points:
            dist = self.min_distance_to_centroids(p, centroids)
            duplicated = any(np.allclose(p, c) for c in centroids)

            if dist > max_dist and not duplicated:
                max_dist = dist
                best = p

        return best

    # raw point mapping

    def nearest_point(
        self,
        raw_points: List[List[float]],
        centroid: List[float],
    ) -> List[float]:
        """
        Map centroid to nearest raw data point.
        """
        best = None
        best_dist = float("inf")

        for p in raw_points:
            d = self.euclidean_distance(p, centroid)
            if d < best_dist:
                best_dist = d
                best = p

        return best

    # Core  K means

    def fit(self, points: List[List[float]]) -> Tuple[
        List[List[float]],
        List[List[List[float]]],
        List[List[float]]
    ]:
        """
        Execute k-means clustering.
        Returns:
            final_centers, clusters, initial_centers
        """

        centroids = random.sample(points, 1)

        while len(centroids) < self.num_clusters:
            centroids.append(self.select_next_centroid(centroids, points))

        self.initial_centroids = centroids.copy()

        for i in range(self.max_iterations):

            self.clusters = [[] for _ in range(self.num_clusters)]

            # Assignment
            for idx, p in enumerate(points):
                distances = [
                    self.euclidean_distance(p, c)
                    for c in centroids
                ]
                cid = distances.index(min(distances))

                self.clusters[cid].append((idx, p)) # Store both index + data point
            # Update
            new_centroids = []

            for cluster in self.clusters:
                if not cluster:
                    new_centroids.append(random.choice(points))
                else:
                    point_list = [p for _, p in cluster]    # only points
                    centroid = np.mean(point_list, axis=0).tolist()
                    new_centroids.append(centroid)


            # Convergence check
            if np.allclose(centroids, new_centroids, atol=1e-6):
                # print(f"KMeans converged at iteration {i}")
                break

            centroids = new_centroids

        self.centroids = centroids

        final_centers = [
            self.nearest_point(points, c)
            for c in centroids
        ]

        return final_centers, self.clusters, self.initial_centroids

    # Full pipeline run

    def run_from_csv(self, csv_path: str) -> Tuple[
        List[List[float]],
        List[List[List[float]]],
        List[List[float]]
    ]:

        data = pd.read_csv(csv_path)
        # print(data.columns)
        self.device_names = data['name']
        raw_points = list(data.values)
        # print(self.device_names)

        # remove ID / index column
        raw_points = np.delete(raw_points, 0, axis=1).tolist()

        # Normalize
        points, _, _ = self.min_max_normalize(raw_points)
        points = np.array(points)

        # Apply weights
        for feat, w in self.weights.items():
            idx = self.feature_index.get(feat)
            if idx is not None:
                points[:, idx] *= w

        points = points.tolist()

        return self.fit(points)

    def _mean_distance_to_cluster(
        self,
        point: List[float],
        cluster: List[Tuple[int, List[float]]],
    ) -> float:
        """
        Average distance from a point to all points in one cluster.
        """
        if not cluster:
            return 0.0

        dists = [
            self.euclidean_distance(point, p)
            for _, p in cluster
        ]

        return sum(dists) / len(dists)

    def _silhouette_for_point(
        self,
        cluster_id: int,
        point: List[float],
        clusters: List[List[Tuple[int, List[float]]]],
    ) -> float:
        """
        Compute silhouette value for one point.
        """

        # a(i): intra-cluster distance
        own_cluster = clusters[cluster_id]
        a = self._mean_distance_to_cluster(point, own_cluster)

        # b(i): min mean distance to other clusters
        b = float("inf")

        for cid, cluster in enumerate(clusters):
            if cid == cluster_id:
                continue

            d = self._mean_distance_to_cluster(point, cluster)
            b = min(b, d)

        if max(a, b) == 0:
            return 0.0

        return (b - a) / max(a, b)

    def estimate_silhouette(self) -> float:
        """
        Compute overall silhouette score for the last clustering result.
        """
        if not self.clusters:
            raise RuntimeError("Call fit() before silhouette estimation.")

        scores = []

        for cid, cluster in enumerate(self.clusters):
            for _, point in cluster:
                s = self._silhouette_for_point(
                    cluster_id=cid,
                    point=point,
                    clusters=self.clusters,
                )
                scores.append(s)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def get_device_names(self):
        return self.device_names

