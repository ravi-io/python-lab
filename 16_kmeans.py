# """
# Program 16: K-Means Clustering
# Time Complexity: O(n * k * d * i) where n=samples, k=clusters, d=dims, i=iterations
# Space Complexity: O(n * d + k * d)
# Requires: pip install scikit-learn matplotlib
# """

# import random
# import math


# class KMeans:
#     """K-Means Clustering from scratch."""

#     def __init__(self, k=3, max_iters=100, seed=42):
#         self.k = k
#         self.max_iters = max_iters
#         self.centroids = []
#         self.labels = []
#         random.seed(seed)

#     def _distance(self, a, b):
#         """Euclidean distance."""
#         return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

#     def _closest_centroid(self, point):
#         """Find index of nearest centroid."""
#         distances = [self._distance(point, c) for c in self.centroids]
#         return distances.index(min(distances))

#     def fit(self, X):
#         """Run K-Means algorithm."""
#         n = len(X)
#         d = len(X[0])

#         # Initialize centroids using random samples
#         indices = random.sample(range(n), self.k)
#         self.centroids = [X[i][:] for i in indices]

#         for iteration in range(self.max_iters):
#             # Assignment step
#             self.labels = [self._closest_centroid(x) for x in X]

#             # Update step
#             new_centroids = [[0.0] * d for _ in range(self.k)]
#             counts = [0] * self.k

#             for i, x in enumerate(X):
#                 cluster = self.labels[i]
#                 counts[cluster] += 1
#                 for j in range(d):
#                     new_centroids[cluster][j] += x[j]

#             for c in range(self.k):
#                 if counts[c] > 0:
#                     for j in range(d):
#                         new_centroids[c][j] /= counts[c]

#             # Check convergence
#             if new_centroids == self.centroids:
#                 print(f"  Converged at iteration {iteration + 1}")
#                 break
#             self.centroids = new_centroids

#         return self.labels

#     def inertia(self, X):
#         """Calculate within-cluster sum of squares."""
#         total = 0
#         for i, x in enumerate(X):
#             total += self._distance(x, self.centroids[self.labels[i]]) ** 2
#         return total


# def generate_clusters(centers, n_per_cluster=30, spread=0.5):
#     """Generate synthetic clustered data."""
#     data = []
#     true_labels = []
#     for idx, (cx, cy) in enumerate(centers):
#         for _ in range(n_per_cluster):
#             x = cx + random.gauss(0, spread)
#             y = cy + random.gauss(0, spread)
#             data.append([x, y])
#             true_labels.append(idx)
#     return data, true_labels


# # Driver Code
# if __name__ == "__main__":
#     print("=== K-Means Clustering ===\n")

#     # Generate synthetic data with 3 clusters
#     centers = [(2, 2), (8, 3), (5, 8)]
#     X, true_labels = generate_clusters(centers, n_per_cluster=30, spread=1.0)

#     print(f"Dataset: {len(X)} points, {len(centers)} true clusters\n")

#     # Run K-Means
#     kmeans = KMeans(k=3, max_iters=100)
#     labels = kmeans.fit(X)

#     print(f"\nFinal Centroids:")
#     for i, c in enumerate(kmeans.centroids):
#         count = labels.count(i)
#         print(f"  Cluster {i}: ({c[0]:.2f}, {c[1]:.2f}) - {count} points")

#     print(f"\nInertia (WCSS): {kmeans.inertia(X):.2f}")

#     # Elbow method
#     print("\n--- Elbow Method ---")
#     print(f"{'k':<5} {'Inertia':<12}")
#     for k in range(1, 7):
#         km = KMeans(k=k, max_iters=100)
#         km.fit(X)
#         print(f"{k:<5} {km.inertia(X):<12.2f}")


import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)

kmeans.fit(X)

print("Cluster Labels:")
print(kmeans.labels_)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)