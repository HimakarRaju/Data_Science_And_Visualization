'''
Start
  |
  v
Input: Dataset, Number of Clusters (k)
  |
  v
Step 1: Initialize Centroids (Randomly or by using the first k points)
  |
  v
Step 2: Assign Each Data Point to the Closest Centroid
  |
  v
Step 3: Recompute Centroids by Averaging Points in Each Cluster
  |
  v
Step 4: Have Centroids Changed?
    /    \
  Yes      No
  /         \
Continue   Converged (End)





'''


import math


def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def assign_clusters(points, centroids):
    clusters = [[] for _ in centroids]
    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = distances.index(min(distances))
        clusters[closest_centroid].append(point)
    return clusters

def calculate_centroids(clusters):
    centroids = []
    for cluster in clusters:
        centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        centroids.append(centroid)
    return centroids

def k_means(points, k, max_iterations=100):
    # Randomly initialize centroids (taking first k points)
    centroids = points[:k]

    for _ in range(max_iterations):
        # Step 1: Assign clusters
        clusters = assign_clusters(points, centroids)

        # Step 2: Recalculate centroids
        new_centroids = calculate_centroids(clusters)

        if new_centroids == centroids:
            break  # If centroids haven't changed, we've converged
        centroids = new_centroids

    return centroids, clusters

# Example usage
points = [[1, 2], [2, 3], [3, 4], [8, 8], [9, 9], [10, 10]]
k = 2
centroids, clusters = k_means(points, k)
print(f"Centroids: {centroids}")
print(f"Clusters: {clusters}")
