import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# Generate clustered data
n = 1000
centers = [(3, 3), (8, 8)]  # Cluster centers
X, _ = make_blobs(n_samples=n, centers=centers, random_state=42)

# KNN algorithm
# Nearest Neighbors algorithm for anomaly detection
nn = NearestNeighbors(n_neighbors=2)
nn.fit(X)
distances, _ = nn.kneighbors(X)
distance_to_nearest_neighbor = distances[:, 1]  # distances to the closest neighbor

# Detect anomalies based on the nearest neighbors
nn_threshold = np.mean(distance_to_nearest_neighbor) + 2 * np.std(distance_to_nearest_neighbor)
nn_anomalies = np.where(distance_to_nearest_neighbor > nn_threshold)[0]




# k-means algorithm
kmeans = KMeans(n_clusters = 2, random_state = 42)
kmeans.fit(X)

# set the cluster centers
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# get the distance between center and data point
distances = np.linalg.norm(X - centers[labels], axis = 1)

# detect anomalies
threshold = np.mean(distances) + 2 * np.std(distances)
anomalies = np.where(distances > threshold)[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


# KNN as ax1
ax1.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6, c='blue', label='Data points')
ax1.scatter(X[nn_anomalies, 0], X[nn_anomalies, 1], s=50, alpha=0.6, c='red', label='Anomalies')
ax1.set_title("Nearest Neighbor Anomaly Detection")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.legend()

# K-Means as ax2
ax2.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6, c='blue')
ax2.scatter(X[anomalies, 0], X[anomalies, 1], s=50, alpha = 0.6, c='red', label = 'Anomalies')
ax2.scatter(centers[:, 0], centers[:, 1], c = "red", marker = "x", s = 100, label = "Cluster Centers")
ax2.set_title("K-Means")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.legend()

#display
plt.tight_layout()
plt.show()
