import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

# Generate clustered data
n = 1000
centers = [(3, 3), (8, 8)]  # Cluster centers
X, _ = make_blobs(n_samples=n, centers=centers, random_state=42)

# Nearest Neighbors algorithm for anomaly detection
nn = NearestNeighbors(n_neighbors=2)
nn.fit(X)
distances, _ = nn.kneighbors(X)
distance_to_nearest_neighbor = distances[:, 1]  # distances to the closest neighbor

# Detect anomalies based on the nearest neighbors
nn_threshold = np.mean(distance_to_nearest_neighbor) + 2 * np.std(distance_to_nearest_neighbor)
nn_anomalies = np.where(distance_to_nearest_neighbor > nn_threshold)[0]

# Graph
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6, c='blue', label='Data points')
plt.scatter(X[nn_anomalies, 0], X[nn_anomalies, 1], s=50, alpha=0.6, c='red', label='Anomalies')
plt.title("Nearest Neighbor Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
