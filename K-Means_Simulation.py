import numpy as np
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# clustered data
n = 1000
centers = [(3, 3), (8, 8)]  # Cluster centers
X, _ = make_blobs(n_samples = n, centers = centers, random_state = 42)

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

# graph
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6, c='blue')
plt.scatter(X[anomalies, 0], X[anomalies, 1], s=50, alpha = 0.6, c='red', label = 'Anomalies')
plt.scatter(centers[:, 0], centers[:, 1], c = "red", marker = "x", s = 100, label = "Cluster Centers")
plt.title("K-Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
