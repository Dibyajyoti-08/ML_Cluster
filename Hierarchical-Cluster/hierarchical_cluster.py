# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Dendrogram method to find optimal number of cluster
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) # ward: minimizing the variants in the cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Train the hierarchical clustering model in the dataset
hc = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')
y_hc = hc.fit_predict(X)

print('\nThe Dependent Variable:')
print(y_hc)

# Visualize the cluster
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.xlabel('Annual Income (Rs.)')
plt.ylabel('Spending Scores (1-100)')
plt.legend()
plt.title('Clusters of Customers')
plt.show()
