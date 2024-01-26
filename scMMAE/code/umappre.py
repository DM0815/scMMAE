'''
---------------------
SCOITfucntions
author:https://github.com/deepomicslab/SCOIT/tree/main
MIT LICENSE
---------------------
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
from communities.algorithms import louvain_method
from sklearn.cluster import SpectralClustering
from igraph import Graph
import leidenalg

# construct KNN graph
def knn_adj_matrix(X, k=20):

    samples_size, features_size = X.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    adj_matrix = nbrs.kneighbors_graph(X).toarray()

    return adj_matrix

# construct SNN graph
def snn_adj_matrix(X, k=20):

    samples_size, features_size = X.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_adj_matrix = nbrs.kneighbors_graph(X).toarray()
    adj_matrix = np.dot(knn_adj_matrix, knn_adj_matrix.T) * knn_adj_matrix * knn_adj_matrix.T
    row, col = np.diag_indices_from(adj_matrix)
    adj_matrix[row, col] = 0
    
    return adj_matrix

# construct jaccard SNN graph
def jsnn_adj_matrix(X, k=20, prune=1/15):

    samples_size, features_size = X.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_adj_matrix = nbrs.kneighbors_graph(X).toarray()
    snn_matrix = np.dot(knn_adj_matrix, knn_adj_matrix.T)
    adj_matrix = snn_matrix / (2 * k - snn_matrix)
    row, col = np.diag_indices_from(adj_matrix)
    adj_matrix[row, col] = 0
    adj_matrix[adj_matrix<prune] = 0

    return adj_matrix


# Louvain algorithm
def RunLouvain(adj_matrix, k=None):
    communities, _ = louvain_method(adj_matrix, n=k)
    labels = np.zeros(adj_matrix.shape[0]).astype(int)
    l = -1
    for each in communities:
        l += 1
        for index in each:
            labels[index] = l
    
    return list(labels)

# Spectral clustering
def RunSpectral(adj_matrix, k=5):
    clustering = SpectralClustering(n_clusters=k, random_state=0).fit(adj_matrix) 
    
    return list(clustering.labels_)

# Leiden algorithm
def RunLeiden(adj_matrix):
    G = Graph.Weighted_Adjacency(adj_matrix)
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    labels = np.zeros(adj_matrix.shape[0]).astype(int)
    for i in range(len(part)):
        labels[part[i]] = i

    return list(labels)
    