import numpy as np
import scipy


def intercluster_distortion(A_gt, A, cluster, node_indices=None):
    """ 
    Compute distortion of inter-cluster distances between unpruned and pruned graph.
    Parameters
    ----------
    A_gt : np.array
        Adjacency matrix of a graph.
    A_gt : np.array
        Adjacency matrix of base truth graph.   
    cluster : np.array
        (shuffled) Cluster labels.
    node_indices : list
        List of node indices for unshuffling (Optional). 
    """
    if node_indices is not None:
        cluster = cluster[node_indices]
    # num of clusters
    n_clusters = len(np.unique(cluster))
    cluster_indices = []
    for i in range(n_clusters):
        cluster_indices.append(np.where(cluster == i)[0])
 
    # compute estimated geodesic distances
    cluster_gt_geo_distances = intercluster_distances(A_gt, cluster_indices)
    cluster_geo_distances = intercluster_distances(A, cluster_indices)
    
    # compute distortion. distortion = |d_pruned - d| / d
    valid = np.where(cluster_gt_geo_distances != 0)[0]
    assert np.all(valid == np.where(cluster_geo_distances != 0)[0]), 'Indices of valid distances do not match.'
    cluster_gt_geo_distances = cluster_gt_geo_distances[valid]
    cluster_geo_distances = cluster_geo_distances[valid]
    distortion = np.abs(cluster_geo_distances - cluster_gt_geo_distances) / cluster_gt_geo_distances
    return distortion

def intercluster_distances(A, cluster_indices):
    """
    Estimate inter-cluster geodesic distances of graph.
    Parameters
    ----------
    A : np.array
        Adjacency matrix of graph.
    cluster_indices : list
        List of lists of node indices for each cluster.
    """
    n_clusters = len(cluster_indices)
    # compute estimated geodesic distances
    geo_distances = scipy.sparse.csgraph.shortest_path(A, directed=False)

    cluster_geo_distances = []
    for i in range(n_clusters):
        cluster_geo_distances_i = geo_distances[cluster_indices[i]][:, cluster_indices[i]].flatten()
        cluster_geo_distances.append(cluster_geo_distances_i)
    cluster_geo_distances = np.concatenate(cluster_geo_distances)
    return cluster_geo_distances
