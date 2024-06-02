import numpy as np
import scipy



def intercluster_distortion(A_gt, A, cluster, node_indices=None, supersample=False, subsample_indices=None):
    """ 
    Compute distortion of inter-cluster distances between unpruned and pruned graph.
    Parameters
    ----------
    A_gt : np.array
        Adjacency matrix of a (more densly sampled) graph.
    A : np.array
        Adjacency matrix of a graph.   
    cluster : np.array
        (shuffled) Cluster labels.
    node_indices : list
        List of node indices for unshuffling (Optional). 
    supersample : bool
        If True, A_gt is a derived denser sampling of A.
    subsample_indices : list
        List of subsample indices that map A_gt to A.
    """
    if node_indices is not None:
        cluster = cluster[node_indices]
        if supersample:
            subsample_indices = subsample_indices[node_indices]
    # num of clusters
    n_clusters = len(np.unique(cluster))
    cluster_indices = []
    for i in range(n_clusters):
        cluster_indices.append(np.where(cluster == i)[0])
 
    # compute estimated geodesic distances
    cluster_gt_geo_distances = intercluster_distances(A_gt, cluster_indices, supersample, subsample_indices)
    cluster_geo_distances = intercluster_distances(A, cluster_indices)
    assert not np.inf in cluster_gt_geo_distances and not np.inf in cluster_geo_distances, 'Geodesic distances contain infinity.'
    
    # compute distortion. distortion = |d_pruned - d| / d
    valid = np.where(cluster_gt_geo_distances != 0)[0]
    assert np.all(valid == np.where(cluster_geo_distances != 0)[0]), 'Indices of valid distances do not match.'
    cluster_gt_geo_distances = cluster_gt_geo_distances[valid]
    cluster_geo_distances = cluster_geo_distances[valid]
    distortion = np.abs(cluster_geo_distances - cluster_gt_geo_distances) / cluster_gt_geo_distances
    return distortion

def intercluster_distances(A, cluster_indices, supersample=False, subsample_indices=None):
    """
    Estimate inter-cluster geodesic distances of graph.
    Parameters
    ----------
    A : np.array
        Adjacency matrix of graph.
    cluster_indices : list
        List of lists of node indices for each cluster.
    supersample : bool
        If True, A is a derived denser sampling of the original graph.
    subsample_indices : list
        List of subsample indices that map A to the original graph.
    """
    n_clusters = len(cluster_indices)
    # compute estimated geodesic distances
    geo_distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    if supersample:
        geo_distances = geo_distances[subsample_indices][:, subsample_indices]
    assert geo_distances.shape[0] == geo_distances.shape[1] == sum([cluster_indices_i.shape[0] for cluster_indices_i in cluster_indices]), 'Geodesic distance matrix has wrong shape.'
    cluster_geo_distances = []
    for i in range(n_clusters):
        cluster_geo_distances_i = geo_distances[cluster_indices[i]][:, cluster_indices[i]].flatten()
        cluster_geo_distances.append(cluster_geo_distances_i)
    cluster_geo_distances = np.concatenate(cluster_geo_distances)
    return cluster_geo_distances
