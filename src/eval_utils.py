import numpy as np
import scipy

from src.orcml import *

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


def noise_vs_orc_experiment(
    n_runs,
    noises,
    data_function,
    data_function_kwargs,
    key='scaledricciCurvature',
    n_neighbors=20,
):
    """
    Estimate max orc for spuriously connected edges for different noise levels.
    Parameters
    ----------
    n_runs : int
        The number of runs to average over.
    noises : list
        The noise levels to consider.
    data_function : function
        The data generation function.
    data_function_kwargs : dict
        The keyword arguments for the data generation function.
    n_neighbors : int
        The number of neighbors to consider when constructing the proximity graph.
    Returns
    -------
    mean_max_orcs : array-like, shape (n_noises,)
        The mean maximum Ollivier-Ricci curvature of spurious edges for each noise level.
    std_max_orcs : array-like, shape (n_noises,)
        The standard deviation of the maximum Ollivier-Ricci curvature of spurious edges for each noise level.
    valid_noises : list
        The noise levels for which spurious edges were formed.
    """
    assert key in ['scaledricciCurvature', 'ricciCurvature'], 'Key must be either scaledricciCurvature or ricciCurvature.'
    mapped_key = 'spurious_edge_scaled_orcs' if key == 'scaledricciCurvature' else 'spurious_edge_orcs'
    
    mean_max_orcs = []
    std_max_orcs = []
    valid_noises = []
    for noise in noises:
        print(f'Running with noise {noise}')
        max_orcs_fixed_noise = []
        valid = True
        for _ in range(n_runs):
            return_dict = data_function(noise=noise, **data_function_kwargs)
            data = return_dict['data']
            cluster = return_dict['cluster']
            G, _ = make_prox_graph(data, mode='nbrs', n_neighbors=n_neighbors)
            return_dict = graph_orc(G, weight='weight')
            G_orc = return_dict['G']
            return_dict = spurious_edge_orc(G_orc, cluster)
            spurious_orc = return_dict[mapped_key]
            if len(spurious_orc) == 0:
                print('No spurious edges found, skipping')
                valid = False
                break
            # get max spurious edge orc
            max_spurious_orc = np.max(spurious_orc)
            max_orcs_fixed_noise.append(max_spurious_orc)
        if not valid:
            continue
        valid_noises.append(noise)
        max_orcs_fixed_noise = np.array(max_orcs_fixed_noise)
        mean_max_orcs.append(np.mean(max_orcs_fixed_noise))
        std_max_orcs.append(np.std(max_orcs_fixed_noise))
    mean_max_orcs = np.array(mean_max_orcs)
    std_max_orcs = np.array(std_max_orcs)
    return mean_max_orcs, std_max_orcs, valid_noises


def compute_metrics(edge_labels, preserved_edges):
    """ 
    Compute metrics for edge preservation. 

    Returns
    -------
    percent_good_removed: float
        Percent of good edges removed.
    percent_bad_removed: float
        Percent of bad edges removed.
    """

    edge_labels = np.array(edge_labels)
    preserved_edges = np.array(preserved_edges)

    N_good_total = np.sum(edge_labels == 1)
    N_bad_total = np.sum(edge_labels == 0)
    N_good_preserved = np.sum(edge_labels[preserved_edges] == 1)
    N_bad_preserved = np.sum(edge_labels[preserved_edges] == 0)
    percent_good_removed = 1 - (N_good_preserved / N_good_total)
    percent_bad_removed = 1 - (N_bad_preserved / N_bad_total)
    return percent_good_removed, percent_bad_removed