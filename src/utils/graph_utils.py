import networkx as nx
import numpy as np
import numpy as np
from sklearn import neighbors
from src.ollivier_ricci import OllivierRicci
import multiprocessing as mp
import tqdm

def compute_eff_eps_graph(G, edge):
    """
    Compute the effective epsilon for a given edge.
    
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    edge : tuple
        The edge.
    
    Returns
    -------
    effective_eps : float
        The effective epsilon.
    """
    i, j = edge
    dists = []
    for k in G.neighbors(i):
        dists.append(G[i][k]['weight'])
    for k in G.neighbors(j):
        dists.append(G[j][k]['weight'])
    effective_eps = np.mean(dists)
    return effective_eps

def compute_eff_eps_adj(A, n_neighbors, edge):
    """
    Compute the effective epsilon for a given edge.
    
    Parameters
    ----------
    A : np.ndarray
        The adjacency matrix.
    edge : tuple
        The edge.
    
    Returns
    -------
    effective_eps : float
        The effective epsilon.
    """
    i, j = edge
    dists = A[i, :]
    dists = dists[dists != 0]
    k_nearest = np.argsort(dists)[1:n_neighbors+1]
    effective_eps_i = np.mean(dists[k_nearest])
    # find the k-nearest neighbors of j
    dists = A[:,j]
    dists = dists[dists != 0]
    k_nearest = np.argsort(dists)[1:n_neighbors+1]
    effective_eps_j = np.mean(dists[k_nearest])
    effective_eps = max(effective_eps_i, effective_eps_j)
    return effective_eps

def compute_orc(G):
    """
    Compute the Ollivier-Ricci curvature on edges of a graph.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    Returns
    -------
    G : networkx.Graph
        The graph with the Ollivier-Ricci curvatures as edge attributes.
    """
    orc = OllivierRicci(G, weight="effective_eps", alpha=0.0, method='OTD', verbose='INFO')
    orc.compute_ricci_curvature()
    orcs = []
    for i, j, _ in orc.G.edges(data=True):
        orcs.append(orc.G[i][j]['ricciCurvature'])
    return {
        'G': orc.G,
        'orcs': orcs,
    }

def get_edge_stats(G, cluster=None, data_supersample_dict=None):
    """ 
    Get the number of good and bad edges in the graph.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    cluster : array-like, shape (n_samples,), optional
        The cluster assignment.
    data_supersample_dict : dict, optional
        The dictionary containing the supersampled data and subsample indices.
    Returns
    -------
    num_good_edges : int
        The number of good edges.
    num_bad_edges : int
        The number of bad edges.
    """
    edge_labels = get_edge_labels(G, cluster, data_supersample_dict)
    num_good_edges = sum(edge_labels)
    num_bad_edges = len(edge_labels) - num_good_edges
    return num_good_edges, num_bad_edges

def get_edge_labels(G, cluster=None, data_supersample_dict=None, scale=None):
    """
    Get the edge labels (good/bad) from a cluster assignment or geodesic distance.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    cluster : array-like, shape (n_samples,), optional
        The cluster assignment.
    data_supersample_dict : dict, optional
        The dictionary containing the supersampled data and subsample indices.
    scale : float, optional
        The scale parameter for estimating labels from geodesic distance.
    Returns
    -------
    edge_labels : list
        The edge labels.
    """
    if cluster is None:
        return get_edge_labels_from_geodesic(G, data_supersample_dict, scale)
    else:
        return get_edge_labels_from_cluster(G, cluster)

def get_edge_labels_from_cluster(G, cluster):
    """
    Get the edge labels (good/bad) from a cluster assignment.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    cluster : array-like, shape (n_samples,)
        The cluster assignment.
    Returns
    -------
    edge_labels : list
        The edge labels.
    """
    edge_labels = []
    for i, j, _ in G.edges(data=True):
        if cluster[i] == cluster[j]:
            edge_labels.append(1)
        else:
            edge_labels.append(0)
    return edge_labels

def get_edge_labels_from_geodesic(G, data_supersample_dict, scale=10):
    """ 
    Get the edge labels (good/bad) from the estimated noiseless geodesic distance.

    Parameters
    ----------
    G : networkx.Graph
        The graph.
    data_supersample_dict : dict
        The dictionary containing the supersampled data and subsample indices.
    
    Returns
    -------
    edge_labels : list
        The edge labels.
    """

    data_supersample = data_supersample_dict['data_supersample']
    subsample_indices = data_supersample_dict['subsample_indices']

    # make a new graph with the supersampled data
    G_supersample, _ = get_nn_graph(data_supersample, exp_params={'mode':'nbrs', 'n_neighbors':20})

    edge_labels = []
    
    with tqdm.tqdm(total=len(G.edges()), desc='Computing edge labels') as pbar:
        for i, j, _ in G.edges(data=True):
            # find geodesic distance in the supersampled graph
            d_G_supersample = nx.shortest_path_length(G_supersample, source=subsample_indices[i], target=subsample_indices[j], weight='weight')
            
            # find max distance between i and i's neighbors, j and j's neighbors
            distances_i = []
            distances_j = []
            for k in G.neighbors(i):
                distances_i.append(G[i][k]['weight'])
            for k in G.neighbors(j):
                distances_j.append(G[j][k]['weight'])
            
            effective_eps = max(np.max(distances_i), np.max(distances_j))
            if d_G_supersample > scale*effective_eps:
                edge_labels.append(0)
            else:
                edge_labels.append(1)
            pbar.update(1)
    return edge_labels


def get_nn_graph(data, exp_params):
    """ 
    Build the nearest neighbor graph.
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The dataset.
    exp_params : dict
        The experimental parameters.
    Returns
    -------
    return_dict : dict
    """
    # check if 'mode' provided, if not, default to 'nbrs'
    if 'mode' not in exp_params:
        exp_params['mode'] = 'nbrs'
        
    if exp_params['mode'] == 'nbrs':
        G, A = _get_nn_graph(data, mode=exp_params['mode'], n_neighbors=exp_params['n_neighbors']) # unpruned k-nn graph
    else:
        G, A = _get_nn_graph(data, mode=exp_params['mode'], epsilon=exp_params['epsilon'])
    return {
        "G": G,
        "A": A,
    }


def _get_nn_graph(X, mode='nbrs', n_neighbors=None, epsilon=None):
    """
    Build the nearest neighbor graph.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The dataset.
    mode : str, optional
        The mode of the graph construction. Either 'nbrs' or 'eps'.
    n_neighbors : int, optional
        The number of neighbors to consider when mode='nbrs'.
    epsilon : float, optional
        The epsilon parameter when mode='eps'.
    Returns
    -------
    G : networkx.Graph
        The proximity graph.
    """
    
    if mode == 'nbrs':
        assert n_neighbors is not None, "n_neighbors must be specified when mode='nbrs'."
        A = neighbors.kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    elif mode == 'eps':
        assert epsilon is not None, "epsilon must be specified when mode='eps'."
        A = neighbors.radius_neighbors_graph(X, radius=epsilon, mode='distance')
    else:
        raise ValueError("Invalid mode. Choose 'nbrs' or 'eps'.")
    # symmetrize the adjacency matrix
    A = np.maximum(A.toarray(), A.toarray().T)
    assert np.allclose(A, A.T), "The adjacency matrix is not symmetric."
    # convert to networkx graph and symmetrize A
    n_points = X.shape[0]
    nodes = set()
    G = nx.Graph()
    for i in range(n_points):
        G.add_node(i)
        G.nodes[i]['pos'] = X[i] # store the position of the node
        G.nodes[i]['vel'] = np.zeros(X.shape[1]) # store the velocity of the node
        for j in range(i+1, n_points):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j]) # weight is the euclidean distance
                if mode == 'eps':
                    G[i][j]['effective_eps'] = epsilon
                else: # estimate effective epsilon as the average of the k-nearest neighbors
                    effective_eps = compute_eff_eps_adj(A, n_neighbors, (i, j))
                    G[i][j]['effective_eps'] = effective_eps
                nodes.add(i)
                nodes.add(j)

    assert G.is_directed() == False, "The graph is directed."
    assert len(G.nodes()) == n_points, "The graph has isolated nodes."
    return G, A
