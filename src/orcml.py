import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn import neighbors
import numpy as np
import multiprocessing as mp

# method

def make_prox_graph(X, mode='nbrs', n_neighbors=None, epsilon=None):
    """
    Create a proximity graph from a dataset.
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
        for j in range(i+1, n_points):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])
                if mode == 'eps':
                    G[i][j]['unweighted_dist'] = epsilon
                nodes.add(i)
                nodes.add(j)

    assert G.is_directed() == False, "The graph is directed."
    if len(G.nodes()) != n_points:
        print("Warning: There are isolated nodes in the graph. This will be artificially fixed.")
        missing_nodes = set(range(n_points)).difference(nodes)
        for node_idx in missing_nodes:
            # find nearest neighbor
            isolated_node = X[node_idx]
            dists = np.linalg.norm(X - isolated_node, axis=1)
            dists[node_idx] = np.inf # exclude self
            nearest_neighbor = np.argmin(dists)
            G.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])
    return G, A


def adjust_orcs(orcs, clip=False, scale=True):
    """
    Rescale the Ollivier-Ricci curvatures.
    Parameters
    ----------
    orcs : array-like, shape (n_edges,)
        The Ollivier-Ricci curvatures.
    Returns
    -------
    adjusted_orcs : array-like, shape (n_edges,)
        The adjusted Ollivier-Ricci curvatures.
    """
    orcs = np.array(orcs)
    mean = np.mean(orcs)
    std = np.std(orcs)
    
    ref_min = mean - 2*std
    ref_max = mean + 2*std
    # clip the Ollivier-Ricci curvatures to lie within 2 standard deviations of the mean
    orcs = np.clip(orcs, ref_min, ref_max) if clip else orcs
    orcs = (orcs - mean) / std if scale else orcs # convert to z-scores
    return orcs

def graph_orc(G, weight='weight', scale=True):
    """
    Compute the Ollivier-Ricci curvature on edges of a graph.
    Parameters
    ----------
    G : networkx.Graph
        The graph.
    weight : str
        The edge attribute to use as the weight.
    alpha : float
        The alpha parameter for the Ollivier-Ricci curvature.
    Returns
    -------
    G : networkx.Graph
        The graph with the Ollivier-Ricci curvatures as edge attributes.
    """
    orc = OllivierRicci(G, weight=weight, alpha=0.0, method='OTD', verbose='INFO')
    orc.compute_ricci_curvature()
    orcs = []
    wasserstein_distances = []
    for i, j, _ in orc.G.edges(data=True):
        orcs.append(orc.G[i][j]['ricciCurvature'])
        # record the Wasserstein distance between the two vertices
        W = orc.G[i][j][weight]*(1 - orc.G[i][j]['ricciCurvature'])
        wasserstein_distances.append(W)
        orc.G[i][j]['wassersteinDistance'] = W
    # adjust the Ollivier-Ricci curvatures
    scaled_orcs = adjust_orcs(orcs)
    # reassign the adjusted Ollivier-Ricci curvatures to the graph
    for idx, (i, j, _) in enumerate(orc.G.edges(data=True)):
        orc.G[i][j]['scaledricciCurvature'] = scaled_orcs[idx]
    return {
        'G': orc.G,
        'orcs': orcs,
        'scaled_orcs': scaled_orcs,
        'wasserstein_distances': wasserstein_distances,
    }


def prune(G, threshold, X, key='scaledricciCurvature', cluster=None):
    """
    Prune the graph based on a threshold. Adjust the node coordinates and colors accordingly.
    Parameters
    ----------
    G : networkx.Graph
        The graph to prune.
    threshold : float
        The threshold for the scaled Ollivier-Ricci curvature.
    Returns
    -------
    G_pruned : networkx.Graph
        The pruned graph.
    """
    assert key in ['ricciCurvature', 'scaledricciCurvature'], "Invalid key."
    G_pruned = nx.Graph()
    preserved_nodes = set()
    
    # bookkeeping
    num_removed_edges = 0
    total_bad_edges = 0
    num_bad_edges_removed = 0 # edges with vertices in different clusters (if cluster is not None)

    for i, j, d in G.edges(data=True):
        total_bad_edges += 1 if cluster is not None and cluster[i] != cluster[j] else 0
        if d[key] > threshold:
            G_pruned.add_edge(i, j, weight=d['weight'])
            preserved_nodes.add(i)
            preserved_nodes.add(j)
            G_pruned[i][j]['ricciCurvature'] = d['ricciCurvature']
            G_pruned[i][j]['scaledricciCurvature'] = d['scaledricciCurvature']
            G_pruned[i][j]['wassersteinDistance'] = d['wassersteinDistance']
        else:
            num_removed_edges += 1
            if cluster is not None and cluster[i] != cluster[j]:
                num_bad_edges_removed += 1

    if len(preserved_nodes) != len(G.nodes()):
        print("Warning: There are isolated nodes in the graph. This will be artificially fixed.")
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            # find nearest neighbor
            isolated_node = X[node_idx]
            dists = np.linalg.norm(X - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])
            # assign this edge 0 curvature
            G_pruned[node_idx][nearest_neighbor]['ricciCurvature'] = 0
            G_pruned[node_idx][nearest_neighbor]['scaledricciCurvature'] = 0
    
    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."
    
    preserved_orcs = []
    preserved_scaled_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])
        preserved_scaled_orcs.append(d['scaledricciCurvature'])
    
    print(f'{num_removed_edges} of {len(G.edges())} total edges were removed.')
    if cluster is not None:
        print(f'{num_bad_edges_removed} of {num_removed_edges} removed edges were bad edges.')
        print(f'{num_bad_edges_removed} of {total_bad_edges} total bad edges were removed.')
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_orcs': preserved_orcs,
        'preserved_scaled_orcs': preserved_scaled_orcs,
    }

def prune_adaptive(G, X, l, cluster=None):
    """
    Prune the graph based on mathematically predicted threshold.
    Parameters
    ----------
    G : networkx.Graph
        The graph to prune.
    threshold : float
        The threshold for the scaled Ollivier-Ricci curvature.
    Returns
    -------
    G_pruned : networkx.Graph
        The pruned graph.
    """
    G_pruned = nx.Graph()
    preserved_nodes = set()
    
    # bookkeeping
    num_removed_edges = 0
    total_bad_edges = 0
    num_bad_edges_removed = 0 # edges with vertices in different clusters (if cluster is not None)

    for i, j, d in G.edges(data=True):
        total_bad_edges += 1 if cluster is not None and cluster[i] != cluster[j] else 0
        # threshold = 1/deg(i) + 1/deg(j)
        edge_length = d['weight']
        # threshold = (2/G.degree(i) + 1/G.degree(j)) - (l/edge_length) * (1 - (2/G.degree(i) + 1/G.degree(j)))
        max_inv_deg = max(1/G.degree(i), 1/G.degree(j))
        min_inv_deg = min(1/G.degree(i), 1/G.degree(j))
        threshold = -(max_inv_deg + 3 * min_inv_deg) - (l/edge_length) * (max_inv_deg)

        if d['ricciCurvature'] > threshold:
            G_pruned.add_edge(i, j, weight=d['weight'])
            preserved_nodes.add(i)
            preserved_nodes.add(j)
            G_pruned[i][j]['ricciCurvature'] = d['ricciCurvature']
            G_pruned[i][j]['scaledricciCurvature'] = d['scaledricciCurvature']
            G_pruned[i][j]['wassersteinDistance'] = d['wassersteinDistance']
        else:
            num_removed_edges += 1
            if cluster is not None and cluster[i] != cluster[j]:
                num_bad_edges_removed += 1

    if len(preserved_nodes) != len(G.nodes()):
        print("Warning: There are isolated nodes in the graph. This will be artificially fixed.")
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            # find nearest neighbor
            isolated_node = X[node_idx]
            dists = np.linalg.norm(X - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])
            # assign this edge 0 curvature
            G_pruned[node_idx][nearest_neighbor]['ricciCurvature'] = 0
            G_pruned[node_idx][nearest_neighbor]['scaledricciCurvature'] = 0
    
    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."
    
    preserved_orcs = []
    preserved_scaled_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])
        preserved_scaled_orcs.append(d['scaledricciCurvature'])
    
    print(f'{num_removed_edges} of {len(G.edges())} total edges were removed.')
    if cluster is not None:
        print(f'{num_bad_edges_removed} of {num_removed_edges} removed edges were bad edges.')
        print(f'{num_bad_edges_removed} of {total_bad_edges} total bad edges were removed.')
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_orcs': preserved_orcs,
        'preserved_scaled_orcs': preserved_scaled_orcs,
    }

def spurious_edge_orc(G_orc, cluster):
    """
    Get the Ollivier-Ricci curvature of spurious edges in the graph.
    Parameters:
    ----------
    G_orc: nx.Graph
        The graph with computed Ollivier-Ricci curvature as edge attributes.
    cluster: list
        A list mapping nodes to their cluster index.
    Returns:
    --------
    spurious_edge_orcs: list
        A list of the Ollivier-Ricci curvature of spurious edges.
    """
    orc = nx.get_edge_attributes(G_orc, 'ricciCurvature')
    scaled_orc = nx.get_edge_attributes(G_orc, 'scaledricciCurvature')

    spurious_edge_orcs = []
    spurious_edge_scaled_orcs = []
    spurious_edge_distances = []
    spurious_edge_wasserstein_distances = []
    for edge in G_orc.edges():
        if cluster[edge[0]] != cluster[edge[1]]:
            spurious_edge_orcs.append(orc[edge])
            spurious_edge_scaled_orcs.append(scaled_orc[edge])
            spurious_edge_distances.append(G_orc[edge[0]][edge[1]]['weight'])
            spurious_edge_wasserstein_distances.append(G_orc[edge[0]][edge[1]]['wassersteinDistance'])
    return {
        'spurious_edge_orcs': spurious_edge_orcs,
        'spurious_edge_scaled_orcs': spurious_edge_scaled_orcs,
        'spurious_edge_distances': spurious_edge_distances,
        'spurious_edge_wasserstein_distances': spurious_edge_wasserstein_distances,
    }