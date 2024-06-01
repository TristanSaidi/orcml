import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn import neighbors
import numpy as np

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

    # convert to networkx graph and symmetrize A
    n_points = X.shape[0]
    G = nx.Graph()
    for i in range(n_points):
        for j in range(i+1, n_points):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])

    assert G.is_directed() == False, "The graph is directed."
    return G, A


def adjust_orcs(orcs):
    """
    Adjust the Ollivier-Ricci curvatures to lie within 2 standard deviations of the mean.
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
    adjusted_orcs = np.clip(orcs, mean - 2*std, mean + 2*std)
    adjusted_orcs = (adjusted_orcs - adjusted_orcs.min()) / (adjusted_orcs.max() - adjusted_orcs.min())
    return adjusted_orcs

def graph_orc(G, weight='weight', alpha=0.5):
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
    orc = OllivierRicci(G, weight=weight, alpha=alpha, verbose='INFO')
    orc.compute_ricci_curvature()
    orcs = []
    for i, j, _ in orc.G.edges(data=True):
        orcs.append(orc.G[i][j]['ricciCurvature'])
    # adjust the Ollivier-Ricci curvatures
    adjusted_orcs = adjust_orcs(orcs)
    # reassign the adjusted Ollivier-Ricci curvatures to the graph
    for idx, (i, j, _) in enumerate(orc.G.edges(data=True)):
        orc.G[i][j]['ricciCurvature'] = adjusted_orcs[idx]
    return orc.G, adjusted_orcs


def prune(G, threshold, X, color, cluster=None):
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
    G_pruned = nx.Graph()
    preserved_nodes = set()
    
    # bookkeeping
    num_removed_edges = 0
    total_bad_edges = 0
    num_bad_edges_removed = 0 # edges with vertices in different clusters (if cluster is not None)

    for i, j, d in G.edges(data=True):
        total_bad_edges += 1 if cluster is not None and cluster[i] != cluster[j] else 0
        if d['ricciCurvature'] > threshold:
            G_pruned.add_edge(i, j, weight=d['weight'])
            preserved_nodes.add(i)
            preserved_nodes.add(j)
            G_pruned[i][j]['ricciCurvature'] = d['ricciCurvature']
        else:
            num_removed_edges += 1
            if cluster is not None and cluster[i] != cluster[j]:
                num_bad_edges_removed += 1
    
    preserved_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])

    X_pruned = X[list(preserved_nodes)]
    color_pruned = color[list(preserved_nodes)]
    
    print(f'{num_removed_edges} of {len(G.edges())} total edges were removed.')
    if cluster is not None:
        print(f'{num_bad_edges_removed} of {num_removed_edges} removed edges were bad edges.')
        print(f'{num_bad_edges_removed} of {total_bad_edges} total bad edges were removed.')
    
    return {
        'G_pruned': G_pruned,
        'preserved_nodes': preserved_nodes,
        'X_pruned': X_pruned,
        'color_pruned': color_pruned,
        'preserved_orcs': preserved_orcs
    }
