import networkx as nx
from src.OllivierRicci import OllivierRicci
from sklearn import neighbors
import numpy as np
import multiprocessing as mp


def random_prune(G, data, p):
    """
    Randomly prune edges from a graph.
    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    data : array-like, shape (n_samples, n_features)
        The input data.
    p : float
        The probability of retaining an edge.
    Returns
    -------
    networkx.Graph
        The pruned graph.
    """
    G_pruned = G.copy()
    preserved_nodes = set()
    for edge in list(G_pruned.edges()):
        if np.random.rand() > p:
            G_pruned.remove_edge(*edge)
        else:
            preserved_nodes.add(edge[0])
            preserved_nodes.add(edge[1])
    
    if len(preserved_nodes) < G_pruned.number_of_nodes():
        print("Warning: There are isolated nodes in the graph. This will be artificially fixed.")
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            # find nearest neighbor
            isolated_node = data[node_idx]
            dists = np.linalg.norm(data - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])
            # assign this edge 0 curvature
            G_pruned[node_idx][nearest_neighbor]['ricciCurvature'] = 0
            G_pruned[node_idx][nearest_neighbor]['scaledricciCurvature'] = 0
    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes is not equal to the number of nodes in the graph."
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned
    }
