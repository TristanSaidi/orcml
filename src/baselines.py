import networkx as nx
from scipy.stats import gaussian_kde
import numpy as np

def prune_random(G, data, p):
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
    preserved_edges = []
    for idx, edge in enumerate(list(G_pruned.edges())):
        if np.random.rand() < p:
            G_pruned.remove_edge(*edge)
        else:
            preserved_nodes.add(edge[0])
            preserved_nodes.add(edge[1])
            preserved_edges.append(idx)
    
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
    
    preserved_orcs = []
    preserved_scaled_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])

    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes is not equal to the number of nodes in the graph."
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_edges': preserved_edges,
        'preserved_orcs': preserved_orcs,
    }


def prune_orc(G, delta, X, weight="unweighted_dist", verbose=False):
    """
    Prune the graph based on a ORC threshold. Adjust the node coordinates and colors accordingly.
    Parameters
    ----------
    G : networkx.Graph
        The graph to prune.
    delta : float
        The threshold for the Ollivier-Ricci curvature.
    Returns
    -------
    G_pruned : networkx.Graph
        The pruned graph.
    """
    G_pruned = nx.Graph()
    preserved_nodes = set()
    
    # bookkeeping
    num_removed_edges = 0

    threshold = -1 + 2*(2-2*delta) # threshold for the Ollivier-Ricci curvature
    preserved_edges = []
    for idx, (i, j, d) in enumerate(G.edges(data=True)):
        if d['ricciCurvature'] > threshold:
            G_pruned.add_edge(i, j, weight=d[weight])
            preserved_nodes.add(i)
            preserved_nodes.add(j)
            preserved_edges.append(idx)
            G_pruned[i][j]['ricciCurvature'] = d['ricciCurvature']
            G_pruned[i][j]['wassersteinDistance'] = d['wassersteinDistance']
        else:
            num_removed_edges += 1

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
    
    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."
    
    preserved_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])
    if verbose:
        print(f'{num_removed_edges} of {len(G.edges())} total edges were removed.')
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_edges': preserved_edges,
        'preserved_orcs': preserved_orcs,
    }


def prune_bisection(G, data, n):
    """ 
    Prune the graph with the bisection method proposed by
    (Tian Xia, Jintao Li, Yongdong Zhang, and Sheng Tang. A more topologically stable locally linear embedding algorithm based on r*-tree.)

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    data : array-like, shape (n_samples, n_features)
        The input data.
    n : int
        Algorithm parameter.

    Returns
    -------
    networkx.Graph
        The pruned graph.
    """

    G_pruned = G.copy()
    preserved_nodes = set()
    preserved_edges = []
    # iterate over all edges
    for idx, edge in enumerate(list(G.edges())):
        x_i = data[edge[0]]
        x_j = data[edge[1]]
        # find midpoint
        x_ij = (x_i + x_j) / 2

        # compute e_ij by taking min average distance of n nearest neighbors of the endpoints
        dists = np.linalg.norm(data - x_i, axis=1)

        nearest_neighbors = np.argsort(dists)[:n]
        e_i = np.mean(np.linalg.norm(data[nearest_neighbors] - x_i, axis=1))

        dists = np.linalg.norm(data - x_j, axis=1)
        nearest_neighbors = np.argsort(dists)[:n]
        e_j = np.mean(np.linalg.norm(data[nearest_neighbors] - x_j, axis=1))

        e_ij = min(e_i, e_j)
        # find number of points in bounding box of x_ij
        bounding_box = np.zeros((data.shape[1], 2))
        for i in range(data.shape[1]):
            bounding_box[i, 0] = x_ij[i] - e_ij
            bounding_box[i, 1] = x_ij[i] + e_ij
        c_ij = np.sum(np.all(data >= bounding_box[:, 0], axis=1) & np.all(data <= bounding_box[:, 1], axis=1))
        # prune edge if c_ij == 0
        if c_ij == 0:
            G_pruned.remove_edge(*edge)
        else:
            preserved_nodes.add(edge[0])
            preserved_nodes.add(edge[1])
            preserved_edges.append(idx)
    
    if len(preserved_nodes) != len(G.nodes()):
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
            
    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."

    preserved_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])

    A_pruned = nx.adjacency_matrix(G).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_edges': preserved_edges,
        'preserved_orcs': preserved_orcs,
    }


def prune_mst(G, data, thresh):
    """
    Prune the graph with MST based method. Proposed by 
    (Richard Zemel, Miguel Carreira-Perpiñán. Proximity graphs for clustering and manifold learning)
    and 
    (Shao Chao, Huanh Hou-kuan, Zhou Lian-wei. P-ISOMAP: A New ISOMAP Based Data Visualization Algorithm with Less Sensitivity to the Neighborhood Size)
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    data : array-like, shape (n_samples, n_features)
        The input data.
    thresh : float
        The distance threshold for pruning edges.
    """
    G_pruned = G.copy()

    # first order MST
    mst_1 = nx.minimum_spanning_tree(G)
    # assert len(G.nodes()) == len(mst_1.nodes()), "The number of nodes in the MST does not match the number of nodes in the graph."
    # assert len(G.nodes)-1 == len(mst_1.edges()), "The number of edges in the MST does not match the number of edges in the graph."

    # second order MST
    G_minus_mst_1 = nx.Graph(G)
    G_minus_mst_1.remove_edges_from(mst_1.edges())
    # assert len(G_minus_mst_1.nodes()) == len(G.nodes()), "The number of nodes in the complement graph does not match the number of nodes in the graph."
    # assert len(G_minus_mst_1.edges()) == len(G.edges()) - len(mst_1.edges()), "The number of edges in the complement graph does not match the number of edges in the graph."

    mst_2 = nx.minimum_spanning_tree(G_minus_mst_1)

    # combine the two MSTs
    mst_combined = nx.compose(mst_1, mst_2)
    # assert len(mst_combined.nodes()) == len(G.nodes()), "The number of nodes in the combined MST does not match the number of nodes in the graph."
    # assert len(mst_combined.edges()) == len(mst_1.edges()) + len(mst_2.edges()), "The number of edges in the combined MST does not match the number of edges in the graph."
    # prune edges
    preserved_nodes = set()
    preserved_edges = []
    for idx, edge in enumerate(list(G_pruned.edges())):
        v1 = edge[0]
        v2 = edge[1]
        # shortest path length between v1 and v2 in the combined MST
        shortest_path_length = nx.shortest_path_length(mst_combined, source=v1, target=v2, weight='weight')
        if shortest_path_length > thresh:
            G_pruned.remove_edge(v1, v2)
        else:
            preserved_nodes.add(v1)
            preserved_nodes.add(v2)
            preserved_edges.append(idx)
    
    if len(preserved_nodes) != len(G.nodes()):
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
    
    preserved_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])

    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_edges': preserved_edges,
        'preserved_orcs': preserved_orcs,
    }
    

def prune_density(G, data, thresh):
    """
    Prune the graph based on density estimation.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    data : array-like, shape (n_samples, n_features)
        The input data.
    thresh : float
        The density threshold for pruning edges.

    Returns
    -------
    networkx.Graph
        The pruned graph.
    """

    G_pruned = G.copy()
    # compute density
    kde = gaussian_kde(data.T)
    # prune edges
    preserved_nodes = set()
    preserved_edges = []
    for idx, edge in enumerate(list(G_pruned.edges())):
        v1 = edge[0]
        v2 = edge[1]
        # compute density at midpoint
        x_i = data[v1]
        x_j = data[v2]
        x_ij = (x_i + x_j) / 2
        density_ij = kde.pdf(x_ij)
        if density_ij < thresh:
            G_pruned.remove_edge(v1, v2)
        else:
            preserved_nodes.add(v1)
            preserved_nodes.add(v2)
            preserved_edges.append(idx)
    
    if len(preserved_nodes) != len(G.nodes()):
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
    
    preserved_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])

    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_edges': preserved_edges,
        'preserved_orcs': preserved_orcs,
    }
 
def prune_distance(G, data, thresh):
    """
    Prune the graph based on distance threshold.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    data : array-like, shape (n_samples, n_features)
        The input data.
    thresh : float
        The distance threshold for pruning edges.
    
    Returns
    -------
    networkx.Graph
        The pruned graph.
    """
    G_pruned = G.copy()
    preserved_nodes = set()
    preserved_edges = []
    for idx, edge in enumerate(list(G_pruned.edges())):
        v1 = edge[0]
        v2 = edge[1]
        # compute distance between v1 and v2
        dist = np.linalg.norm(data[v1] - data[v2])
        if dist > thresh:
            G_pruned.remove_edge(v1, v2)
        else:
            preserved_nodes.add(v1)
            preserved_nodes.add(v2)
            preserved_edges.append(idx)
    
    if len(preserved_nodes) != len(G.nodes()):
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
    
    preserved_orcs = []
    for i, j, d in G_pruned.edges(data=True):
        preserved_orcs.append(d['ricciCurvature'])

    assert len(G.nodes()) == len(G_pruned.nodes()), "The number of preserved nodes does not match the number of nodes in the pruned graph."
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'A_pruned': A_pruned,
        'preserved_edges': preserved_edges,
        'preserved_orcs': preserved_orcs,
    }