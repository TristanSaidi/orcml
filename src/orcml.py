import networkx as nx
from src.OllivierRicci import OllivierRicci
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

def graph_orc(G, weight='weight'):
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


def prune_ORC(G, delta, X, cluster=None):
    """
    Prune the graph based on a threshold. Adjust the node coordinates and colors accordingly.
    Parameters
    ----------
    G : networkx.Graph
        The graph to prune.
    delta : float
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

    threshold = -1 + 2*(2-2*delta)

    for i, j, d in G.edges(data=True):
        total_bad_edges += 1 if cluster is not None and cluster[i] != cluster[j] else 0
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
        'N_good_removed': num_removed_edges - num_bad_edges_removed if cluster is not None else None,
        'N_bad_removed': num_bad_edges_removed if cluster is not None else None,
        'N_good_total': len(G.edges()) - total_bad_edges if cluster is not None else None,
        'N_bad_total': total_bad_edges if cluster is not None else None,
    }

def prune_adaptive(G, X, eps, lda, delta=1.0, weight='unweighted_dist', cluster=None):
    """
    Prune the graph based on mathematically predicted threshold.
    Parameters
    ----------
    G : networkx.Graph
        The graph to prune.
    X : array-like, shape (n_samples, n_features)
        The dataset.
    eps : float
        The epsilon parameter for the proximity graph.
    lda : float
        The lambda parameter for the adaptive pruning.
    delta : float, optional
        The delta (confidence) parameter for the adaptive pruning.
    Returns
    -------
    G_pruned : networkx.Graph
        The pruned graph.
    """    
    # construct the candidate set C, and filtered graph G'
    C = []
    G_prime = nx.Graph()
    bad_edges = []

    threshold = -1 + 2*(2-2*delta)

    for i, j, d in G.edges(data=True):
        bad_edges.append((i,j)) if cluster is not None and cluster[i] != cluster[j] else None
        
        if cluster is not None and cluster[i] != cluster[j]:
            print()
            print(f"Bad edge: {i} - {j}")
            print(f"Ricci curvature: {d['ricciCurvature']}")
            print(f"Ricci curvature threshold: {threshold}")
            print(f"distance: {d['weight']}")
            print()
        if d['ricciCurvature'] < threshold:
            C.append((i,j))
        else:
            G_prime.add_edge(i, j, weight=d[weight])
            G_prime[i][j]['ricciCurvature'] = d['ricciCurvature']
            G_prime[i][j]['scaledricciCurvature'] = d['scaledricciCurvature']
            G_prime[i][j]['wassersteinDistance'] = d['wassersteinDistance']
            G_prime[i][j]['unweighted_dist'] = d['unweighted_dist']
            G_prime[i][j]['weight'] = d['weight']
    
    total_bad_edges = len(bad_edges)
    print(f"Number of candidate edges: {len(C)}")
    # bookkeeping
    num_removed_edges = 0
    num_bad_edges_removed = 0 # edges with vertices in different clusters (if cluster is not None)

    G_pruned = G_prime.copy()
    preserved_nodes = set()

    for num, (i, j) in enumerate(C):
        # check distance d_G'(x,y) for all x,y in C
        threshold = 2/(np.sqrt(5)-1)*(7*np.pi/5)*(1-lda)

        if eps is not None:
            threshold *= eps
        else:
            # find the edge distance for all edges incident to i or j
            dists = []
            for k in G.neighbors(i):
                dists.append(G[i][k]['weight'])
            for k in G.neighbors(j):
                dists.append(G[j][k]['weight'])
            effective_eps = np.max(dists)
            threshold *= effective_eps

        if i not in G_prime.nodes() or j not in G_prime.nodes():
            continue
        try:
            d_G_prime = nx.shortest_path_length(G_prime, source=i, target=j, weight="weight") # use euclidean distance
        except nx.NetworkXNoPath:
            d_G_prime = np.inf
        print()
        if (i,j) not in bad_edges and cluster is not None:
            print(f"{num}. Good edge: {i} - {j}")
        elif (i,j) in bad_edges and cluster is not None:
            print(f"{num}. Bad edge: {i} - {j}")
        else:
            print(f"{num}. Edge: {i} - {j}")
        print(f"d_G_prime: {d_G_prime}")
        print(f"Threshold: {threshold}")
        print()
        if d_G_prime > threshold:
            # G_pruned.remove_edge(i, j)
            num_removed_edges += 1
            if cluster is not None and cluster[i] != cluster[j]:
                num_bad_edges_removed += 1
        else:
            G_pruned.add_node(i)
            G_pruned.add_node(j)
            G_pruned.add_edge(i, j, weight=G[i][j][weight])
            G_pruned[i][j]['ricciCurvature'] = G[i][j]['ricciCurvature']
            G_pruned[i][j]['scaledricciCurvature'] = G[i][j]['scaledricciCurvature']
            G_pruned[i][j]['wassersteinDistance'] = G[i][j]['wassersteinDistance']
            G_pruned[i][j]['unweighted_dist'] = G[i][j]['unweighted_dist']
            G_pruned[i][j]['weight'] = G[i][j]['weight']

            preserved_nodes.add(i)
            preserved_nodes.add(j)

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
    print("Min ORC:", min(preserved_orcs))
    print(f'{num_removed_edges} of {len(G.edges())} total edges were removed.')
    if cluster is not None:
        print(f'{num_bad_edges_removed} of {num_removed_edges} removed edges were bad edges.')
        print(f'{num_bad_edges_removed} of {total_bad_edges} total bad edges were removed.')
        print(f'All bad edges in C? {all([edge in C for edge in bad_edges])}')
    A_pruned = nx.adjacency_matrix(G_pruned).toarray()
    return {
        'G_pruned': G_pruned,
        'G_prime': G_prime,
        'A_pruned': A_pruned,
        'preserved_orcs': preserved_orcs,
        'preserved_scaled_orcs': preserved_scaled_orcs,
        'N_good_removed': num_removed_edges - num_bad_edges_removed if cluster is not None else None,
        'N_bad_removed': num_bad_edges_removed if cluster is not None else None,
        'N_good_total': len(C) - len(bad_edges) if cluster is not None else None,
        'N_bad_total': len(bad_edges) if cluster is not None else None,
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