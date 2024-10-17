import networkx as nx
import numpy as np
from src.utils.graph_utils import *

shortcut_str = "Shortcut Edge Detected: edge {num}\n d_G'(x,y)/effective_eps: {emp_ratio}\n Threshold/effective_eps: {theo_ratio}\n\n"

default_exp_params = {
    'mode': 'nbrs',
    'n_neighbors': 20,
    'epsilon': None,
    'lda': 0.01,
    'delta': 0.8
}

# create ORCManL class
class ORCManL(object):

    def __init__(
            self, 
            exp_params=default_exp_params, 
            verbose=False, 
            reattach=True
        ):
        """ 
        Initialize the ORCML class.
        Parameters
        ----------
        exp_params : dict
            The experimental parameters. Includes 'mode', 'n_neighbors', 'epsilon', 'lda', 'delta'.
        verbose : bool, optional
            Whether to print verbose output for ORCManL algorithm.
        reattach : bool, optional
            Whether to reattach isolated nodes.
        """
        self.exp_params = exp_params
        if 'epsilon' not in exp_params:
            self.exp_params['epsilon'] = None
        if 'n_neighbors' not in exp_params:
            self.exp_params['n_neighbors'] = None
        self.verbose = verbose
        self.reattach = reattach
        self._setup_structs()
        self._setup_thresholds()

    def _setup_structs(self):
        """
        Setup data structures for the ORCManL algorithm.
        """
        # data
        self.X = None
        # NN-graph
        self.G = None # original nn-graph
        self.A = None # adjacency matrix of original nn-graph
        # list of Ollivier-Ricci curvatures
        self.orcs = None
        # Pruned graph
        self.C = None # candidate set
        self.C_indices = None # indices of candidate edges
        self.G_prime = None # thresholded nn-graph
        self.G_pruned = None # pruned nn-graph
        self.A_pruned = None # adjacency matrix of pruned nn-graph
        self.non_shortcut_edges = None # indices of non-shortcut edges
        self.shortcut_edges = None # indices of shortcut edges
        # Annotated graph
        self.G_ann = None

    def _setup_thresholds(self):
        """
        Compute the thresholds for pruning specified by the ORCManL algorithm.
        """
        self.orc_threshold = -1 + 4*(1-self.exp_params['delta'])
        self.dist_threshold = ((1-self.exp_params['lda'])*np.pi**2)/(2*np.sqrt(24*self.exp_params['lda']))

    def build_nnG(self):
        """
        Build the nearest neighbor graph and compute ORC for each edge.
        """
        if self.X is None or (self.A is not None and self.G is not None):
            raise ValueError("Data must be provided to build the nearest neighbor graph.")
        return_dict = get_nn_graph(self.X, self.exp_params)
        G = return_dict['G']
        self.A = return_dict['A']
        # compute ORC
        return_dict = compute_orc(G)
        self.G = return_dict['G']
        self.orcs = return_dict['orcs']

    def fit(self, data, return_self=False):
        """
        Build nearest neighbor graph of data and apply the ORCManL algorithm.
        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            The dataset.
        Returns
        -------
        self : ORCManL
        """
        self.X = data
        self.build_nnG()
        self._fit()
        if return_self:
            return self
            
    def _fit(self):
        """ 
        Run the ORCManL algorithm.
        """
        self.C = []
        self._construct_C()
        if self.verbose:
            print(f"Number of candidate edges: {len(self.C)}, Number of edges in G': {len(self.G.edges())}")
        self._validate_and_prune()
        self._reattach_isolated_nodes()
        self.A_pruned = self._create_A(self.G_pruned)

    def _construct_C(self):
        """
        Construct the candidate set C.
        """
        self.G_prime = nx.Graph()
        self.C_indices = []
        for idx, (i, j, d) in enumerate(self.G.edges(data=True)):
            if d['ricciCurvature'] < self.orc_threshold:
                self.C.append((i,j))
                self.C_indices.append(idx)
            else:
                self.G_prime.add_edge(i, j, weight=d["weight"])
                self.G_prime[i][j]['ricciCurvature'] = d['ricciCurvature']
                self.G_prime[i][j]['effective_eps'] = d['effective_eps']

    def _validate_and_prune(self):
        """
        Validation step for the ORCManL algorithm.
        """
        self.G_ann = self.G.copy()
        self.G_pruned = self.G_prime.copy()
        self.non_shortcut_edges = list(range(len(self.G.edges()))) # start from all edges and remove as we go
        self.shortcut_edges = [] # start empty and add as we go

        # iterate over candidate set
        for num, (i, j) in enumerate(self.C):
            # get epsilon
            effective_eps = self.G[i][j]['effective_eps']
            # check distance d_G'(i,j)
            threshold = self.dist_threshold * effective_eps
            if i not in self.G_prime.nodes() or j not in self.G_prime.nodes():
                continue
            try:
                d_G_prime = nx.shortest_path_length(self.G_prime, source=i, target=j, weight="weight")
            except nx.NetworkXNoPath:
                d_G_prime = np.inf
            # adjust G_ann
            self.G_ann[i][j]['G_prime_dist'] = d_G_prime
            if d_G_prime > threshold:
                self._remove_edge(num, i, j)
                if self.verbose:
                    print(shortcut_str.format(num=num, emp_ratio=d_G_prime/effective_eps, theo_ratio=threshold/effective_eps))
            else:
                self._preserve_edge(i, j)
            
    def _remove_edge(self, num, i, j):
        """ 
        Remove an edge from the graph.
        """
        self.non_shortcut_edges.remove(self.C_indices[num])
        self.shortcut_edges.append(self.C_indices[num])
        self.G_ann[i][j]['shortcut'] = 1

    def _preserve_edge(self, i, j):
        """ 
        Preserve an edge in the graph.
        """
        self.G_pruned.add_node(i)
        self.G_pruned.add_node(j)
        self.G_pruned.add_edge(i, j, weight=self.G[i][j]["weight"])
        self.G_pruned[i][j]['ricciCurvature'] = self.G[i][j]['ricciCurvature']
        self.G_pruned[i][j]['effective_eps'] = self.G[i][j]['effective_eps']

    def _reattach_isolated_nodes(self):
        """
        Reattach isolated nodes.
        """
        if not self.reattach:
            return
        if len(self.G_pruned.nodes()) != len(self.G.nodes()):
            print("Warning: There are isolated nodes in the graph. This will be artificially fixed.")
            print(f"Number of isolated nodes: {len(self.G.nodes()) - len(self.G_pruned.nodes())}")
            missing_nodes = set(self.G.nodes()).difference(self.G_pruned.nodes())
            for node_idx in missing_nodes:
                # find nearest neighbor
                isolated_node = self.X[node_idx]
                dists = np.linalg.norm(self.X - isolated_node, axis=1)
                dists[node_idx] = np.inf
                nearest_neighbor = np.argmin(dists)
                self.G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])
                # assign this edge 0 curvature
                self.G_pruned[node_idx][nearest_neighbor]['ricciCurvature'] = 0

    def _create_A(self, G):
        """
        Create the adjacency matrix of the pruned graph.
        """
        A = nx.adjacency_matrix(G).toarray()
        # symmetrize the adjacency matrix
        A = np.maximum(A, A.T)
        assert np.allclose(A, A.T), "The adjacency matrix is not symmetric."
        return A

    def get_pruned_graph(self):
        """
        Get the pruned graph.
        Returns
        -------
        G_pruned : networkx.Graph
            The pruned graph.
        """
        return self.G_pruned
    
    def get_annotated_graph(self):
        """
        Get the annotated graph.
        Returns
        -------
        G_ann : networkx.Graph
            The annotated graph.
        """
        return self.G_ann