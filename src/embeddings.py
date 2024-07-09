from sklearn import manifold
import scipy
import numpy as np
import warnings
import networkx as nx

from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.utils.graph import _fix_connected_components

# embeddings

def tsne(A, n_components, X=None):
    """
    Compute the t-SNE embedding of a graph.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    n_components : int
        The number of components to keep.
    X : array-like, shape (n_samples, n_features), default=None
        Embeddings of the original data. To be used only if the graph is not connected.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The t-SNE embedding of the graph.
    """

    n_connected_components, component_labels = scipy.sparse.csgraph.connected_components(A)
    
    if n_connected_components > 1:
        if X is None:
            raise ValueError("The graph is not connected. Please provide the original data.")
        warnings.warn(
            (
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " Isomap might be slow. Increase the number of neighbors to "
                "avoid this issue."
            ),
            stacklevel=2,
        )
        # use array validated by NearestNeighbors
        ambient_distances = scipy.spatial.distance.pdist(X, metric="euclidean")
        ambient_distances = scipy.spatial.distance.squareform(ambient_distances)

        A = _fix_connected_components(
            X=A,
            graph=ambient_distances,
            component_labels=component_labels,
            n_connected_components=n_connected_components,
            mode="distance",
            metric="precomputed",
        )

    distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T), "The distance matrix is not symmetric."

    tsne = manifold.TSNE(n_components=n_components, metric='precomputed', init='random')
    Y = tsne.fit_transform(distances)
    return Y

def spectral_embedding(A, n_components):
    """
    Compute the spectral embedding of a graph.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    n_components : int
        The number of components to keep.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The spectral embedding of the graph.
    """
    se = manifold.SpectralEmbedding(n_components=n_components, affinity='precomputed')
    Y = se.fit_transform(A)
    return Y

def isomap_connected_component(A, n_components, X=None):
    """
    Compute the Isomap embedding of a graph with (potentially) > 1 connected component.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    n_components : int
        The number of components to keep.
    X : array-like, shape (n_samples, n_features), default=None
        Unused.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The Isomap embedding of the graph.
    """
    n_connected_components, component_labels = scipy.sparse.csgraph.connected_components(A)
    Y = []
    # separately embed each connected component        
    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        print('Connected component:', i, 'Number of points:', len(idx_i))
        # skip really small connected components
        if len(idx_i) < A.shape[0] // (n_connected_components * 2):
            continue
        Ai = A[np.ix_(idx_i, idx_i)]
        assert np.allclose(Ai, Ai.T), "The adjacency matrix is not symmetric."
        Yi = isomap(Ai, n_components)
        Y.append(Yi)

    num_clusters = len(Y)
    print('Number of clusters:', num_clusters)
    if num_clusters > 1:
        for cluster in range(num_clusters):
            theta = 2 * np.pi * cluster / num_clusters
            centroid = 3 * np.array([np.cos(theta), np.sin(theta)])
            Y[cluster] += centroid

    Y = np.concatenate(Y)
    return Y
    
def isomap(A, n_components, X=None):
    """
    Compute the Isomap embedding of a graph.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    n_components : int
        The number of components to keep.
    X : array-like, shape (n_samples, n_features), default=None
        Embeddings of the original data. To be used only if the graph is not connected.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The Isomap embedding of the graph.
    """

    n_connected_components, _ = scipy.sparse.csgraph.connected_components(A)
    assert n_connected_components == 1, "The graph is not connected."
    # isomap with precomputed distances
    iso = Isomap(metric='precomputed', n_components=n_components)
    # compute geodesic distances
    distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T), "The from scipy.sparse.csgraph import connected_componentsdistance matrix is not symmetric."
    
    Y = iso.fit_transform(distances)
    return Y


class Isomap(manifold.Isomap):

    def __init__(
        self,
        *,
        n_neighbors=5,
        radius=None,
        n_components=2,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            n_components=n_components,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            path_method=path_method,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
            metric=metric,
            p=p,
            metric_params=metric_params,
        )
    
    def _fit_transform(self, X):
        if self.metric != "precomputed":
            raise ValueError("This Isomap implementation requires a precomputed distance matrix.")

        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,
            kernel="precomputed",
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        ).set_output(transform="default")

        self.dist_matrix_ = X # metric is precomputed

        G = self.dist_matrix_**2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)
        self._n_features_out = self.embedding_.shape[1]