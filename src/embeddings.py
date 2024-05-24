from sklearn import manifold
import scipy
import numpy as np

from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.utils.graph import _fix_connected_components

# embeddings

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
    
def isomap(A, n_components):
    """
    Compute the Isomap embedding of a graph.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    n_components : int
        The number of components to keep.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The Isomap embedding of the graph.
    """
    # isomap with precomputed distances
    iso = Isomap(metric='precomputed', n_components=n_components)
    # compute geodesic distances
    distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    assert scipy.sparse.csgraph.connected_components(A)[0] == 1, "The graph is not connected."
    assert np.allclose(distances, distances.T), "The distance matrix is not symmetric."
    
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