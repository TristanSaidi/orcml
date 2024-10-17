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

from numbers import Integral, Real

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh


from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.manifold._locally_linear import null_space
from sklearn.utils.validation import FLOAT_DTYPES
import umap

# embeddings
def UMAP(A, n_neighbors, n_components, X=None):
    """
    Compute the UMAP embedding of a graph.
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
        The UMAP embedding of the graph.
    """

    n_connected_components, component_labels = scipy.sparse.csgraph.connected_components(A)
    
    if n_connected_components > 1:
        if X is None:
            raise ValueError("The graph is not connected. Please provide the original data.")
        warnings.warn(
            (
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " UMAP might be slow. Increase the number of neighbors to "
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

    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='precomputed')
    Y = umap_obj.fit_transform(distances)
    return Y


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
                " tSNE might be slow. Increase the number of neighbors to "
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

def spectral_embedding(A, n_components, affinity=True):
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
    if not affinity:
        W = A
    # convert A to affinity matrix. nonzero entries become exp(-A**2). zero entries become 0
    else:
        # scale distances so that max distance is 1
        A /= np.max(A)
        W = np.exp(-A**2)
        W[np.where(A == 0)] = 0
        # diagonal entries are set to 1
        np.fill_diagonal(W, 1)
    se = manifold.SpectralEmbedding(n_components=n_components, affinity='precomputed')
    Y = se.fit_transform(W)
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

    # isomap with precomputed distances
    iso = Isomap(metric='precomputed', n_components=n_components)
    # compute geodesic distances
    distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T), "The distance matrix is not symmetric."
    
    Y = iso.fit_transform(distances)
    return Y

def lle(A, embedding, n_neighbors, n_components, X=None):
    """
    Compute the LLE embedding of a graph.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    embedding : array-like, shape (n_samples, n_features)
        The original data.
    n_components : int
        The number of components to keep.
    X : array-like, shape (n_samples, n_features), default=None
        Embeddings of the original data. To be used only if the graph is not connected.
    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        The LLE embedding of the graph.
    """
    distances = scipy.sparse.csgraph.shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T), "The distance matrix is not symmetric."
    Y, _ = locally_linear_embedding(distances=distances, embedding=embedding, n_neighbors=n_neighbors, n_components=n_components)
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

def locally_linear_embedding(
    distances,
    embedding,
    *,
    n_components,
    n_neighbors,
    reg=1e-3,
    eigen_solver="auto",
    tol=1e-5,
    max_iter=200,
    random_state=None,
    n_jobs=None,
):
    nbrs = NearestNeighbors(metric='precomputed', n_neighbors=n_neighbors, n_jobs=n_jobs)
    nbrs.fit(distances)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError(
            "output dimension must be less than or equal to input dimension"
        )

    M_sparse = eigen_solver != "dense"

    W = barycenter_kneighbors_graph(
        nbrs, embedding=embedding, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs
    )

    # we'll compute M = (I-W)'(I-W)
    # depending on the solver, we'll do this differently
    if M_sparse:
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[:: M.shape[0] + 1] += 1  # W = W - I = W - I

    return null_space(
        M,
        n_components,
        k_skip=1,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )



def barycenter_kneighbors_graph(distances, embedding, n_neighbors, reg=1e-3, n_jobs=None):
    knn = NearestNeighbors(metric='precomputed', n_neighbors=n_neighbors+1, n_jobs=n_jobs).fit(distances)
    # knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)
    ind = ind[:, 1:]
    data = barycenter_weights(embedding, embedding, ind, reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))



def barycenter_weights(X, Y, indices, reg=1e-3):
    X = check_array(X, dtype=FLOAT_DTYPES)
    Y = check_array(Y, dtype=FLOAT_DTYPES)
    indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, assume_a="pos")
        B[i, :] = w / np.sum(w)
    return B