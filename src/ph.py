from scipy.sparse.csgraph import shortest_path
from gph.python import ripser_parallel
import gudhi
from gudhi.wasserstein import wasserstein_distance
import numpy as np

def rips_ph(A, maxdim=2, thresh=np.inf):
    """
    Compute the persistence diagram of a Rips filtration on a graph.
    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the graph.
    maxdim : int
        Maximum homology dimension to compute.
    Returns
    -------
    proc_dgms : list
        List of persistence diagrams, where each element is a list [homology, persistence].
    """
    A = shortest_path(A, directed=False)
    dgm = ripser_parallel(A, metric='precomputed', maxdim=maxdim, collapse_edges=True, thresh=thresh)['dgms']
    proc_dgms = []
    for homology, persistence in enumerate(dgm):
        for element in persistence:
            proc_dgms.append([homology, element])
    return proc_dgms

def ph_dist(dgm1, dgm2):
    """ 
    Compute the Wasserstein distance between two persistence diagrams.
    Parameters
    ----------
    dgm1 : list
        Persistence diagram 1.
    dgm2 : list
        Persistence diagram 2.
    Returns
    -------
    distances : list
        List of Wasserstein distances per homology dimension
    """
    dgms1_dimensions = np.unique([dgm1[i][0] for i in range(len(dgm1))])
    dgms2_dimensions = np.unique([dgm2[i][0] for i in range(len(dgm2))]) 
    dimensions = np.unique(np.concatenate([dgms1_dimensions, dgms2_dimensions]))
    distances = []

    for dim in dimensions:
        dgm1_dim = np.array([dgm1[i][1] for i in range(len(dgm1)) if dgm1[i][0] == dim])
        dgm2_dim = np.array([dgm2[i][1] for i in range(len(dgm2)) if dgm2[i][0] == dim])
        # find number of infinities in dgm1_dim and dgm2_dim
        n_inf1 = np.sum(np.isinf(dgm1_dim))
        n_inf2 = np.sum(np.isinf(dgm2_dim))
        # if the number of infinities is different, the distance is infinity
        if n_inf1 != n_inf2:
            distances.append(np.inf)
        elif dgm1_dim.size == 0 and dgm2_dim.size == 0:
            distances.append(0)
        elif dgm1_dim.size == 0 or dgm2_dim.size == 0:
            distances.append(np.inf)
        else:
            distances.append(wasserstein_distance(dgm1_dim, dgm2_dim))
    return distances
