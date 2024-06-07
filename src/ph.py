import ripser
import numpy as np

def rips_ph(A, maxdim=1):
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
    # Prepare adjacency matrix for Ripser
    A = A.astype(float)
    A[A == 0] = np.inf
    np.fill_diagonal(A, [0]*A.shape[0])
    dgms = ripser.ripser(A, distance_matrix=True, maxdim=maxdim)['dgms']
    # Process persistence diagrams
    proc_dgms = []
    for homology, persistence in enumerate(dgms):
        for element in persistence:
            proc_dgms.append([homology, element])
    return proc_dgms
