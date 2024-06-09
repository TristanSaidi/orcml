from scipy.sparse.csgraph import shortest_path
from gph.python import ripser_parallel

def rips_ph(A, maxdim=2):
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
    dgm = ripser_parallel(A, metric='precomputed', maxdim=maxdim, collapse_edges=True)['dgms']
    proc_dgms = []
    for homology, persistence in enumerate(dgm):
        for element in persistence:
            proc_dgms.append([homology, element])
    return proc_dgms
