import numpy as np
import scipy

from src.orcml import *

def compute_metrics(edge_labels, preserved_edges, percent=True):
    """ 
    Compute metrics for edge preservation. 

    Returns
    -------
    percent_good_removed: float
        Percent of good edges removed.
    percent_bad_removed: float
        Percent of bad edges removed.
    """

    edge_labels = np.array(edge_labels)
    preserved_edges = np.array(preserved_edges)

    N_good_total = np.sum(edge_labels == 1)
    N_bad_total = np.sum(edge_labels == 0)
    N_good_preserved = np.sum(edge_labels[preserved_edges] == 1)
    N_bad_preserved = np.sum(edge_labels[preserved_edges] == 0)
    percent_good_removed = 1 - (N_good_preserved / N_good_total)
    percent_bad_removed = 1 - (N_bad_preserved / N_bad_total)
    if percent:
        return percent_good_removed, percent_bad_removed
    else:
        return N_good_total - N_good_preserved, N_bad_total - N_bad_preserved