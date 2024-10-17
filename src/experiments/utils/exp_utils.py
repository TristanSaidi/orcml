from src.orcmanl import ORCManL

def prune_helper(X, exp_params, verbose=False, reattach=True):
    """ 
    Build the nearest neighbor graph and prune it with the orcml method.
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The dataset.
    exp_params : dict
        The experimental parameters.
    verbose : bool, optional
        Whether to print verbose output for orcmanl algorithm.
    reattach : bool, optional
        Whether to reattach isolated nodes.
    Returns
    -------
    return_dict : dict with keys 'G_original', 'A_original', 'G_orcmanl', 'A_orcmanl'
    """
    orcmanl = ORCManL(exp_params, verbose=verbose, reattach=reattach)
    orcmanl.fit(X)
    return {
        'G_original': orcmanl.G,
        'A_original': orcmanl.A,
        'G_orcmanl': orcmanl.G_pruned,
        'A_orcmanl': orcmanl.A_pruned,
        'non_shortcut_edges': orcmanl.non_shortcut_edges,
        'shortcut_edges': orcmanl.shortcut_edges,
    }

