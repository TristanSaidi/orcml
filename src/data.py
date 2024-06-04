from sklearn import datasets
from src.manifold import Torus, Hyperboloid
import numpy as np
# Data generation functions

def concentric_circles(n_points, factor, noise, supersample=False, supersample_factor=2.5):
    """ 
    Generate concentric circles with noise. 
    Parameters
    
    n_points : int
        The number of samples to generate.
    factor : float
        The scaling factor between the circles.
    noise : float
        The standard deviation of the Gaussian noise.
    supersample : bool
        If True, the circles are supersampled.
    supersample_factor : float
        The factor by which to supersample the circles.
    Returns
    -------
    Dictionary providing the following keys:
        data : array-like, shape (n_points, 2)
            The generated samples.
        cluster : array-like, shape (n_points,)
            The integer labels for class membership of each sample.
        data_supersample : array-like, shape (n_points*supersample_factor, 2)
            The supersampled circles.
        subsample_indices : list
            The indices of the subsampled circles.
    """
    if supersample:
        N_total = int(n_points * supersample_factor)
        subsample_indices = np.random.choice(N_total, n_points, replace=False)
    else:
        N_total = n_points
        subsample_indices = None
    circles, cluster = datasets.make_circles(n_samples=N_total, factor=factor)
    if supersample:
        circles_supersample = circles.copy()
        circles = circles[subsample_indices]
        cluster = cluster[subsample_indices]
    else:
        circles_supersample = None
    circles += noise * np.random.randn(*circles.shape)
    return_dict = {
        'data': circles,
        'cluster': cluster,
        'data_supersample': circles_supersample,
        'subsample_indices': subsample_indices
    }
    return return_dict

def swiss_roll(n_points, noise, dim=3, supersample=False, supersample_factor=2.5):
    """
    Generate a Swiss roll dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    Dictionary providing the following keys:
        swiss_roll : array-like, shape (n_points, dim)
            The generated Swiss roll.
        color : array-like, shape (n_points,)
            The color of each point.
        dim: int
            The dimension of the Swiss roll.
    """
    if supersample:
        N_total = int(n_points * supersample_factor)
        subsample_indices = np.random.choice(N_total, n_points, replace=False)
    else:
        N_total = n_points
        subsample_indices = None
    swiss_roll, color = datasets.make_swiss_roll(N_total)
    if dim == 2:
        swiss_roll = swiss_roll[:, [0, 2]]
    if supersample:
        swiss_roll_supersample = swiss_roll.copy()
        swiss_roll = swiss_roll[subsample_indices]
        color = color[subsample_indices]
    else:
        swiss_roll_supersample = None
        subsample_indices = None
    swiss_roll += noise * np.random.randn(*swiss_roll.shape)
    return_dict = {
        'data': swiss_roll,
        'cluster': color,
        'data_supersample': swiss_roll_supersample,
        'subsample_indices': subsample_indices
    }
    return return_dict

def torus(n_points, noise, r=1.5, R=5, double=False, supersample=False, supersample_factor=2.5):
    """
    Generate a 2-torus dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    torus : array-like, shape (n_points, 3)
        The generated torus.
    color : array-like, shape (n_points,)
        The color of each point.
    cluster : array-like, shape (n_points,)
        The cluster labels.
    torus_subsample : array-like, shape (n_points, 3)
        The subsampled torus.
    subsample_indices : list
        The indices of the subsampled torus.
    """
    if double and R <= 2*r:
        raise Warning("Double torii will intersect")
    torus, thetas, cluster, torus_subsample, subsample_indices = Torus.sample(N=n_points, r=r, R=R, double=double, supersample=supersample, supersample_factor=supersample_factor)
    color = Torus.exact_curvatures(thetas, r, R)
    color = np.array(color)
    torus += noise * np.random.randn(*torus.shape)
    return_dict = {
        'data': torus,
        'cluster': cluster,
        'color': color,
        'data_supersample': torus_subsample,
        'subsample_indices': subsample_indices
    }
    return return_dict

def hyperboloid(n_points, noise, double=False, supersample=False, supersample_factor=2.5):
    """ 
    Generate a hyperboloid dataset.
    Parameters
    ----------
    n_points : int
        The number of points to generate.
    noise : float
        The standard deviation of the Gaussian noise.
    Returns
    -------
    hyperboloid : array-like, shape (n_points, 3)
        The generated hyperboloid.
    color : array-like, shape (n_points,)
        The color of each point.
    """
    hyperboloid, cluster, hyperboloid_subsample, subsample_indices = Hyperboloid.sample(n_points, double=double, supersample=supersample, supersample_factor=supersample_factor)
    color = Hyperboloid.S(hyperboloid[:, 2]) # curvature (proxy) for color
    color = np.array(color)
    hyperboloid += noise * np.random.randn(*hyperboloid.shape)
    return_dict = {
        'data': hyperboloid,
        'cluster': cluster,
        'color': color,
        'data_supersample': hyperboloid_subsample,
        'subsample_indices': subsample_indices
    }
    return return_dict