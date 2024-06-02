from sklearn import datasets
from src.manifold import Torus, Hyperboloid
import numpy as np
# Data generation functions

def concentric_circles(n_samples, factor, noise):
    """ 
    Generate concentric circles with noise. 
    Parameters
    
    n_samples : int
        The number of samples to generate.
    factor : float
        The scaling factor between the circles.
    noise : float
        The standard deviation of the Gaussian noise.

    Returns
    -------
    X : array-like, shape (n_samples, 2)
        The generated samples.
    
    y : array-like, shape (n_samples,)
        The integer labels for class membership of each sample.
    """
    X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=factor)
    return X, y

def swiss_roll(n_points, noise, dim=3):
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
    swiss_roll : array-like, shape (n_points, dim)
        The generated Swiss roll.
    color : array-like, shape (n_points,)
        The color of each point.
    dim: int
        The dimension of the Swiss roll.
    """
    swiss_roll, color = datasets.make_swiss_roll(n_points, noise=noise)
    if dim == 2:
        swiss_roll = swiss_roll[:, [0, 2]]
    return swiss_roll, color

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
    """
    if double and R <= 2*r:
        raise Warning("Double torii will intersect")
    torus, thetas, cluster, torus_subsample, subsample_indices = Torus.sample(N=n_points, r=r, R=R, double=double, supersample=supersample, supersample_factor=supersample_factor)
    color = Torus.exact_curvatures(thetas, r, R)
    color = np.array(color)
    torus += noise * np.random.randn(*torus.shape)
    return torus, color, cluster, torus_subsample, subsample_indices

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
    return hyperboloid, color, cluster, hyperboloid_subsample, subsample_indices