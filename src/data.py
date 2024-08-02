from sklearn import datasets
from src.manifold import Torus, Hyperboloid
import numpy as np
import torch
import torchvision
# Data generation functions

def concentric_circles(n_points, factor, noise, supersample=False, supersample_factor=2.5, noise_thresh=0.275):
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
    
    # clip noise and resample if necessary
    z =  noise*np.random.randn(*circles.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    circles += z

    return_dict = {
        'data': circles,
        'cluster': cluster,
        'data_supersample': circles_supersample,
        'subsample_indices': subsample_indices
    }
    return return_dict

def swiss_roll(n_points, noise, dim=3, supersample=False, supersample_factor=1.5, noise_thresh=0.275):
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

    # clip noise and resample if necessary
    z =  noise*np.random.randn(*swiss_roll.shape)
    resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    while len(resample_indices) > 0:
        z[resample_indices] = noise*np.random.randn(*z[resample_indices].shape)
        resample_indices = np.where(np.linalg.norm(z, axis=1) > noise_thresh)[0]
    swiss_roll += z

    return_dict = {
        'data': swiss_roll,
        'cluster': None,
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


def get_mnist_data(n_samples, label=None):
    """
    Get n_samples MNIST data points with the specified label. If label is None, get n_samples random data points.
    Parameters:

    n_samples: int
        Number of data points to get
    label: int or None
        Label of the data points to get. If None, get random data points.
    Returns:
    ----------
    mnist_data: np.ndarray
        n_samples x 784 array of MNIST data points
    mnist_labels: np.ndarray
        n_samples array of MNIST labels
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_data = torch.stack([x for x, _ in mnist]).numpy().astype(np.float64)
    # scale so distances are in a reasonable range
    mnist_data /= 40
    mnist_labels = torch.tensor([y for _, y in mnist]).numpy().astype(np.float64)
    if label is not None:
        label_indices = np.where(mnist_labels == label)[0]
        np.random.seed(0)
        np.random.shuffle(label_indices)
        label_indices = label_indices[:n_samples]
        mnist_data = mnist_data[label_indices]
        mnist_labels = mnist_labels[label_indices]
    else:
        np.random.seed(0)
        indices = np.random.choice(mnist_data.shape[0], n_samples, replace=False)
        mnist_data = mnist_data[indices]
        mnist_labels = mnist_labels[indices]
    return mnist_data, mnist_labels


def get_fmnist_data(n_samples, label=None):
    """
    Get n_samples Fashion MNIST data points with the specified label. If label is None, get n_samples random data points.
    Parameters:

    n_samples: int
        Number of data points to get
    label: int or None
        Label of the data points to get. If None, get random data points.
    Returns:
    ----------
    fmnist_data: np.ndarray
        n_samples x 784 array of Fashion MNIST data points
    fmnist_labels: np.ndarray
        n_samples array of Fashion MNIST labels
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    fmnist = torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    fmnist_data = torch.stack([x for x, _ in fmnist]).numpy().astype(np.float64)
    # scale so distances are in a reasonable range
    fmnist_data /= 40
    fmnist_labels = torch.tensor([y for _, y in fmnist]).numpy().astype(np.float64)
    if label is not None:
        label_indices = np.where(fmnist_labels == label)[0]
        np.random.seed(0)
        np.random.shuffle(label_indices)
        label_indices = label_indices[:n_samples]
        fmnist_data = fmnist_data[label_indices]
        fmnist_labels = fmnist_labels[label_indices]
    else:
        np.random.seed(0)
        indices = np.random.choice(fmnist_data.shape[0], n_samples, replace=False)
        fmnist_data = fmnist_data[indices]
        fmnist_labels = fmnist_labels[indices]
    return fmnist_data, fmnist_labels