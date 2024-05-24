import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# plotting functions

def plot_data_2D(X, y, title):
    """
    Plot the data with the points colored by class membership.
    Parameters
    
    X : array-like, shape (n_samples, 2)
        The coordinates of the points.
    y : array-like, shape (n_samples,)
        The integer labels for class membership of each point.
    title : str
        The title of the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.gca().set_axis_off()
    plt.show()

def plot_graph_2D(X, graph, title, node_color='#1f78b4', edge_color='lightgray', node_size=10, colorbar=False):
    """
    Plot the graph with the desired node or edge coloring.
    Parameters
    
    X : array-like, shape (n_samples, 2)
        The coordinates of the nodes.
    graph : networkx.Graph
        The graph to plot.
    title : str
        The title of the plot.
    node_color : str
        The color of the nodes.
    edge_color : str
        The color of the edges.
    """
    nx.draw(graph, X, node_color=node_color, edge_color=edge_color, node_size=node_size, cmap=plt.cm.Spectral, edge_cmap=plt.cm.coolwarm, edge_vmin=0, edge_vmax=1)
    plt.title(title)
    plt.gca().set_aspect('equal')
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        plt.colorbar(sm)
    plt.show()

def plot_emb(Y, color, title):
    """
    Plot the embedding of the data.
    Parameters
    
    Y : array-like, shape (n_samples, 2)
        The coordinates of the points in the embedding.
    title : str
        The title of the plot.
    """
    if Y.shape[1] == 1:
        plt.scatter(Y, np.zeros(Y.shape), c=color, cmap=plt.cm.Spectral, s=10)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
    plt.title(title)
    plt.gca().set_axis_off()
    plt.show()