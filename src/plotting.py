import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import plotly.graph_objs as go


# plotting functions

def plot_data_2D(X, y, title, exp_name=None, filename=None):
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
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)

def plot_graph_2D(X, graph, title, node_color='#1f78b4', edge_color='lightgray', node_size=10, colorbar=False, exp_name=None, filename=None):
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
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)

def plot_data_3D(X, color, title, exp_name=None, filename=None):
    marker_data = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=color,
            colorscale='Viridis',
            opacity=0.8
        ),
    )
    fig = go.Figure(data=[marker_data])
    fig.update_layout(title=title)
    fig.update_layout(scene=dict(aspectmode='data'))
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        fig.write_image(path)
    fig.show()

def plot_graph_3D(X, graph, title, node_color='#1f78b4', edge_color='lightgrey', colorbar=False, exp_name=None, filename=None):
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
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in graph.edges():
        x0, y0, z0 = X[edge[0]]
        x1, y1, z1 = X[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(
            width=0.1 if isinstance(edge_color, str) else 1,
            color=np.repeat(edge_color, 3),
            colorscale='Spectral_r',
            colorbar=dict(
                thickness=15,
                title='ORC',
                xanchor='left',
                titleside='right',
            ) if colorbar else None,
            cmin=0,
            cmax=1,
        ),
    )

    marker_data = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=node_color,
            colorscale='Viridis',
            opacity=0.8
        ),
    )
    fig = go.Figure(data=[edge_trace, marker_data])
    fig.update_layout(title=title)
    fig.update_layout(scene=dict(aspectmode='data'))
    if colorbar:
        fig.update_layout(coloraxis=dict(colorscale='Viridis', colorbar=dict(title='Color')))
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        fig.write_image(path)
    fig.show()    

def plot_emb(Y, color, title, cmap=plt.cm.Spectral, exp_name=None, filename=None):
    """
    Plot the embedding of the data.
    Parameters
    
    Y : array-like, shape (n_samples, 2)
        The coordinates of the points in the embedding.
    title : str
        The title of the plot.
    """
    if Y.shape[1] == 1:
        plt.scatter(Y, np.zeros(Y.shape), c=color, cmap=cmap, s=10)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=cmap, s=10)
    plt.title(title)
    plt.gca().set_axis_off()
    plt.show()
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)