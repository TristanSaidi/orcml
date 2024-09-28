import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import plotly.graph_objs as go
import seaborn as sns
import gudhi
import persim

from src.eval_utils import *

# plotting functions

def plot_data_2D(X, color, title, node_size=10, axes=False, exp_name=None, filename=None, cmap=plt.cm.Spectral):
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
    plt.scatter(X[:, 0], X[:, 1], c=color, cmap=cmap, s=node_size)
    plt.title(title)
    plt.gca().set_aspect('equal')
    if not axes:
        plt.gca().set_axis_off()
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)

def plot_graph_2D(X, graph, title, node_color='#1f78b4', edge_color='lightgray', node_size=10, edge_width=1.0, colorbar=False, exp_name=None, filename=None, cmap=plt.cm.Spectral):
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
    if type(edge_color) == str:
        edge_cmap = plt.cm.viridis
    else:
        edge_cmap = plt.cm.coolwarm
    plt.figure(dpi=1200)
    nx.draw(graph, X, node_color=node_color, edge_color=edge_color, node_size=node_size, cmap=cmap, edge_cmap=edge_cmap, edge_vmin=-1, edge_vmax=1, width=edge_width)
    plt.title(title)
    plt.gca().set_aspect('equal')
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
        sm._A = []
        plt.colorbar(sm)
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)

def plot_data_3D(X, color, title, exp_name=None, filename=None, axes=False, node_size=3, opacity=1, cmap=None, labels=None, camera=None):
    # If labels are provided, we'll plot one trace per label/color group
    fig = go.Figure()
    if labels is not None:
        unique_colors = np.unique(color)  # Find unique color values
        for c in unique_colors:
            # Get the corresponding label for the color if available
            label_name = labels[c] if c in labels else f"Group {c}"
            # Filter points that match the current color group
            mask = (color == c)
            # convert to indices
            mask = np.where(mask)[0]
            fig.add_trace(go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=X[mask, 2],
                mode='markers',
                marker=dict(
                    size=node_size,
                    color=color[mask],  # Use the same color value for each group
                    colorscale=cmap,     # Use colormap for unique categories
                    opacity=opacity
                ),
                name=label_name,  # Use the label for the legend
                showlegend=True,
                legendgroup=label_name  # Group the legend by label
            ))
    else:
        # If no labels, plot with continuous color mapping across the entire dataset
        fig.add_trace(go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=node_size,
                color=color,  # Color mapped to a continuous array
                colorscale=cmap,  # Apply the colormap
                opacity=opacity,
                colorbar=dict(title="Color Scale"),  # Show colorbar for continuous colormap
            ),
            showlegend=False
        ))

    # Update layout for the legend to increase marker size in the legend
    fig.update_layout(
        title=title,
        legend=dict(
            x=0.85,  # Move the legend a little to the left
            y=1,  # Keep the legend at the top
            itemsizing='constant',  # Makes legend items consistent
            font=dict(size=12),
            itemclick='toggleothers',  # Click legend to toggle
            itemdoubleclick='toggle'  # Double click for fine toggle
        )
    )
    
    # Custom marker size in the legend
    fig.update_traces(marker=dict(size=node_size * 2), selector=dict(mode='markers'))

    fig.update_layout(title=title)
    fig.update_layout(scene=dict(aspectmode='data'))
    if not axes:
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    if camera is not None:
        fig.update_layout(scene_camera=camera)
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        fig.write_image(path)
    fig.show()
    return fig

def plot_graph_3D(X, graph, title, node_color='#1f78b4', node_size=3, edge_width=0.5, edge_color='lightgrey', colorbar=False, camera=None, exp_name=None, filename=None, axes=False, cmap='Viridis', opacity=None, cmin=-1, cmax=1, node_colorbar=False, node_colorbar_title=None):
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
            width=edge_width,
            color=np.repeat(edge_color, 3),
            colorscale='Spectral_r',
            colorbar=dict(
                thickness=15,
                title='ORC',
                xanchor='left',
                titleside='right',
            ) if colorbar else None,
            cmin=-1,
            cmax=1,
        ),
        opacity=opacity,
        showlegend=False
    )
    marker_data = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            colorbar=dict(
                title=node_colorbar_title,
                thickness=40,
                xanchor='left',
                titleside='right',
                tickfont=dict(size=30),
            ) if node_colorbar else None,
            colorscale=cmap,
            opacity=0.8,
            cmin=cmin,
            cmax=cmax
        ),
        showlegend=False
    )
    if node_size != 0:
        fig = go.Figure(data=[edge_trace, marker_data])
    else:
        fig = go.Figure(data=[edge_trace])
    fig.update_layout(title=title)
    fig.update_layout(scene=dict(aspectmode='data'))
    if camera is not None:
        fig.update_layout(scene_camera=camera)
    if not axes:
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    if colorbar:
        fig.update_layout(coloraxis=dict(colorscale='Viridis', colorbar=dict(title='Color')))
    # marker colorbar
    if node_colorbar:
        fig.update_layout(coloraxis=dict(colorscale=cmap, colorbar=dict(title='Color')))
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        fig.write_image(path)
    return fig   

def plot_emb(Y, color, title, cmap=plt.cm.Spectral, exp_name=None, filename=None):
    """
    Plot the embedding of the data.
    Parameters
    
    Y : array-like, shape (n_samples, 2)
        The coordinates of the points in the embedding.
    title : str
        The title of the plot.
    """
    plt.figure(dpi=1200)
    if Y.shape[1] == 1:
        plt.scatter(Y, np.zeros(Y.shape), c=color, cmap=cmap, s=10)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=cmap, s=10)
    plt.title(title)
    plt.gca().set_axis_off()
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)

def plot_barcode(dgms, exp_name=None, filename=None, thresh=None):
    """
    Plot a persistence barcode.
    Parameters
    ----------
    dgms : list
        List of persistence diagrams, where each element is a list [homology, persistence].
    """
    plt.figure()
    ax = gudhi.plot_persistence_barcode(dgms, max_intervals=25, alpha=0.9, legend=False)
    if thresh is not None:
        ax.set_xlim([0, thresh])
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)

def plot_persistence_diagram(dgms, thresh=None, exp_name=None, filename=None):
    """
    Plot a persistence diagram.
    Parameters
    ----------
    dgms : list
        List of persistence diagrams, where each element is a list [homology, persistence].
    """
    # convert to persim format
    plt.figure()
    num_dims = len(np.unique([dgm[0] for dgm in dgms]))
    dgms_converted = []
    for dim in range(num_dims):
        dgms_converted.append(np.array([dgm[1] for dgm in dgms if dgm[0] == dim]))
    if thresh is not None:
        xy_range = [-0.1, thresh, -0.1, thresh]
    else:
        xy_range = None
    persim.plot_diagrams(dgms_converted, xy_range=xy_range)
    if filename is not None and exp_name is not None:
        os.makedirs('figures', exist_ok=True)
        exp_dir = os.path.join('figures', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, filename)
        plt.savefig(path)
