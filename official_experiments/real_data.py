import os
import torch
import torchvision
from src.data import *
from src.embeddings import *
from src.orcml import *
from src.plotting import *
from src.eval_utils import *
from official_experiments.experiments import *
import skdim as skd
import pandas as pd
import scanpy as sc
from natsort import natsorted
import decoupler as dc

sc.set_figure_params(dpi_save=1200)

def pbmc_experiment(experiment_dir):
    save_dir = f'{experiment_dir}/pbmc'
    os.makedirs(save_dir, exist_ok=True)

    save_dir_original = f'{save_dir}/original'
    save_dir_orcml = f'{save_dir}/orcml'
    os.makedirs(save_dir_original, exist_ok=True)
    os.makedirs(save_dir_orcml, exist_ok=True)
    # PBMC 10k dataset
    pbmc_data = sc.datasets.pbmc68k_reduced()
    sc.tl.pca(pbmc_data, svd_solver='arpack')
    sc.pl.pca(pbmc_data, color='CST3')
    sc.pl.pca_variance_ratio(pbmc_data, log=True)

    pbmc_data_original = pbmc_data.copy()
    pbmc_data_orcml = pbmc_data.copy()

    # use PCA embeddings with 40 pcs
    pbmc_data_X_pca = pbmc_data.obsm['X_pca'][:, :40]
    pbmc_labels = pbmc_data.obs['bulk_labels'] # string

    pbmc_labels_int, label_dict = pd.factorize(pbmc_labels)

    # compute nn graphs (raw and orcml pruned)
    exp_params = {
        'mode': 'nbrs',
        'n_neighbors': 10,
        'epsilon': None,
        'lda': 1e-5,
        'delta': 0.8
    }

    # get pruned and unpruned graphs
    return_dict = get_pruned_unpruned_graph(pbmc_data_X_pca, exp_params, verbose=True)
    G_original, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']
    pbmc_data_orcml = pbmc_data_orcml[list(G_orcml.nodes())]
    pbmc_labels_orcml_int = pbmc_labels_int[list(G_orcml.nodes())]
    # connected components of original
    n_connected_components, _ = scipy.sparse.csgraph.connected_components(A_original)
    print(f'Number of connected components in original graph: {n_connected_components}')

    # connected components of orcml
    n_connected_components, _ = scipy.sparse.csgraph.connected_components(A_orcml)
    print(f'Number of connected components in orcml graph: {n_connected_components}')

    cmap = plt.cm.Spectral
    unique_pbmc_labels_str = np.unique(pbmc_labels.to_numpy())


    from matplotlib.lines import Line2D

    # Create the figure
    _, ax = plt.subplots()

    # Number of unique labels
    n_labels = len(unique_pbmc_labels_str)

    # Create legend handles with dots
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label,
            markerfacecolor=cmap(i / (n_labels - 1)), markersize=10)
        for i, label in enumerate(unique_pbmc_labels_str)
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='center')

    # Remove axes (since we don't want any plot)
    ax.set_axis_off()

    # Display the legend
    plt.show()
    plt.savefig(f'{save_dir}/pbmc_legend.png', dpi=1200)

    ######################## Unpruned Graph ########################
    # run umap
    umap_original = UMAP(A_original, n_neighbors=10, n_components=2)
    pbmc_data_original.obsm['X_umap'] = umap_original

    # run tSNE
    tsne_original = tsne(A_original, n_components=2)
    pbmc_data_original.obsm['X_tsne'] = tsne_original

    # run spectral embedding: might need to hand-select eigenvectors here because of the REP
    spectral_embedding_original = spectral_embedding(A_original, n_components=3)
    spectral_embedding_original_0_1 = spectral_embedding_original[:, [0,1]]
    spectral_embedding_original_0_2 = spectral_embedding_original[:, [0,2]]
    spectral_embedding_original_1_2 = spectral_embedding_original[:, [1,2]]

    pbmc_data_original.obsm['X_spectral_0_1'] = spectral_embedding_original_0_1
    pbmc_data_original.obsm['X_spectral_0_2'] = spectral_embedding_original_0_2
    pbmc_data_original.obsm['X_spectral_1_2'] = spectral_embedding_original_1_2
    ######################## Unpruned Graph ########################

    ######################## ORCManL Pruned Graph ########################
    # run umap
    umap_orcml = UMAP(A_orcml, n_neighbors=10, n_components=2, X=pbmc_data_X_pca[list(G_orcml.nodes())])
    pbmc_data_orcml.obsm['X_umap'] = umap_orcml

    # run tSNE
    tsne_orcml = tsne(A_orcml, n_components=2, X=pbmc_data_X_pca[list(G_orcml.nodes())])
    pbmc_data_orcml.obsm['X_tsne'] = tsne_orcml
    
    # run spectral embedding: might need to hand-select eigenvectors here because of the REP
    spectral_embedding_orcml = spectral_embedding(A_orcml, n_components=3)
    spectral_embedding_orcml_0_1 = spectral_embedding_orcml[:, [0,1]]
    spectral_embedding_orcml_0_2 = spectral_embedding_orcml[:, [0,2]]
    spectral_embedding_orcml_1_2 = spectral_embedding_orcml[:, [1,2]]

    # add gaussian noise as cc's get mapped to the same point
    noise = np.random.normal(0, scale=0.0003, size=spectral_embedding_orcml_0_1.shape)
    spectral_embedding_orcml_0_1 += noise
    spectral_embedding_orcml_0_2 += noise
    spectral_embedding_orcml_1_2 += noise 
    pbmc_data_orcml.obsm['X_spectral_0_1'] = spectral_embedding_orcml_0_1
    pbmc_data_orcml.obsm['X_spectral_0_2'] = spectral_embedding_orcml_0_2
    pbmc_data_orcml.obsm['X_spectral_1_2'] = spectral_embedding_orcml_1_2
    ######################## ORCManL Pruned Graph ########################


    def plot_pbmc(X, graph, node_color, emb_alg, save_dir, extra_title=None):
        plot_graph_2D(X, graph, node_color=node_color, title=None, node_size=1, edge_width=0.4)
        plt.axis('on')
        # turn on axes
        ax = plt.gca()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xlabel(f'{emb_alg}1')
        # make xlabel bigger
        ax.xaxis.label.set_size(20)
        ax.set_ylabel(f'{emb_alg}2')
        # make ylabel bigger
        ax.yaxis.label.set_size(20)
        # turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        if extra_title is not None:
            title = f'{emb_alg}_{extra_title}'
        else:
            title = f'{emb_alg}'
        plt.savefig(f'{save_dir}/pbmc_{title}.png', dpi=1200)


    # umap original  
    plot_pbmc(umap_original, G_original, pbmc_labels_int[G_original.nodes()], 'UMAP', save_dir_original)

    # tsne original
    plot_pbmc(tsne_original, G_original, pbmc_labels_int[G_original.nodes()], 'tSNE', save_dir_original)

    # spectral embedding original
    plot_pbmc(spectral_embedding_original_0_1, G_original, pbmc_labels_int[G_original.nodes()], 'spectral', save_dir_original, '0_1')
    plot_pbmc(spectral_embedding_original_0_2, G_original, pbmc_labels_int[G_original.nodes()], 'spectral', save_dir_original, '0_2')
    plot_pbmc(spectral_embedding_original_1_2, G_original, pbmc_labels_int[G_original.nodes()], 'spectral', save_dir_original, '1_2')

    # umap orcml
    umap_orcml_unscrambled = umap_orcml[np.argsort(list(G_orcml.nodes()))]
    plot_pbmc(umap_orcml_unscrambled, G_orcml, pbmc_labels_orcml_int, 'UMAP', save_dir_orcml)

    # tsne orcml
    tsne_orcml_unscrambled = tsne_orcml[np.argsort(list(G_orcml.nodes()))]
    plot_pbmc(tsne_orcml_unscrambled, G_orcml, pbmc_labels_orcml_int, 'tSNE', save_dir_orcml)

    # spectral embedding orcml
    spectral_embedding_orcml_0_1_unscrambled = spectral_embedding_orcml_0_1[np.argsort(list(G_orcml.nodes()))]
    spectral_embedding_orcml_0_2_unscrambled = spectral_embedding_orcml_0_2[np.argsort(list(G_orcml.nodes()))]
    spectral_embedding_orcml_1_2_unscrambled = spectral_embedding_orcml_1_2[np.argsort(list(G_orcml.nodes()))]
    plot_pbmc(spectral_embedding_orcml_0_1_unscrambled, G_orcml, pbmc_labels_orcml_int, 'spectral', save_dir_orcml, '0_1')
    plot_pbmc(spectral_embedding_orcml_0_2_unscrambled, G_orcml, pbmc_labels_orcml_int, 'spectral', save_dir_orcml, '0_2')
    plot_pbmc(spectral_embedding_orcml_1_2_unscrambled, G_orcml, pbmc_labels_orcml_int, 'spectral', save_dir_orcml, '1_2')



def alm_allen_brain_experiment(experiment_dir):
    save_dir = f'{experiment_dir}/alm_allen_brain'
    os.makedirs(save_dir, exist_ok=True)
    # Brain scRNAseq dataset. Indices are 1:12552
    path = 'data/MouseV1_MouseALM_HumanMTG/MouseV1_MouseALM_HumanMTG.csv'
    data = pd.read_csv(path)

    labels_path = 'data/MouseV1_MouseALM_HumanMTG/MouseV1_MouseALM_HumanMTG_Labels3.csv'
    labels = pd.read_csv(labels_path)

    # Mouse ALM data: indices are 20681:34735
    alm_indices = (20681, 34735)
    n_points = 10000

    alm_data = data.iloc[alm_indices[0]:alm_indices[0]+n_points, :].values
    alm_float = alm_data[:, 1:].astype(float)
    alm_labels = labels.iloc[alm_indices[0]:alm_indices[0]+n_points].values

    # create dictionary mapping labels to integers
    label_dict = {label: i for i, label in enumerate(np.unique(alm_labels))}
    # convert labels to integers
    alm_labels_int = np.array([label_dict[label[0]] for label in alm_labels])

    # use scanpy to preprocess the data - identify highly variable genes, scale the data, and perform PCA
    adata_alm = sc.AnnData(alm_float)

    # logarithmize the data
    sc.pp.log1p(adata_alm)
    sc.pp.highly_variable_genes(adata_alm, n_top_genes=2000)

    sc.pp.scale(adata_alm)

    adata_filtered_alm = adata_alm[:, adata_alm.var.highly_variable]
    # perform PCA
    sc.tl.pca(adata_filtered_alm, svd_solver='arpack')
    # plot the variance ratio
    sc.pl.pca_variance_ratio(adata_filtered_alm)
    alm_data_float_filtered = adata_filtered_alm.obsm['X_pca']

    exp_params = {
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }
    # get pruned and unpruned graphs
    return_dict = get_pruned_unpruned_graph(alm_data_float_filtered, exp_params, verbose=True)
    G_original, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']

    # convert alm labels to color
    colors = ['#33B964', '#257CDF',"#D10101" ]
    alm_labels_color = np.array([colors[label] for label in alm_labels_int])

    # isomap embedding of the original graph
    y_original = isomap(A_original, 2)
    plot_graph_2D(y_original, G_original, node_color=alm_labels_color[G_original.nodes()], title=None, node_size=0.3, edge_width=0.1)
    plt.axis('on')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('isomap1')
    ax.set_ylabel('isomap2')
    # turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{save_dir}/alm_original_isomap_10k_2D.png', dpi=1200)

    # isomap embedding of the orcml pruned graph
    y_orcml = isomap(A_orcml, 2, X=alm_data_float_filtered)
    # rotate the plot
    y_orcml_rot = y_orcml[:, [1, 0]]
    plot_graph_2D(y_orcml_rot, G_orcml, node_color=alm_labels_color[G_orcml.nodes()], title=None, node_size=0.3, edge_width=0.1)
    plt.axis('on')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('isomap1')
    ax.set_ylabel('isomap2')
    # turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{save_dir}/alm_orcml_isomap_10k_2D.png', dpi=1200)
    plt.show()

    # umap embedding of the original graph
    umap_original = UMAP(A_original, n_neighbors=20, n_components=2)
    plot_graph_2D(umap_original, G_original, node_color=alm_labels_color[G_original.nodes()], title=None, node_size=0.3, edge_width=0.1)
    plt.axis('on')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    # turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{save_dir}/alm_original_umap_10k_2D.png', dpi=1200)

    # tsne embedding of the original graph
    tsne_original = tsne(A_original, n_components=2)
    plot_graph_2D(tsne_original, G_original, node_color=alm_labels_color[G_original.nodes()], title=None, node_size=0.3, edge_width=0.1)
    plt.axis('on')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('tSNE1')
    ax.set_ylabel('tSNE2')
    # turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{save_dir}/alm_original_tsne_10k_2D.png', dpi=1200)

    # umap embedding of the orcml pruned graph
    umap_orcml = UMAP(A_orcml, n_neighbors=20, n_components=2, X=alm_data_float_filtered)
    plot_graph_2D(umap_orcml, G_orcml, node_color=alm_labels_color[G_orcml.nodes()], title=None, node_size=0.3, edge_width=0.1)
    plt.axis('on')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    # turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{save_dir}/alm_orcml_umap_10k_2D.png', dpi=1200)

    # tsne embedding of the orcml pruned graph
    tsne_orcml = tsne(A_orcml, n_components=2, X=alm_data_float_filtered)
    plot_graph_2D(tsne_orcml, G_orcml, node_color=alm_labels_color[G_orcml.nodes()], title=None, node_size=0.3, edge_width=0.1)
    plt.axis('on')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('tSNE1')
    ax.set_ylabel('tSNE2')
    # turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{save_dir}/alm_orcml_tsne_10k_2D.png', dpi=1200)

if __name__ == '__main__':
    import datetime
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = f'outputs/official_experiments/{experiment_name}'
    os.makedirs(experiment_dir, exist_ok=True)
    print('Running PBMC experiment')
    pbmc_experiment(experiment_dir)
    # print('\n\nRunning ALM Allen Brain experiment')
    # alm_allen_brain_experiment(experiment_name)