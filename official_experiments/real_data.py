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


def pbmc_experiment():
    save_dir = 'outputs/official_experiments/pbmc'
    os.makedirs(save_dir, exist_ok=True)
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
    return_dict = get_pruned_unpruned_graph(pbmc_data_X_pca, exp_params, verbose=True, reattach=False)
    _, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']
    pbmc_data_orcml = pbmc_data_orcml[list(G_orcml.nodes())]
    pbmc_labels_orcml_int = pbmc_labels_int[list(G_orcml.nodes())]
    # connected components of original
    n_connected_components, _ = scipy.sparse.csgraph.connected_components(A_original)
    print(f'Number of connected components in original graph: {n_connected_components}')

    # connected components of orcml
    n_connected_components, _ = scipy.sparse.csgraph.connected_components(A_orcml)
    print(f'Number of connected components in orcml graph: {n_connected_components}')

    ######################## Unpruned Graph ########################
    # run umap
    umap_original = UMAP(A_original, n_neighbors=10, n_components=2)
    pbmc_data_original.obsm['X_umap'] = umap_original

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

    # plot umap and spectral embeddings
    fig = sc.pl.embedding(pbmc_data_orcml, basis="spectral_0_1", color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/spectral_embedding_evecs_0_1_orcml.png', dpi=1200)

    fig = sc.pl.embedding(pbmc_data_orcml, basis="spectral_0_2", color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/spectral_embedding_evecs_0_2_orcml.png', dpi=1200)

    fig = sc.pl.embedding(pbmc_data_orcml, basis="spectral_1_2", color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/spectral_embedding_evecs_1_2_orcml.png', dpi=1200)

    fig = sc.pl.embedding(pbmc_data_original, basis="spectral_0_1", color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/spectral_embedding_evecs_0_1_original.png', dpi=1200)

    fig = sc.pl.embedding(pbmc_data_original, basis="spectral_0_2", color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/spectral_embedding_evecs_0_2_original.png', dpi=1200)

    fig = sc.pl.embedding(pbmc_data_original, basis="spectral_1_2", color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/spectral_embedding_evecs_1_2_original.png', dpi=1200)

    fig = sc.pl.umap(pbmc_data_orcml, color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/umap_orcml.png', dpi=1200)

    fig = sc.pl.umap(pbmc_data_original, color='bulk_labels', return_fig=True)
    fig.savefig(f'{save_dir}/umap_original.png', dpi=1200)    

def alm_allen_brain_experiment():
    save_dir = 'outputs/official_experiments/alm_allen_brain'
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
    _, A_original, _, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']

    # convert alm labels to color
    colors = ['#33B964', '#257CDF',"#D10101" ]
    alm_labels_color = np.array([colors[label] for label in alm_labels_int])

    # isomap embedding of the original graph
    y_original = isomap(A_original, 2)
    plot_data_2D(y_original, color=alm_labels_color, title=None, axes=True, node_size=0.3)
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
    plot_data_2D(y_orcml_rot, color=alm_labels_color, title=None, axes=True, node_size=0.3)
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
    plot_data_2D(umap_original, color=alm_labels_color, title=None, axes=True, node_size=1)
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
    plot_data_2D(tsne_original, color=alm_labels_color, title=None, axes=True, node_size=1)
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
    plot_data_2D(umap_orcml, color=alm_labels_color, title=None, axes=True, node_size=1)
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
    plot_data_2D(tsne_orcml, color=alm_labels_color, title=None, axes=True, node_size=1)
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
    print('Running PBMC experiment')
    pbmc_experiment()
    print('\n\nRunning ALM Allen Brain experiment')
    alm_allen_brain_experiment()