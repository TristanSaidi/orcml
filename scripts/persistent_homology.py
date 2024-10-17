import os
from data.data import *
from src.experiments.embeddings import *
from src.experiments.ph import *
from src.orcmanl import *
from src.utils.plotting import *
from src.utils.eval_utils import *
from src.utils.graph_utils import *
from src.experiments.utils.exp_utils import *
from src.experiments.baselines import *
from src.utils.plotting import *
import datetime

if __name__ == '__main__':
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'./outputs/official_experiments/{experiment_name}/persistent_homology/concentric_circles'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\nRunning persistent homology experiments for concentric circles\n')
    # concentric circles
    n_points = 1000
    factor = 0.385
    noise = 0.09
    noise_threshold = 0.275

    exp_params = {
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }

    return_dict = concentric_circles(n_points, factor, noise, noise_threshold, supersample_factor=1.0)
    circles, cluster, circles_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
    return_dict = prune_helper(circles_supersample, exp_params)
    G_supersample, A_supersample = return_dict['G_original'], return_dict['A_original']

    plot_graph_2D(circles_supersample, G_supersample, title=None, node_color=None, node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/concentric_circles_noiseless_graph.png')
    plt.close()

    dgms_circles = rips_ph(A_supersample, maxdim=1)
    plot_barcode(dgms_circles, thresh=2)
    plt.savefig(f'{save_dir}/concentric_circles_noiseless_barcode.png')
    plt.close()

    plot_persistence_diagram(dgms_circles)
    plt.savefig(f'{save_dir}/concentric_circles_noiseless_persistence_diagram.png', dpi=1200)
    plt.close()

    return_dict = prune_helper(circles, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    plot_graph_2D(circles, G_original, title=None, node_color=None, node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/concentric_circles_noisy_graph.png')
    plt.close()

    dgms_original = rips_ph(A_original, maxdim=1)
    plot_barcode(dgms_original, thresh=2)
    plt.savefig(f'{save_dir}/concentric_circles_noisy_barcode.png')
    plt.close()

    plot_persistence_diagram(dgms_original, thresh=2.55)
    plt.savefig(f'{save_dir}/concentric_circles_noisy_persistence_diagram.png', dpi=1200)
    plt.close()

    plot_graph_2D(circles, G_orcmanl, title=None, node_color=None, node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/concentric_circles_orcmanl_graph.png')
    plt.close()

    dgms_orcmanl = rips_ph(A_orcmanl, maxdim=1)
    plot_barcode(dgms_orcmanl, thresh=2)
    plt.savefig(f'{save_dir}/concentric_circles_orcmanl_barcode.png')
    plt.close()

    plot_persistence_diagram(dgms_orcmanl, thresh=2.55)
    plt.savefig(f'{save_dir}/concentric_circles_orcmanl_persistence_diagram.png', dpi=1200)
    plt.close()

    # print the Wasserstein distance between the persistence diagrams
    wass_dist_original = ph_dist(dgms_circles, dgms_original)
    wass_dist_orcmanl = ph_dist(dgms_circles, dgms_orcmanl)

    distances_per_dim = zip(wass_dist_original, wass_dist_orcmanl)

    for homology_class, wass_dist in zip(['H0', 'H1'], distances_per_dim):
        print(f'Wasserstein distance between {homology_class} of unpruned concentric circles and unpruned noisy concentric circles: {wass_dist[0]:.4f}')
        print(f'Wasserstein distance between {homology_class} of pruned concentric circles and orcmanl-pruned noisy concentric circles: {wass_dist[1]:.4f}')

    print(f'\n\n\n')
    print(f'Finished running persistent homology experiments for concentric circles\n\n\n')

    # torii
    print(f'\nRunning persistent homology experiments for torii\n')
    save_dir = f'./outputs/official_experiments/{experiment_name}/persistent_homology/torus'
    os.makedirs(save_dir, exist_ok=True)
    n_points = 3000
    noise = 0.4
    noise_threshold = 0.75

    return_dict = torus(n_points=n_points, noise=noise, noise_thresh=noise_threshold, supersample=True, double=True, supersample_factor=1.0)
    torus_data, torus_cluster, torus_data_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']

    exp_params = {
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }

    return_dict = prune_helper(torus_data_supersample, exp_params)
    G_supersample, A_supersample = return_dict['G_original'], return_dict['A_original']

    fig = plot_graph_3D(torus_data_supersample, G_supersample, title=None, node_color='#1f77b4', node_size=2, edge_width=0.25)
    fig.write_image(f'{save_dir}/torus_noiseless_graph.png', width=1200, height=1200, scale=10)

    dgms_torus = rips_ph(A_supersample, maxdim=2)
    plot_barcode(dgms_torus, thresh=8)
    plt.savefig(f'{save_dir}/original_torus_noiseless_barcode.png')

    plot_persistence_diagram(dgms_torus, thresh=8.5)
    plt.savefig(f'{save_dir}/original_torus_noiseless_persistence_diagram.png', dpi=1200)

    return_dict = prune_helper(torus_data, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    fig = plot_graph_3D(torus_data, G_original, title=None, node_color='#1f77b4', node_size=2, edge_width=0.25)
    fig.write_image(f'{save_dir}/torus_noisy_graph.png', width=1200, height=1200, scale=10)

    dgms_original_torus = rips_ph(A_original, maxdim=2)
    plot_barcode(dgms_original_torus, thresh=8)
    plt.savefig(f'{save_dir}/original_torus_noisy_barcode.png')

    plot_persistence_diagram(dgms_original_torus, thresh=8.5)
    plt.savefig(f'{save_dir}/original_torus_noisy_persistence_diagram.png', dpi=1200)

    fig = plot_graph_3D(torus_data, G_orcmanl, title=None, node_color='#1f77b4', node_size=2, edge_width=0.25)
    fig.write_image(f'{save_dir}/torus_orcmanl_graph.png', width=1200, height=1200, scale=10)

    dgms_original_orcmanl = rips_ph(A_orcmanl, maxdim=2)
    plot_barcode(dgms_original_orcmanl, thresh=8)
    plt.savefig(f'{save_dir}/original_torus_orcmanl_barcode.png')

    plot_persistence_diagram(dgms_original_orcmanl, thresh=8.5)
    plt.savefig(f'{save_dir}/original_torus_orcmanl_persistence_diagram.png', dpi=1200)

    # print the Wasserstein distance between the persistence diagrams
    wass_dist_original = ph_dist(dgms_torus, dgms_original_torus)
    wass_dist_orcmanl = ph_dist(dgms_torus, dgms_original_orcmanl)

    distances_per_dim = zip(wass_dist_original, wass_dist_orcmanl)

    for homology_class, wass_dist in zip(['H0', 'H1', 'H2'], distances_per_dim):
        print(f'Wasserstein distance between {homology_class} of unpruned torus and unpruned noisy torus: {wass_dist[0]:.4f}')
        print(f'Wasserstein distance between {homology_class} of pruned torus and orcmanl-pruned noisy torus: {wass_dist[1]:.4f}')