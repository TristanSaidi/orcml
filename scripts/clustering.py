import os
from data.data import *
from src.experiments.embeddings import *
from src.orcmanl import *
from src.utils.plotting import *
from src.utils.eval_utils import *
from src.utils.graph_utils import *
from src.experiments.utils.exp_utils import *
from src.experiments.baselines import *
import datetime
from sklearn import metrics


def k_means(X, n_clusters):
    """ 
    Run K-means clustering on the data X
    Args:
        X: np.array of shape (n_samples, n_features)
        n_clusters: int, number of clusters
    Returns:
        labels: np.array of shape (n_samples, )
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=None, init='k-means++', algorithm='lloyd', n_init=10).fit(X)
    return kmeans.labels_

if __name__ == '__main__':
    np.random.seed(20)
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ############################## Concentric Circles ##############################
    save_dir = f'./outputs/official_experiments/{experiment_name}/clustering/concentric_circles'
    os.makedirs(save_dir, exist_ok=True)

    n_points = 4000
    factor = 0.385
    noise = 0.175
    noise_threshold = 0.28

    exp_params = {
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }

    return_dict = concentric_circles(n_points, factor, noise, noise_threshold)
    circles, cluster, circles_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
    return_dict = prune_helper(circles, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['non_shortcut_edges'],
    )
    print(f'Concentric Circles: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcmanl = spectral_embedding(A_orcmanl, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcmanl = k_means(emb_orcmanl, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcmanl = metrics.adjusted_rand_score(cluster[G_orcmanl.nodes()], labels_orcmanl)

    print(f'Concentric Circles unpruned ARI: {rand_index_original}')
    print(f'Concentric Circles orcmanl ARI: {rand_index_orcmanl}')

    plot_graph_2D(circles, G_original, title=None, node_color=labels_original[G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/original_clusters_concentric_circles.png', dpi=1000)


    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])
    plot_graph_2D(circles, G_original, title=None, node_color=labels_orcmanl[reverse_indices][G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/orcmanl_clusters_concentric_circles.png', dpi=1000)

    ############################## Concentric Circles ##############################

    ############################## Hyperboloids ##############################
    n_points = 3000
    noise = 0.25
    noise_thresh = 0.25
    dataset_info = {
                'name': 'hyperboloids',
                'n_points': n_points,
                'noise': noise,
                'noise_thresh': noise_thresh
    }
    return_dict = hyperboloid(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, double=True)
    hyperboloid_data, cluster, hyperboloid_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']

    save_dir = f'./outputs/official_experiments/{experiment_name}/clustering/concentric_hyperboloids'
    os.makedirs(save_dir, exist_ok=True)

    return_dict = prune_helper(hyperboloid_data, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['non_shortcut_edges'],
    )
    print(f'Hyperboloids: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcmanl = spectral_embedding(A_orcmanl, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcmanl = k_means(emb_orcmanl, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcmanl = metrics.adjusted_rand_score(cluster[G_orcmanl.nodes()], labels_orcmanl)

    print(f'Hyperboloids unpruned ARI: {rand_index_original}')
    print(f'Hyperboloids orcmanl ARI: {rand_index_orcmanl}')

    # map labels to css code for red (#FF5733) and purple (#7A33FF)
    labels_original = np.array(['#FF5733' if i == 0 else '#7A33FF' for i in labels_original])
    labels_orcmanl = np.array(['#FF5733' if i == 0 else '#7A33FF' for i in labels_orcmanl])

    fig = plot_graph_3D(hyperboloid_data, G_original, title=None, node_color=labels_original, node_size=3, edge_width=0.5)
    fig.write_image(f'{save_dir}/original_hyperboloids_clusters.png', width=1200, height=1200, scale=10) 
    
    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])
    fig = plot_graph_3D(hyperboloid_data, G_original, title=None, node_color=labels_orcmanl[reverse_indices], node_size=3, edge_width=0.5)
    fig.write_image(f'{save_dir}/orcmanl_hyperboloids_clusters.png', width=1200, height=1200, scale=10) 
    ############################## Hyperboloids ##############################

    ############################## Moons ##############################
    n_points = 3000
    noise = 0.215
    noise_thresh = 0.21
    dataset_info = {
                'name': 'moons',
                'n_points': n_points,
                'noise': noise,
                'noise_thresh': noise_thresh
    }

    return_dict = moons(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
    moons_data, cluster, moons_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']

    save_dir = f'./outputs/official_experiments/{experiment_name}/clustering/moons'
    os.makedirs(save_dir, exist_ok=True)

    return_dict = prune_helper(moons_data, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['non_shortcut_edges'],
    )
    print(f'Moons: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcmanl = spectral_embedding(A_orcmanl, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcmanl = k_means(emb_orcmanl, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcmanl = metrics.adjusted_rand_score(cluster[G_orcmanl.nodes()], labels_orcmanl)
    
    print(f'Moons unpruned ARI: {rand_index_original}')
    print(f'Moons orcmanl ARI: {rand_index_orcmanl}')

    plot_graph_2D(moons_data, G_original, title=None, node_color=labels_original[G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/original_clusters_moons.png', dpi=1000)

    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])
    plot_graph_2D(moons_data, G_original, title=None, node_color=labels_orcmanl[reverse_indices][G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/orcmanl_clusters_moons.png', dpi=1000)      
    ############################## Moons ##############################

    ############################## Quadratics ##############################
    n_points = 2000
    noise = 0.20
    noise_thresh = 0.475

    dataset_info = {
        'name': 'quadratics',
        'n_points': n_points,
        'noise': noise,
        'noise_thresh': noise_thresh
    }

    return_dict = quadratics(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
    quadratics_data, cluster, quadratics_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']

    save_dir = f'./outputs/official_experiments/{experiment_name}/clustering/quadratics'
    os.makedirs(save_dir, exist_ok=True)

    return_dict = prune_helper(quadratics_data, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['non_shortcut_edges'],
    )
    print(f'Quadratics: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcmanl = spectral_embedding(A_orcmanl, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcmanl = k_means(emb_orcmanl, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcmanl = metrics.adjusted_rand_score(cluster[G_orcmanl.nodes()], labels_orcmanl)

    print(f'Quadratics unpruned ARI: {rand_index_original}')
    print(f'Quadratics orcmanl ARI: {rand_index_orcmanl}')

    plot_graph_2D(quadratics_data, G_original, title=None, node_color=labels_original[G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/original_clusters_quadratics.png', dpi=1000)

    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])
    plot_graph_2D(quadratics_data, G_original, title=None, node_color=labels_orcmanl[reverse_indices][G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/orcmanl_clusters_quadratics.png', dpi=1000)
    ############################## Quadratics ##############################