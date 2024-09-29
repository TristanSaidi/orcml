import os
from src.data import *
from src.embeddings import *
from src.orcml import *
from src.plotting import *
from src.eval_utils import *
from src.baselines import *
from official_experiments.experiments import *
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
    return_dict = get_pruned_unpruned_graph(circles, exp_params)
    G_original, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['preserved_edges'],
    )
    print(f'Concentric Circles: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcml = spectral_embedding(A_orcml, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcml = k_means(emb_orcml, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcml = metrics.adjusted_rand_score(cluster[G_orcml.nodes()], labels_orcml)

    print(f'Concentric Circles unpruned ARI: {rand_index_original}')
    print(f'Concentric Circles ORCML ARI: {rand_index_orcml}')

    plot_graph_2D(circles, G_original, title=None, node_color=labels_original[G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/original_clusters_concentric_circles.png', dpi=1000)


    reverse_indices = np.array([np.where(np.array(list(G_orcml)) == i)[0][0] for i in range(len(G_orcml.nodes()))])
    plot_graph_2D(circles, G_original, title=None, node_color=labels_orcml[reverse_indices][G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/orcml_clusters_concentric_circles.png', dpi=1000)

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

    return_dict = get_pruned_unpruned_graph(hyperboloid_data, exp_params)
    G_original, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['preserved_edges'],
    )
    print(f'Hyperboloids: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcml = spectral_embedding(A_orcml, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcml = k_means(emb_orcml, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcml = metrics.adjusted_rand_score(cluster[G_orcml.nodes()], labels_orcml)

    print(f'Hyperboloids unpruned ARI: {rand_index_original}')
    print(f'Hyperboloids ORCML ARI: {rand_index_orcml}')

    # map labels to css code for red (#FF5733) and purple (#7A33FF)
    labels_original = np.array(['#FF5733' if i == 0 else '#7A33FF' for i in labels_original])
    labels_orcml = np.array(['#FF5733' if i == 0 else '#7A33FF' for i in labels_orcml])

    fig = plot_graph_3D(hyperboloid_data, G_original, title=None, node_color=labels_original, node_size=3, edge_width=0.5)
    fig.write_image(f'{save_dir}/original_hyperboloids_clusters.png', width=1200, height=1200, scale=10) 
    
    reverse_indices = np.array([np.where(np.array(list(G_orcml)) == i)[0][0] for i in range(len(G_orcml.nodes()))])
    fig = plot_graph_3D(hyperboloid_data, G_original, title=None, node_color=labels_orcml[reverse_indices], node_size=3, edge_width=0.5)
    fig.write_image(f'{save_dir}/orcml_hyperboloids_clusters.png', width=1200, height=1200, scale=10) 
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

    return_dict = get_pruned_unpruned_graph(moons_data, exp_params)
    G_original, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['preserved_edges'],
    )
    print(f'Moons: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcml = spectral_embedding(A_orcml, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcml = k_means(emb_orcml, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcml = metrics.adjusted_rand_score(cluster[G_orcml.nodes()], labels_orcml)
    
    print(f'Moons unpruned ARI: {rand_index_original}')
    print(f'Moons ORCML ARI: {rand_index_orcml}')

    plot_graph_2D(moons_data, G_original, title=None, node_color=labels_original[G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/original_clusters_moons.png', dpi=1000)

    reverse_indices = np.array([np.where(np.array(list(G_orcml)) == i)[0][0] for i in range(len(G_orcml.nodes()))])
    plot_graph_2D(moons_data, G_original, title=None, node_color=labels_orcml[reverse_indices][G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/orcml_clusters_moons.png', dpi=1000)      
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

    return_dict = get_pruned_unpruned_graph(quadratics_data, exp_params)
    G_original, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']

    labels = get_edge_labels(
        G_original,
        cluster=cluster,
    )
    percent_good_removed, percent_bad_removed = compute_metrics(
        labels,
        return_dict['preserved_edges'],
    )
    print(f'Quadratics: {percent_good_removed*100} percent good edges removed, {percent_bad_removed*100} percent bad edges removed')

    emb_original = spectral_embedding(A_original, n_components=1)
    emb_orcml = spectral_embedding(A_orcml, n_components=1)

    labels_original = k_means(emb_original, 2)
    labels_orcml = k_means(emb_orcml, 2)

    rand_index_original = metrics.adjusted_rand_score(cluster, labels_original)
    rand_index_orcml = metrics.adjusted_rand_score(cluster[G_orcml.nodes()], labels_orcml)

    print(f'Quadratics unpruned ARI: {rand_index_original}')
    print(f'Quadratics ORCML ARI: {rand_index_orcml}')

    plot_graph_2D(quadratics_data, G_original, title=None, node_color=labels_original[G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/original_clusters_quadratics.png', dpi=1000)

    reverse_indices = np.array([np.where(np.array(list(G_orcml)) == i)[0][0] for i in range(len(G_orcml.nodes()))])
    plot_graph_2D(quadratics_data, G_original, title=None, node_color=labels_orcml[reverse_indices][G_original.nodes()], node_size=2, edge_width=0.25)
    plt.savefig(f'{save_dir}/orcml_clusters_quadratics.png', dpi=1000)
    ############################## Quadratics ##############################