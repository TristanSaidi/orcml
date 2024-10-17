import os
from data.data import *
from src.experiments.embeddings import *
from src.orcmanl import *
from src.utils.plotting import *
from src.utils.eval_utils import *
from src.utils.graph_utils import *
from src.experiments.utils.exp_utils import *
from src.experiments.baselines import *
from src.scalar_curvature import *
import skdim as skd
import datetime


if __name__ == '__main__':
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'./outputs/official_experiments/{experiment_name}/estimators/swiss_roll'
    os.makedirs(save_dir, exist_ok=True)

    # Swiss Roll: SCE
    print('Estimating scalar curvature for Swiss Roll...')
    n_points = 4000
    noise = 6.25
    noise_thresh = 2.25

    exp_params = {
        'dataset': '3D_swiss_roll',
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }

    return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, dim=3)
    swiss_roll_data, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']

    return_dict = prune_helper(swiss_roll_data, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    Rdist_original = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_matrix(A_original), directed=False)
    assert np.allclose(Rdist_original, Rdist_original.T), "Distance matrix is not symmetric"

    Rdist_orcmanl = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_matrix(A_orcmanl), directed=False)
    assert np.allclose(Rdist_orcmanl, Rdist_orcmanl.T), "Distance matrix is not symmetric"

    sce_original = scalar_curvature_est(n=2, Rdist=Rdist_original)
    Ss_original = np.array(sce_original.estimate(rmax=50))

    sce_orcmanl = scalar_curvature_est(n=2, Rdist=Rdist_orcmanl)
    Ss_orcmanl = np.array(sce_orcmanl.estimate(rmax=50))

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0, z=0),
        eye=dict(x=0.1, y=1.8, z=0.0)
    )
    fig = plot_graph_3D(swiss_roll_data, G_original, title=None, node_color=Ss_original, cmax=0.08, cmin=0.00, camera=camera, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None)
    fig.write_image(f'{save_dir}/swiss_roll_original_sce.png', width=1200, height=1200, scale=10)

    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])
    fig = plot_graph_3D(swiss_roll_data, G_original, title=None, node_color=Ss_orcmanl[reverse_indices], cmax=0.08, cmin=0.00, camera=camera, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None)
    fig.write_image(f'{save_dir}/swiss_roll_orcmanl_sce.png', width=1200, height=1200, scale=10)

    base_truth = np.ones(len(swiss_roll_data)) * 0
    mse_original = np.mean((base_truth - Ss_original)**2)
    mse_orcmanl = np.mean((base_truth - Ss_orcmanl[reverse_indices])**2)

    print(f'MSE original: {mse_original}')
    print(f'MSE orcmanl: {mse_orcmanl}')

    # Swiss Roll: MLE i.d. estimation
    print('\n\nEstimating intrinsic dimension for Swiss Roll...')

    n_nbrs = 200
    ide = skd.id.MLE(n=3, neighborhood_based=True)

    sorted_indices_original = np.argsort(Rdist_original, axis=1)[:, 1:n_nbrs+1]
    sorted_distances_original = np.sort(Rdist_original, axis=1)[:, 1:n_nbrs+1]

    precomputed_knn_arrays = (sorted_distances_original, sorted_indices_original)

    pw_original, pw_smooth_original = ide.fit_transform_pw(swiss_roll_data, precomputed_knn_arrays=precomputed_knn_arrays, n_neighbors=n_nbrs, smooth=True)
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0, z=0),
        eye=dict(x=0.1, y=1.8, z=0.0)
    )
    fig = plot_graph_3D(swiss_roll_data, G_original, node_color=pw_smooth_original, cmax=3, cmin=2, title=None, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None, camera=camera)
    fig.write_image(f'{save_dir}/swiss_roll_original_id_mle.png', width=1200, height=1200, scale=10)
    # fig.show()

    ide = skd.id.MLE(n=3, neighborhood_based=True)
    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])

    sorted_indices_orcmanl = np.argsort(Rdist_orcmanl[reverse_indices][:, reverse_indices], axis=1)[:, 1:n_nbrs+1]
    sorted_distances_orcmanl = np.sort(Rdist_orcmanl[reverse_indices][:, reverse_indices], axis=1)[:, 1:n_nbrs+1]

    precomputed_knn_arrays = (sorted_distances_orcmanl, sorted_indices_orcmanl)

    pw_orcmanl, pw_smooth_orcmanl = ide.fit_transform_pw(swiss_roll_data, precomputed_knn_arrays=precomputed_knn_arrays, n_neighbors=n_nbrs, smooth=True)
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0, z=0),
        eye=dict(x=0.1, y=1.8, z=0.0)
    )
    fig = plot_graph_3D(swiss_roll_data, G_original, node_color=pw_smooth_orcmanl, cmax=3, cmin=2, title=None, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None, camera=camera)
    fig.write_image(f'{save_dir}/swiss_roll_orcmanl_id_mle.png', width=1200, height=1200, scale=10)
    # fig.show()

    # compute mse between base truth (2) and estimated intrinsic dimension
    base_truth = np.ones(len(swiss_roll_data)) * 2
    mse_original = np.mean((base_truth - pw_smooth_original)**2)
    mse_orcmanl = np.mean((base_truth - pw_smooth_orcmanl)**2)

    print(f'MSE original: {mse_original}')
    print(f'MSE orcmanl: {mse_orcmanl}')

    # adjacent spheres: SCE
    print('\n\nEstimating scalar curvature for adjacent spheres...')
    save_dir = f'./outputs/official_experiments/{experiment_name}/estimators/spheres'
    os.makedirs(save_dir, exist_ok=True)

    return_dict = spheres(n_points=4000, noise=0.08, noise_thresh=0.11)
    spheres_data, cluster, spheres_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']

    exp_params = {
        'dataset': '3D_spheres',
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }

    return_dict = prune_helper(spheres_data, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    Rdist_original = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_matrix(A_original), directed=False)
    assert np.allclose(Rdist_original, Rdist_original.T), "Distance matrix is not symmetric"

    Rdist_orcmanl = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_matrix(A_orcmanl), directed=False)
    assert np.allclose(Rdist_orcmanl, Rdist_orcmanl.T), "Distance matrix is not symmetric"

    sce_original = scalar_curvature_est(n=2, Rdist=Rdist_original)
    Ss_original = np.array(sce_original.estimate(rmax=5))

    sce_orcmanl = scalar_curvature_est(n=2, Rdist=Rdist_orcmanl)
    Ss_orcmanl = np.array(sce_orcmanl.estimate(rmax=5))

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0, z=0),
        eye=dict(x=0.1, y=0.0, z=2.8)
    )

    fig = plot_graph_3D(spheres_data, G_original, title=None, node_color=Ss_original, cmax=2, cmin=-1.0, camera=camera, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None)
    fig.write_image(f'{save_dir}/spheres_original_sce.png', width=1200, height=1200, scale=10)

    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])
    fig = plot_graph_3D(spheres_data, G_original, title=None, node_color=Ss_orcmanl[reverse_indices], cmax=2, cmin=-1.0, camera=camera, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None)
    fig.write_image(f'{save_dir}/spheres_orcmanl_sce.png', width=1200, height=1200, scale=10)

    # mse between base truth (2) and estimated scalar curvature
    base_truth = np.ones(len(spheres_data)) * 2
    mse_original = np.mean((base_truth - Ss_original)**2)
    mse_orcmanl = np.mean((base_truth - Ss_orcmanl[reverse_indices])**2)

    print(f'MSE original: {mse_original}')
    print(f'MSE orcmanl: {mse_orcmanl}')


    ### Intrinsic dimension
    print('\n\nEstimating intrinsic dimension for adjacent spheres...')

    data_dict = spheres(n_points=4000, noise=0.08, noise_thresh=0.11)
    spheres_data = data_dict['data']

    exp_params = {
        'dataset': '3D_spheres',
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0, z=0),
        eye=dict(x=0.1, y=0.0, z=2.8)
    )

    return_dict = prune_helper(spheres_data, exp_params)
    G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

    # get edge labels
    edge_labels = get_edge_labels(
        G_original,
        cluster=data_dict['cluster'],
    )
    percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, return_dict['non_shortcut_edges'])
    print(f'Percent good edges removed: {percent_good_removed}')
    print(f'Percent bad edges removed: {percent_bad_removed}')

    Rdist_original = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_matrix(A_original), directed=False)
    assert np.allclose(Rdist_original, Rdist_original.T), "Distance matrix is not symmetric"

    Rdist_orcmanl = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_matrix(A_orcmanl), directed=False)
    assert np.allclose(Rdist_orcmanl, Rdist_orcmanl.T), "Distance matrix is not symmetric"

    # mle id
    n_nbrs = 200
    ide = skd.id.MLE(n=3, neighborhood_based=True)

    sorted_indices_original = np.argsort(Rdist_original, axis=1)[:, 1:n_nbrs+1]
    sorted_distances_original = np.sort(Rdist_original, axis=1)[:, 1:n_nbrs+1]

    precomputed_knn_arrays = (sorted_distances_original, sorted_indices_original)

    pw_original, pw_smooth_original = ide.fit_transform_pw(spheres_data, precomputed_knn_arrays=precomputed_knn_arrays, n_neighbors=n_nbrs, smooth=True)
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0, z=0),
        eye=dict(x=0.1, y=0.0, z=2.8)
    )

    fig = plot_graph_3D(spheres_data, G_original, node_color=pw_smooth_original, cmax=2.2, cmin=1.9, title=None, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None, camera=camera)
    fig.write_image(f'{save_dir}/spheres_original_id_mle.png', width=1200, height=1200, scale=10)

    # mle id
    n_nbrs = 200

    ide = skd.id.MLE(n=3, neighborhood_based=True)
    reverse_indices = np.array([np.where(np.array(list(G_orcmanl)) == i)[0][0] for i in range(len(G_orcmanl.nodes()))])

    sorted_indices_orcmanl = np.argsort(Rdist_orcmanl[reverse_indices][:, reverse_indices], axis=1)[:, 1:n_nbrs+1]
    sorted_distances_orcmanl = np.sort(Rdist_orcmanl[reverse_indices][:, reverse_indices], axis=1)[:, 1:n_nbrs+1]

    precomputed_knn_arrays = (sorted_distances_orcmanl, sorted_indices_orcmanl)

    pw_orcmanl, pw_smooth_orcmanl = ide.fit_transform_pw(spheres_data, precomputed_knn_arrays=precomputed_knn_arrays, n_neighbors=n_nbrs, smooth=True)

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0, z=0),
        eye=dict(x=0.1, y=0.0, z=2.8)
    )

    fig = plot_graph_3D(spheres_data, G_original, node_color=pw_smooth_orcmanl, cmax=2.2, cmin=1.9, title=None, node_size=6, edge_width=0.25, opacity=1.0, node_colorbar=True, node_colorbar_title=None, camera=camera)
    fig.write_image(f'{save_dir}/spheres_orcmanl_id_mle.png', width=1200, height=1200, scale=10)

    # compute mse between base truth (2) and estimated intrinsic dimension
    base_truth = np.ones(len(spheres_data)) * 2
    mse_original = np.mean((base_truth - pw_smooth_original)**2)
    mse_orcmanl = np.mean((base_truth - pw_smooth_orcmanl)**2)

    print(f'MSE original: {mse_original}')
    print(f'MSE orcmanl: {mse_orcmanl}')