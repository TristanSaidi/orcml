import os
from src.data import *
from src.embeddings import *
from src.orcml import *
from src.plotting import *
from src.eval_utils import *
from src.baselines import *
import json

class OneDimPruningExperiment:

    def __init__(self):
        self.map = {
            'concentric_circles': self.get_concentric_circles,
            'swiss_roll': self.get_swiss_roll,
            'moons': self.get_moons,
            's_curve': self.get_s_curve,
            'cassini': self.get_cassini,
            'mixture_of_gaussians': self.get_mixture_of_gaussians
        }

        self.param_map = {
            'concentric_circles': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':0.5,
                'density_thresh':0.125,
                'distance_thresh':0.15,
                'random_p':0.025,
                'edge_label_est_scale': 10 
            },
            'swiss_roll': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':5,
                'density_thresh':0.00125,
                'distance_thresh':1.25,
                'random_p':0.025, 
                'edge_label_est_scale': 10 
            },
            'moons': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':0.5,
                'density_thresh':0.175,
                'distance_thresh':0.15,
                'random_p':0.025,
                'edge_label_est_scale': 10 
            },
            's_curve': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':0.5,
                'density_thresh':0.05,
                'distance_thresh':0.15,
                'random_p':0.025,
                'edge_label_est_scale': 12 
            },
            'cassini': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':0.5,
                'density_thresh':0.2,
                'distance_thresh':0.09,
                'random_p':0.025,
                'edge_label_est_scale': 20 
            },
            'mixture_of_gaussians': {
                'k': 20,
                'lda': 0.01,
                'delta': 0.8,
                'orc_delta':0.8,
                'n_bisection':10,
                'mst_thresh':0.3,
                'density_thresh':0.15,
                'distance_thresh':0.1,
                'random_p':0.025,
                'edge_label_est_scale': 10
            }
        }

    def run_experiment_knn(
            self, 
            dataset, 
            n_iter=10, 
            experiment_name=None,
            seed=42
        ):

        np.random.seed(seed)

        if experiment_name is not None:
            save_dir = f'./outputs/official_experiments/{experiment_name}/knn/{dataset}'
            os.makedirs(save_dir, exist_ok=True)
        
        exp_params = self.param_map[dataset]

        # orcml
        percent_good_total_orcml = [] # percent of good edges removed by orcml
        percent_bad_total_orcml = [] # percent of bad edges removed by orcml
        
        # orc only
        percent_good_total_orc = [] # percent of good edges removed by orc
        percent_bad_total_orc = [] # percent of bad edges removed by orc

        # bisection prune
        percent_good_total_bisection = [] # percent of good edges removed by bisection prune
        percent_bad_total_bisection = [] # percent of bad edges removed by bisection prune

        # mst prune
        percent_good_total_mst = [] # percent of good edges removed by mst prune
        percent_bad_total_mst = [] # percent of bad edges removed by mst prune

        # density prune
        percent_good_total_density = [] # percent of good edges removed by density prune
        percent_bad_total_density = [] # percent of bad edges removed by density prune

        # distance prune
        percent_good_total_distance = [] # percent of good edges removed by distance prune
        percent_bad_total_distance = [] # percent of bad edges removed by distance prune

        # random prune
        percent_good_total_random = [] # percent of good edges removed by random prune
        percent_bad_total_random = [] # percent of bad edges removed by random prune

        i = 0
        while i < n_iter:
            print(f'\nIteration {i+1}/{n_iter}')
            data, cluster, data_supersample, subsample_indices, dataset_info = self.map[dataset]()

            # knn
            G, A = make_prox_graph(data, mode='nbrs', n_neighbors=exp_params['k'])

            edge_labels = get_edge_labels(
                G, 
                cluster=cluster, 
                data_supersample_dict={
                    'data_supersample': data_supersample,
                    'subsample_indices': subsample_indices
                },
                scale=exp_params['edge_label_est_scale']
            )

            # if no bad edges formed, skip this iteration
            if np.sum(np.array(edge_labels) == 0) == 0:
                print('No bad edges formed, resampling...')
                continue

            # orc
            return_dict = graph_orc(G, weight='unweighted_dist')
            G_orc = return_dict['G']
            # prune with orcml
            print('Pruning with orcml...')
            pruned_orcml = prune_orcml(G_orc, data, eps=None, lda=exp_params['lda'], delta=exp_params['delta'])
            print('Pruning with orc...')
            pruned_orc = prune_orc(G_orc, exp_params['orc_delta'], data)
            print('Pruning with bisection...')
            pruned_bisection = prune_bisection(G_orc, data, n=exp_params['n_bisection'])
            print('Pruning with mst...')
            pruned_mst = prune_mst(G_orc, data, exp_params['mst_thresh'])
            print('Pruning with density...')
            pruned_density = prune_density(G_orc, data, exp_params['density_thresh'])
            print('Pruning with distance...')
            pruned_distance = prune_distance(G_orc, data, exp_params['distance_thresh'])
            print('Pruning with random...')
            pruned_random = prune_random(G_orc, data, exp_params['random_p'])

            # get number of good edges removed, number of bad edges removed for each meth
            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_orcml['preserved_edges'])
            percent_good_total_orcml.append(percent_good_removed)
            percent_bad_total_orcml.append(percent_bad_removed)
            print(f'orcml: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_orc['preserved_edges'])
            percent_good_total_orc.append(percent_good_removed)
            percent_bad_total_orc.append(percent_bad_removed)
            print(f'orc: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_bisection['preserved_edges'])
            percent_good_total_bisection.append(percent_good_removed)
            percent_bad_total_bisection.append(percent_bad_removed)
            print(f'bisection: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_mst['preserved_edges'])
            percent_good_total_mst.append(percent_good_removed)
            percent_bad_total_mst.append(percent_bad_removed)
            print(f'mst: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_density['preserved_edges'])
            percent_good_total_density.append(percent_good_removed)
            percent_bad_total_density.append(percent_bad_removed)
            print(f'density: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_distance['preserved_edges'])
            percent_good_total_distance.append(percent_good_removed)
            percent_bad_total_distance.append(percent_bad_removed)
            print(f'distance: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_random['preserved_edges'])
            percent_good_total_random.append(percent_good_removed)
            percent_bad_total_random.append(percent_bad_removed)
            print(f'random: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')
            
            plot_graph_2D(data, G_orc, None, edge_color=(0.2, 0.2, 0.2), node_size=0.0, edge_width=0.2, colorbar=False)
            plt.savefig(f'{save_dir}/original_data_iter_{i}.png')
            plt.close()
            plot_graph_2D(data, pruned_orcml['G_pruned'], None, edge_color=(0.2, 0.2, 0.2), node_size=0.0, edge_width=0.2, colorbar=False)
            plt.savefig(f'{save_dir}/orcml_pruned_data_iter_{i}.png')
            plt.close()
            i += 1
            
        # plot results for last run
        if experiment_name is not None:
            # save metrics
            orcml_metrics = {
                'orcml_lda': exp_params['lda'],
                'orcml_delta': exp_params['delta'],
                'percent_good_total_mean': np.mean(percent_good_total_orcml),
                'percent_good_total_std': np.std(percent_good_total_orcml),
                'percent_bad_total_mean': np.mean(percent_bad_total_orcml),
                'percent_bad_total_std': np.std(percent_bad_total_orcml),
            }

            orc_metrics = {
                'orc_delta': exp_params['orc_delta'],
                'percent_good_total_mean': np.mean(percent_good_total_orc),
                'percent_good_total_std': np.std(percent_good_total_orc),
                'percent_bad_total_mean': np.mean(percent_bad_total_orc),
                'percent_bad_total_std': np.std(percent_bad_total_orc),
            }

            bisection_metrics = {
                'n_bisection': exp_params['n_bisection'],
                'percent_good_total_mean': np.mean(percent_good_total_bisection),
                'percent_good_total_std': np.std(percent_good_total_bisection),
                'percent_bad_total_mean': np.mean(percent_bad_total_bisection),
                'percent_bad_total_std': np.std(percent_bad_total_bisection),
            }

            mst_metrics = {
                'mst_thresh': exp_params['mst_thresh'],
                'percent_good_total_mean': np.mean(percent_good_total_mst),
                'percent_good_total_std': np.std(percent_good_total_mst),
                'percent_bad_total_mean': np.mean(percent_bad_total_mst),
                'percent_bad_total_std': np.std(percent_bad_total_mst),
            }

            density_metrics = {
                'density_thresh': exp_params['density_thresh'],
                'percent_good_total_mean': np.mean(percent_good_total_density),
                'percent_good_total_std': np.std(percent_good_total_density),
                'percent_bad_total_mean': np.mean(percent_bad_total_density),
                'percent_bad_total_std': np.std(percent_bad_total_density),
            }

            distance_metrics = {
                'distance_thresh': exp_params['distance_thresh'],
                'percent_good_total_mean': np.mean(percent_good_total_distance),
                'percent_good_total_std': np.std(percent_good_total_distance),
                'percent_bad_total_mean': np.mean(percent_bad_total_distance),
                'percent_bad_total_std': np.std(percent_bad_total_distance),
            }

            random_metrics = {
                'random_p': exp_params['random_p'],
                'percent_good_total_mean': np.mean(percent_good_total_random),
                'percent_good_total_std': np.std(percent_good_total_random),
                'percent_bad_total_mean': np.mean(percent_bad_total_random),
                'percent_bad_total_std': np.std(percent_bad_total_random),
            }

            metrics = {
                'num_trials': n_iter,
                'seed': seed,
                'k-NN, k': exp_params['k'],
                'orcml': orcml_metrics,
                'orc': orc_metrics,
                'bisection': bisection_metrics,
                'mst': mst_metrics,
                'density': density_metrics,
                'distance': distance_metrics,
                'random': random_metrics,
                'dataset_info': dataset_info
            }
            with open(f'{save_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
        
        return

    def get_concentric_circles(self, n_points=4000, factor=0.385, noise=0.09, noise_thresh=0.275):
        dataset_info = {
            'name': 'concentric_circles',
            'n_points': n_points,
            'factor': factor,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = concentric_circles(n_points=n_points, factor=factor, noise=noise, noise_thresh=noise_thresh, supersample=True)
        circles, cluster, circles_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return circles, cluster, circles_supersample, subsample_indices, dataset_info
    
    def get_swiss_roll(self, n_points=4000, noise=1, noise_thresh=2.7):
        dataset_info = {
            'name': 'swiss_roll',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, dim=2)
        swiss_roll_data, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return swiss_roll_data, cluster, swiss_roll_supersample, subsample_indices, dataset_info
    
    def get_moons(self, n_points=4000, noise=0.2, noise_thresh=0.1925):
        dataset_info = {
            'name': 'moons',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = moons(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
        moons_data, cluster, moons_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return moons_data, cluster, moons_supersample, subsample_indices, dataset_info
    
    def get_s_curve(self, n_points=4000, noise=0.28, noise_thresh=0.52):
        dataset_info = {
            'name': 's_curve',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = s_curve(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
        s_curve_data, cluster, s_curve_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return s_curve_data, cluster, s_curve_supersample, subsample_indices, dataset_info
    
    def get_cassini(self, n_points=4000, noise=0.05, noise_thresh=0.135):
        dataset_info = {
            'name': 'cassini',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = cassini(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
        cassini_data, cluster, cassini_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return cassini_data, cluster, cassini_supersample, subsample_indices, dataset_info

    def get_mixture_of_gaussians(self, n_points=4000, noise=0.175, noise_thresh=0.45):
        dataset_info = {
            'name': 'mixture_of_gaussians',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = mixture_of_gaussians(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
        mog_data, cluster, mog_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return mog_data, cluster, mog_supersample, subsample_indices, dataset_info
    


class TwoDimPruningExperiment:

    def __init__(self):
        self.map = {
            'torii': self.get_torii,
            'hyperboloids': self.get_hyperboloids,
            'parab_and_hyp': self.get_parab_and_hyp,
            'double_paraboloid': self.get_double_paraboloid,
            '3D_swiss_roll': self.get_3D_swiss_roll
        }

        self.param_map = {
            'torii': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':1.5,
                'density_thresh':0.0007,
                'distance_thresh':1.0,
                'random_p':0.025,
                'edge_label_est_scale': 10 
            },
            'hyperboloids': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':1.5,
                'density_thresh':0.015,
                'distance_thresh':0.4,
                'random_p':0.025,
                'edge_label_est_scale': 10 
            },
            'parab_and_hyp': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':0.75,
                'density_thresh':0.015,
                'distance_thresh':0.4,
                'random_p':0.025,
                'edge_label_est_scale': 10 
            },
            'double_paraboloid': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':1.0,
                'density_thresh':0.01,
                'distance_thresh':0.5,
                'random_p':0.025,
                'edge_label_est_scale': 10 
            },
            '3D_swiss_roll': {
                'k': 20, 
                'lda': 0.01, 
                'delta': 0.8, 
                'orc_delta':0.8, 
                'n_bisection':10, 
                'mst_thresh':10,
                'density_thresh':0.00005,
                'distance_thresh':3.0,
                'random_p':0.025,
                'edge_label_est_scale': 10 
            }
            
        }

    def run_experiment_knn(
            self, 
            dataset, 
            n_iter=10, 
            experiment_name=None,
            seed=42
        ):

        np.random.seed(seed)

        if experiment_name is not None:
            save_dir = f'./outputs/official_experiments/{experiment_name}/knn/{dataset}'
            os.makedirs(save_dir, exist_ok=True)
        
        exp_params = self.param_map[dataset]

        # orcml
        percent_good_total_orcml = [] # percent of good edges removed by orcml
        percent_bad_total_orcml = [] # percent of bad edges removed by orcml
        
        # orc only
        percent_good_total_orc = [] # percent of good edges removed by orc
        percent_bad_total_orc = [] # percent of bad edges removed by orc

        # bisection prune
        percent_good_total_bisection = [] # percent of good edges removed by bisection prune
        percent_bad_total_bisection = [] # percent of bad edges removed by bisection prune

        # mst prune
        percent_good_total_mst = [] # percent of good edges removed by mst prune
        percent_bad_total_mst = [] # percent of bad edges removed by mst prune

        # density prune
        percent_good_total_density = [] # percent of good edges removed by density prune
        percent_bad_total_density = [] # percent of bad edges removed by density prune

        # distance prune
        percent_good_total_distance = [] # percent of good edges removed by distance prune
        percent_bad_total_distance = [] # percent of bad edges removed by distance prune

        # random prune
        percent_good_total_random = [] # percent of good edges removed by random prune
        percent_bad_total_random = [] # percent of bad edges removed by random prune

        i = 0
        while i < n_iter:
            print(f'\nIteration {i+1}/{n_iter}')
            data, cluster, data_supersample, subsample_indices, dataset_info = self.map[dataset]()

            # knn
            G, A = make_prox_graph(data, mode='nbrs', n_neighbors=exp_params['k'])
            edge_labels = get_edge_labels(
                G, 
                cluster=cluster, 
                data_supersample_dict={
                    'data_supersample': data_supersample,
                    'subsample_indices': subsample_indices
                },
                scale=exp_params['edge_label_est_scale']
            )

            # if no bad edges formed, skip this iteration
            if np.sum(np.array(edge_labels) == 0) == 0:
                print('No bad edges formed, resampling...')
                continue

            # orc
            return_dict = graph_orc(G, weight='unweighted_dist')
            G_orc = return_dict['G']
            # prune with orcml
            print('Pruning with orcml...')
            pruned_orcml = prune_orcml(G_orc, data, eps=None, lda=exp_params['lda'], delta=exp_params['delta'])
            print('Pruning with orc...')
            pruned_orc = prune_orc(G_orc, exp_params['orc_delta'], data)
            print('Pruning with bisection...')
            pruned_bisection = prune_bisection(G_orc, data, n=exp_params['n_bisection'])
            print('Pruning with mst...')
            pruned_mst = prune_mst(G_orc, data, exp_params['mst_thresh'])
            print('Pruning with density...')
            pruned_density = prune_density(G_orc, data, exp_params['density_thresh'])
            print('Pruning with distance...')
            pruned_distance = prune_distance(G_orc, data, exp_params['distance_thresh'])
            print('Pruning with random...')
            pruned_random = prune_random(G_orc, data, exp_params['random_p'])

            # get number of good edges removed, number of bad edges removed for each meth
            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_orcml['preserved_edges'])
            percent_good_total_orcml.append(percent_good_removed)
            percent_bad_total_orcml.append(percent_bad_removed)
            print(f'orcml: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_orc['preserved_edges'])
            percent_good_total_orc.append(percent_good_removed)
            percent_bad_total_orc.append(percent_bad_removed)
            print(f'orc: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_bisection['preserved_edges'])
            percent_good_total_bisection.append(percent_good_removed)
            percent_bad_total_bisection.append(percent_bad_removed)
            print(f'bisection: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_mst['preserved_edges'])
            percent_good_total_mst.append(percent_good_removed)
            percent_bad_total_mst.append(percent_bad_removed)
            print(f'mst: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_density['preserved_edges'])
            percent_good_total_density.append(percent_good_removed)
            percent_bad_total_density.append(percent_bad_removed)
            print(f'density: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_distance['preserved_edges'])
            percent_good_total_distance.append(percent_good_removed)
            percent_bad_total_distance.append(percent_bad_removed)
            print(f'distance: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, pruned_random['preserved_edges'])
            percent_good_total_random.append(percent_good_removed)
            percent_bad_total_random.append(percent_bad_removed)
            print(f'random: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')
            i += 1
            
        # plot results for last run
        if experiment_name is not None:
            # save metrics
            orcml_metrics = {
                'orcml_lda': exp_params['lda'],
                'orcml_delta': exp_params['delta'],
                'percent_good_total_mean': np.mean(percent_good_total_orcml),
                'percent_good_total_std': np.std(percent_good_total_orcml),
                'percent_bad_total_mean': np.mean(percent_bad_total_orcml),
                'percent_bad_total_std': np.std(percent_bad_total_orcml),
            }

            orc_metrics = {
                'orc_delta': exp_params['orc_delta'],
                'percent_good_total_mean': np.mean(percent_good_total_orc),
                'percent_good_total_std': np.std(percent_good_total_orc),
                'percent_bad_total_mean': np.mean(percent_bad_total_orc),
                'percent_bad_total_std': np.std(percent_bad_total_orc),
            }

            bisection_metrics = {
                'n_bisection': exp_params['n_bisection'],
                'percent_good_total_mean': np.mean(percent_good_total_bisection),
                'percent_good_total_std': np.std(percent_good_total_bisection),
                'percent_bad_total_mean': np.mean(percent_bad_total_bisection),
                'percent_bad_total_std': np.std(percent_bad_total_bisection),
            }

            mst_metrics = {
                'mst_thresh': exp_params['mst_thresh'],
                'percent_good_total_mean': np.mean(percent_good_total_mst),
                'percent_good_total_std': np.std(percent_good_total_mst),
                'percent_bad_total_mean': np.mean(percent_bad_total_mst),
                'percent_bad_total_std': np.std(percent_bad_total_mst),
            }

            density_metrics = {
                'density_thresh': exp_params['density_thresh'],
                'percent_good_total_mean': np.mean(percent_good_total_density),
                'percent_good_total_std': np.std(percent_good_total_density),
                'percent_bad_total_mean': np.mean(percent_bad_total_density),
                'percent_bad_total_std': np.std(percent_bad_total_density),
            }

            distance_metrics = {
                'distance_thresh': exp_params['distance_thresh'],
                'percent_good_total_mean': np.mean(percent_good_total_distance),
                'percent_good_total_std': np.std(percent_good_total_distance),
                'percent_bad_total_mean': np.mean(percent_bad_total_distance),
                'percent_bad_total_std': np.std(percent_bad_total_distance),
            }

            random_metrics = {
                'random_p': exp_params['random_p'],
                'percent_good_total_mean': np.mean(percent_good_total_random),
                'percent_good_total_std': np.std(percent_good_total_random),
                'percent_bad_total_mean': np.mean(percent_bad_total_random),
                'percent_bad_total_std': np.std(percent_bad_total_random),
            }

            metrics = {
                'num_trials': n_iter,
                'seed': seed,
                'k-NN, k': exp_params['k'],
                'orcml': orcml_metrics,
                'orc': orc_metrics,
                'bisection': bisection_metrics,
                'mst': mst_metrics,
                'density': density_metrics,
                'distance': distance_metrics,
                'random': random_metrics,
                'dataset_info': dataset_info
            }
            with open(f'{save_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
        
        return

    def get_torii(self, n_points=4000, noise=0.4, noise_thresh=0.75):
        dataset_info = {
            'name': 'torii',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = torus(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, double=True)
        torus_data, cluster, torus_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return torus_data, cluster, torus_supersample, subsample_indices, dataset_info
    
    def get_hyperboloids(self, n_points=4000, noise=0.2, noise_thresh=0.25):
        dataset_info = {
            'name': 'hyperboloids',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = hyperboloid(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, double=True)
        hyperboloids_data, cluster, hyperboloids_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return hyperboloids_data, cluster, hyperboloids_supersample, subsample_indices, dataset_info
    
    def get_parab_and_hyp(self, n_points=4000, noise=0.4, noise_thresh=0.475):
        dataset_info = {
            'name': 'parab_and_hyp',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = parab_and_hyp(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, double=True)
        parab_and_hyp_data, cluster, parab_and_hyp_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return parab_and_hyp_data, cluster, parab_and_hyp_supersample, subsample_indices, dataset_info
    
    def get_double_paraboloid(self, n_points=4000, noise=0.6, noise_thresh=0.7):
        dataset_info = {
            'name': 'double_paraboloid',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = double_paraboloid(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
        double_paraboloid_data, cluster, double_paraboloid_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return double_paraboloid_data, cluster, double_paraboloid_supersample, subsample_indices, dataset_info
        
    def get_3D_swiss_roll(self, n_points=4000, noise=6.25, noise_thresh=2.25):
        dataset_info = {
            'name': '3D_swiss_roll',
            'n_points': n_points,
            'noise': noise
        }
        return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
        swiss_roll_data, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return swiss_roll_data, cluster, swiss_roll_supersample, subsample_indices, dataset_info

def get_pruned_unpruned_graph(data, exp_params):
    if exp_params['mode'] == 'nbrs':
        G, A = make_prox_graph(data, mode=exp_params['mode'], n_neighbors=exp_params['n_neighbors']) # unpruned k-nn graph
    else:
        G, A = make_prox_graph(data, mode=exp_params['mode'], epsilon=exp_params['epsilon'])
    return_dict = graph_orc(G, weight='unweighted_dist')
    pruned_orcml = prune_orcml(return_dict['G'], data, eps=exp_params['epsilon'], lda=exp_params['lda'], delta=exp_params['delta'])
    G_orcml = pruned_orcml['G_pruned']
    A_orcml = nx.adjacency_matrix(G_orcml).toarray()
    # symmetrize
    A = np.maximum(A, A.T)
    return {
        "G_original": G,
        "A_original": A,
        "G_orcml": G_orcml,
        "A_orcml": A_orcml
    }

class ManifoldLearningExperiment:
    
    def __init__(self, exp_params):
        self.exp_params = exp_params
        self.map = {
            '3D_swiss_roll': self.get_3D_swiss_roll,
            '3D_swiss_hole': self.get_3D_swiss_hole,
        }

    def run_experiment(
            self, 
            dataset, 
            experiment_name=None, 
            seed=0
        ):
        
        np.random.seed(seed)
        data, color, cluster, data_supersample, subsample_indices, dataset_info = self.map[dataset]()

        if experiment_name is not None:
            save_dir = f'./outputs/official_experiments/{experiment_name}/manifold_learning/{dataset}'
            os.makedirs(save_dir, exist_ok=True)

        return_dict = get_pruned_unpruned_graph(data, self.exp_params)
        G_original, A_original, G_orcml, A_orcml = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcml'], return_dict['A_orcml']
    
        # isomap
        Y_isomap_original = isomap(A_original, n_components=2)
        plot_emb(Y_isomap_original, color, title=None)
        plt.savefig(f'{save_dir}/isomap_original.png')
        plt.close()
        
        Y_isomap_orcml = isomap(A_orcml, n_components=2)
        plot_emb(Y_isomap_orcml, color[list(G_orcml)], title=None)
        plt.savefig(f'{save_dir}/isomap_orcml.png')
        plt.close()

        # spectral
        Y_spectral_original = spectral_embedding(A_original, n_components=2)
        plot_emb(Y_spectral_original, color, title=None)
        plt.savefig(f'{save_dir}/spectral_original.png')
        plt.close()

        Y_spectral_orcml = spectral_embedding(A_orcml, n_components=2)
        plot_emb(Y_spectral_orcml, color[list(G_orcml)], title=None)
        plt.savefig(f'{save_dir}/spectral_orcml.png')
        plt.close()

        # lle
        Y_lle_original = lle(A_original, data, n_neighbors= self.exp_params['n_neighbors'], n_components=2)
        plot_emb(Y_lle_original, color, title=None)
        plt.savefig(f'{save_dir}/lle_original.png')
        plt.close()

        Y_lle_orcml = lle(A_orcml, data[list(G_orcml)], n_neighbors= self.exp_params['n_neighbors'], n_components=2)
        plot_emb(Y_lle_orcml, color[list(G_orcml)], title=None)
        plt.savefig(f'{save_dir}/lle_orcml.png')
        plt.close()

        # umap
        Y_umap_original = UMAP(A_original, n_neighbors=self.exp_params['n_neighbors'], n_components=2)
        plot_emb(Y_umap_original, color, title=None)
        plt.savefig(f'{save_dir}/umap_original.png')
        plt.close()

        Y_umap_orcml = UMAP(A_orcml, n_neighbors=self.exp_params['n_neighbors'], n_components=2)
        plot_emb(Y_umap_orcml, color[list(G_orcml)], title=None)
        plt.savefig(f'{save_dir}/umap_orcml.png')
        plt.close()

        # tsne
        Y_tsne_original = tsne(A_original, n_components=2)
        plot_emb(Y_tsne_original, color, title=None)
        plt.savefig(f'{save_dir}/tsne_original.png')
        plt.close()

        Y_tsne_orcml = tsne(A_orcml, n_components=2)
        plot_emb(Y_tsne_orcml, color[list(G_orcml)], title=None)
        plt.savefig(f'{save_dir}/tsne_orcml.png')
        plt.close()

        info = {
            'seed': seed,
            'dataset_info': dataset_info,
            'exp_params': self.exp_params
        }
        with open(f'{save_dir}/info.json', 'w') as f:
            json.dump(info, f, indent=4)


    def get_3D_swiss_roll(self, n_points=2000, noise=1.2, noise_thresh=1.1):
        dataset_info = {
            'name': '3D_swiss_roll',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True)
        swiss_roll_data, color, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['color'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return swiss_roll_data, color, cluster, swiss_roll_supersample, subsample_indices, dataset_info
    

    def get_3D_swiss_hole(self, n_points=2000, noise=1.2, noise_thresh=1.1):
        dataset_info = {
            'name': '3D_swiss_hole',
            'n_points': n_points,
            'noise': noise,
            'noise_thresh': noise_thresh
        }
        return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, hole=True)
        swiss_roll_data, color, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['color'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return swiss_roll_data, color, cluster, swiss_roll_supersample, subsample_indices, dataset_info