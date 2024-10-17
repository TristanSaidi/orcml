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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20,
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

        # orcmanl
        percent_good_total_orcmanl = [] # percent of good edges removed by orcmanl
        percent_bad_total_orcmanl = [] # percent of bad edges removed by orcmanl
        
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

        i = 0
        while i < n_iter:
            print(f'\nIteration {i+1}/{n_iter}')
            data, cluster, data_supersample, subsample_indices, dataset_info = self.map[dataset]()

            # knn
            return_dict = prune_helper(data, exp_params=exp_params)
            G = return_dict['G_original']
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

            print('Pruning with orc...')
            pruned_orc = prune_orc(G, exp_params['orc_delta'], data)
            print('Pruning with bisection...')
            pruned_bisection = prune_bisection(G, data, n=exp_params['n_bisection'])
            print('Pruning with mst...')
            pruned_mst = prune_mst(G, data, exp_params['mst_thresh'])
            print('Pruning with density...')
            pruned_density = prune_density(G, data, exp_params['density_thresh'])
            print('Pruning with distance...')
            pruned_distance = prune_distance(G, data, exp_params['distance_thresh'])

            # get number of good edges removed, number of bad edges removed for each meth
            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, return_dict['non_shortcut_edges'])
            percent_good_total_orcmanl.append(percent_good_removed)
            percent_bad_total_orcmanl.append(percent_bad_removed)
            print(f'orcmanl: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

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
            i += 1
            
        # plot results for last run
        if experiment_name is not None:
            # save metrics
            orcmanl_metrics = {
                'orcmanl_lda': exp_params['lda'],
                'orcmanl_delta': exp_params['delta'],
                'percent_good_total_mean': np.mean(percent_good_total_orcmanl),
                'percent_good_total_std': np.std(percent_good_total_orcmanl),
                'percent_bad_total_mean': np.mean(percent_bad_total_orcmanl),
                'percent_bad_total_std': np.std(percent_bad_total_orcmanl),
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

            metrics = {
                'num_trials': n_iter,
                'seed': seed,
                'k-NN, k': exp_params['n_neighbors'],
                'orcmanl': orcmanl_metrics,
                'orc': orc_metrics,
                'bisection': bisection_metrics,
                'mst': mst_metrics,
                'density': density_metrics,
                'distance': distance_metrics,
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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
                'n_neighbors': 20, 
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

        # orcmanl
        percent_good_total_orcmanl = [] # percent of good edges removed by orcmanl
        percent_bad_total_orcmanl = [] # percent of bad edges removed by orcmanl
        
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

        i = 0
        while i < n_iter:
            print(f'\nIteration {i+1}/{n_iter}')
            data, cluster, data_supersample, subsample_indices, dataset_info = self.map[dataset]()

            # knn
            return_dict = prune_helper(data, exp_params=exp_params)
            G = return_dict['G_original']
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

            print('Pruning with orc...')
            pruned_orc = prune_orc(G, exp_params['orc_delta'], data)
            print('Pruning with bisection...')
            pruned_bisection = prune_bisection(G, data, n=exp_params['n_bisection'])
            print('Pruning with mst...')
            pruned_mst = prune_mst(G, data, exp_params['mst_thresh'])
            print('Pruning with density...')
            pruned_density = prune_density(G, data, exp_params['density_thresh'])
            print('Pruning with distance...')
            pruned_distance = prune_distance(G, data, exp_params['distance_thresh'])

            # get number of good edges removed, number of bad edges removed for each meth
            percent_good_removed, percent_bad_removed = compute_metrics(edge_labels, return_dict['non_shortcut_edges'])
            percent_good_total_orcmanl.append(percent_good_removed)
            percent_bad_total_orcmanl.append(percent_bad_removed)
            print(f'orcmanl: {100*percent_good_removed:.2f}% good edges removed, {100*percent_bad_removed:.2f}% bad edges removed')

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
            i += 1
            
        # plot results for last run
        if experiment_name is not None:
            # save metrics
            orcmanl_metrics = {
                'orcmanl_lda': exp_params['lda'],
                'orcmanl_delta': exp_params['delta'],
                'percent_good_total_mean': np.mean(percent_good_total_orcmanl),
                'percent_good_total_std': np.std(percent_good_total_orcmanl),
                'percent_bad_total_mean': np.mean(percent_bad_total_orcmanl),
                'percent_bad_total_std': np.std(percent_bad_total_orcmanl),
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

            metrics = {
                'num_trials': n_iter,
                'seed': seed,
                'k-NN, k': exp_params['n_neighbors'],
                'orcmanl': orcmanl_metrics,
                'orc': orc_metrics,
                'bisection': bisection_metrics,
                'mst': mst_metrics,
                'density': density_metrics,
                'distance': distance_metrics,
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


if __name__ == '__main__':
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    exp = OneDimPruningExperiment()
    exp.run_experiment_knn('concentric_circles', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('moons', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('s_curve', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('cassini', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('mixture_of_gaussians', n_iter=10, experiment_name=experiment_name)
    
    exp = TwoDimPruningExperiment()
    exp.run_experiment_knn('torii', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('hyperboloids', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('parab_and_hyp', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('double_paraboloid', n_iter=10, experiment_name=experiment_name)
    exp.run_experiment_knn('3D_swiss_roll', n_iter=10, experiment_name=experiment_name)