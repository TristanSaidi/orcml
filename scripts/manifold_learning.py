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

if __name__ == '__main__':
    np.random.seed(20)
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'./outputs/official_experiments/{experiment_name}/manifold_learning/3D_swiss_roll'
    os.makedirs(save_dir, exist_ok=True)
    

    # UMAP separately
    n_points = 5000
    noise = 1.15
    noise_thresh = 3.6

    exp_params = {
        'dataset': '3D_swiss_roll',
        'mode': 'nbrs',
        'n_neighbors': 20,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.75
    }

    dataset_info = {
        'name': '3D_swiss_roll',
        'n_points': n_points,
        'noise': noise,
        'noise_thresh': noise_thresh
    }
    print('Running UMAP')
    for hole in [True, False]:
        return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, dim=3, hole=hole)
        swiss_roll_data, color, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['color'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return_dict = prune_helper(swiss_roll_data, exp_params)
        G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

        Y_umap_original = UMAP(A_original, n_neighbors=exp_params['n_neighbors'], n_components=2)
        plot_emb(Y_umap_original, color, title=None)
        plt.savefig(f'{save_dir}/umap_original_hole_{hole}.png')
        plt.close()

        Y_umap_orcmanl = UMAP(A_orcmanl, n_neighbors=exp_params['n_neighbors'], n_components=2)
        plot_emb(Y_umap_orcmanl, color[list(G_orcmanl.nodes())], title=None)
        plt.savefig(f'{save_dir}/umap_orcmanl_hole_{hole}.png')
        plt.close()


    n_points = 4000
    noise = 6.2
    noise_thresh = 2.2

    exp_params = {
        'dataset': '3D_swiss_roll',
        'mode': 'nbrs',
        'n_neighbors': 25,
        'epsilon': None,
        'lda': 0.01,
        'delta': 0.8
    }

    dataset_info = {
        'name': '3D_swiss_roll',
        'n_points': n_points,
        'noise': noise,
        'noise_thresh': noise_thresh
    }

    for hole in [False, True]:
        print(f'\nRunning manifold learning experiments for the Swiss {"Roll" if not hole else "Hole"}\n')
        return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, dim=3, hole=hole)
        swiss_roll_data, color, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['color'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
        return_dict = prune_helper(swiss_roll_data, exp_params)
        G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

        # plot the data
        # 3D plot with matplotlib and no axes and rotated
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(swiss_roll_data[:, 0], swiss_roll_data[:, 1], swiss_roll_data[:, 2], c=color, cmap=plt.cm.Spectral, s=10, alpha=1)
        ax.set_axis_off()
        ax.view_init(00, 235)
        # move view closer to the center
        ax.dist = 8
        # highest resolution
        plt.savefig(f'{save_dir}/original_data_hole_{hole}.png', dpi=1000)
        plt.close()

        print('Running LLE')
        # might need to manually pick the eigenvectors because of the REP
        Y_lle_original = lle(A_original, swiss_roll_data, n_neighbors=exp_params['n_neighbors'], n_components=3)
        
        plot_emb(Y_lle_original[:, [0,1]], color, title=None)
        plt.savefig(f'{save_dir}/lle_original_hole_{hole}_evecs_0_1.png')
        plt.close()

        plot_emb(Y_lle_original[:, [0,2]], color, title=None)
        plt.savefig(f'{save_dir}/lle_original_hole_{hole}_evecs_0_2.png')
        plt.close()

        plot_emb(Y_lle_original[:, [1,2]], color, title=None)
        plt.savefig(f'{save_dir}/lle_original_hole_{hole}_evecs_1_2.png')
        plt.close()

        # might need to manually pick the eigenvectors because of the REP
        Y_lle_orcmanl = lle(A_orcmanl, swiss_roll_data[list(G_orcmanl)], n_neighbors=exp_params['n_neighbors'], n_components=3)
        
        plot_emb(Y_lle_orcmanl[:, [0,1]], color[list(G_orcmanl)], title=None)
        plt.savefig(f'{save_dir}/lle_orcmanl_hole_{hole}_evecs_0_1.png')
        plt.close()

        plot_emb(Y_lle_orcmanl[:, [0,2]], color[list(G_orcmanl)], title=None)
        plt.savefig(f'{save_dir}/lle_orcmanl_hole_{hole}_evecs_0_2.png')
        plt.close()

        plot_emb(Y_lle_orcmanl[:, [1, 2]], color[list(G_orcmanl)], title=None)
        plt.savefig(f'{save_dir}/lle_orcmanl_hole_{hole}_evecs_1_2.png')
        plt.close()

        print('Running Isomap')
        Y_isomap_original = isomap(A_original, n_components=2)
        plot_emb(Y_isomap_original, color, title=None)
        plt.savefig(f'{save_dir}/isomap_original_hole_{hole}.png')
        plt.close()

        Y_isomap_orcmanl = isomap(A_orcmanl, n_components=2)
        plot_emb(Y_isomap_orcmanl, color[list(G_orcmanl)], title=None)
        plt.savefig(f'{save_dir}/isomap_orcmanl_hole_{hole}.png')
        plt.close()

        print('Running Spectral Embedding')
        # might need to manually pick the eigenvectors because of the REP
        Y_spectral_original = spectral_embedding(A_original, n_components=3)

        plot_emb(Y_spectral_original[:, [0,1]], color, title=None)
        plt.savefig(f'{save_dir}/spectral_original_hole_{hole}_evecs_0_1.png')
        plt.close()

        plot_emb(Y_spectral_original[:, [0,2]], color, title=None)
        plt.savefig(f'{save_dir}/spectral_original_hole_{hole}_evecs_0_2.png')
        plt.close()

        plot_emb(Y_spectral_original[:, [1,2]], color, title=None)
        plt.savefig(f'{save_dir}/spectral_original_hole_{hole}_evecs_1_2.png')
        plt.close()

        # might need to manually pick the eigenvectors because of the REP
        Y_spectral_orcmanl = spectral_embedding(A_orcmanl, n_components=3)

        plot_emb(Y_spectral_orcmanl[:, [0,1]], color[list(G_orcmanl)], title=None)
        plt.savefig(f'{save_dir}/spectral_orcmanl_hole_{hole}_evecs_0_1.png')
        plt.close()

        plot_emb(Y_spectral_orcmanl[:, [0,2]], color[list(G_orcmanl)], title=None)
        plt.savefig(f'{save_dir}/spectral_orcmanl_hole_{hole}_evecs_0_2.png')
        plt.close()

        plot_emb(Y_spectral_orcmanl[:, [1,2]], color[list(G_orcmanl)], title=None)
        plt.savefig(f'{save_dir}/spectral_orcmanl_hole_{hole}_evecs_1_2.png')
        plt.close()


    # run 5 t-SNE samples
    for trial in range(5):
        for hole in [False, True]:
            print('Running t-SNE')
            return_dict = swiss_roll(n_points=n_points, noise=noise, noise_thresh=noise_thresh, supersample=True, dim=3, hole=hole)
            swiss_roll_data, color, cluster, swiss_roll_supersample, subsample_indices = return_dict['data'], return_dict['color'], return_dict['cluster'], return_dict['data_supersample'], return_dict['subsample_indices']
            return_dict = prune_helper(swiss_roll_data, exp_params)
            G_original, A_original, G_orcmanl, A_orcmanl = return_dict['G_original'], return_dict['A_original'], return_dict['G_orcmanl'], return_dict['A_orcmanl']

            Y_tsne_original = tsne(A_original, n_components=2)
            plot_emb(Y_tsne_original, color, title=None)
            plt.savefig(f'{save_dir}/tsne_original_hole_{hole}_trial_{trial}.png')
            plt.close()

            Y_tsne_orcmanl = tsne(A_orcmanl, n_components=2)
            plot_emb(Y_tsne_orcmanl, color[list(G_orcmanl)], title=None)
            plt.savefig(f'{save_dir}/tsne_orcmanl_hole_{hole}_trial_{trial}.png')
            plt.close()
