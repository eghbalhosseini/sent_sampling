
import pickle as pkl
from netrep.multiset import pairwise_distances, frechet_mean, pt_frechet_mean
from tqdm import tqdm
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
from sklearn.decomposition import PCA
import pandas as pd
import sys
from netrep.utils import align, pt_align
from datasets import load_dataset
import getpass
if getpass.getuser() == 'ehoseini':
    sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
    image_paths = '/om2/user/ehoseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/om2/user/ehoseini/MyData/neural_nlp_bench/activations/DeepJuice_DsParametricfMRI/'
    benchmark_path = '/om2/user/ehoseini/MyData/DeepJuice/nsd_data/'
    platonic_path='/rdma/vast-rdma/vast/evlab/ehoseini/MyData/shape_metric/'
else:
    sys.path.append('/Users/eghbalhosseini/MyCodes/DeepJuiceDev/')
    image_paths = '/Users/eghbalhosseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/Users/eghbalhosseini/MyData/DeepJuice/workspace/nsd/'
    benchmark_path = '/Users/eghbalhosseini/MyData/DeepJuice/nsd_data/'

import sys
import os
# Add the root directory of the repo to sys.path
sys.path.append(os.path.abspath('/om2/user/ehoseini/platonic-rep'))
from utils import to_feature_filename
from extract_features import extract_llm_features, extract_lvm_features
from sklearn.preprocessing import StandardScaler
import multiprocessing
import os
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
os.environ['OMP_NUM_THREADS'] = '4'
import matplotlib.pyplot as plt
# Check operating system
# Check operating system
import platform
if platform.system() == 'Darwin':  # Darwin is the system name for macOS
    # Check if MPS (Metal Performance Shaders) backend is available, for Apple Silicon Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS on supported Macs
    else:
        device = torch.device("cpu")  # Fallback to CPU if MPS is not available
    float_version=torch.float32
else:
    # For non-macOS, you can default to CPU or check for CUDA (NVIDIA GPU) availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    float_version=torch.float64


print(f"Using device: {device}")
import matplotlib.pyplot as plt
# Check operating system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import pickle
from glob import glob
import numpy as np
from pathlib import Path
# Switch to a different linear algebra backend
from measure_alignment import prepare_features,compute_score, compute_alignment
lvm_models = [
    {
        "model_name": "vit_tiny_patch16_224.augreg_in21k",
        "size": "tiny",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "augreg",
        "pretraining_dataset": "in21k",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_small_patch16_224.augreg_in21k",
        "size": "small",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "augreg",
        "pretraining_dataset": "in21k",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_base_patch16_224.augreg_in21k",
        "size": "base",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "augreg",
        "pretraining_dataset": "in21k",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_large_patch16_224.augreg_in21k",
        "size": "large",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "augreg",
        "pretraining_dataset": "in21k",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_base_patch16_224.mae",
        "size": "base",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "mae",
        "pretraining_dataset": "N/A (Self-Supervised)",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_large_patch16_224.mae",
        "size": "large",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "mae",
        "pretraining_dataset": "N/A (Self-Supervised)",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_huge_patch14_224.mae",
        "size": "huge",
        "patch_size": 14,
        "resolution": 224,
        "training_method": "mae",
        "pretraining_dataset": "N/A (Self-Supervised)",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_small_patch14_dinov2.lvd142m",
        "size": "small",
        "patch_size": 14,
        "resolution": None, # Note: Resolution is not specified in this model name
        "training_method": "dinov2",
        "pretraining_dataset": "lvd142m",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_base_patch14_dinov2.lvd142m",
        "size": "base",
        "patch_size": 14,
        "resolution": None, # Note: Resolution is not specified in this model name
        "training_method": "dinov2",
        "pretraining_dataset": "lvd142m",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_large_patch14_dinov2.lvd142m",
        "size": "large",
        "patch_size": 14,
        "resolution": None, # Note: Resolution is not specified in this model name
        "training_method": "dinov2",
        "pretraining_dataset": "lvd142m",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_giant_patch14_dinov2.lvd142m",
        "size": "giant",
        "patch_size": 14,
        "resolution": None, # Note: Resolution is not specified in this model name
        "training_method": "dinov2",
        "pretraining_dataset": "lvd142m",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_base_patch16_clip_224.laion2b",
        "size": "base",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "clip",
        "pretraining_dataset": "laion2b",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_large_patch14_clip_224.laion2b",
        "size": "large",
        "patch_size": 14,
        "resolution": 224,
        "training_method": "clip",
        "pretraining_dataset": "laion2b",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_huge_patch14_clip_224.laion2b",
        "size": "huge",
        "patch_size": 14,
        "resolution": 224,
        "training_method": "clip",
        "pretraining_dataset": "laion2b",
        "fine_tuned": False,
        "fine_tuning_dataset": None,
    },
    {
        "model_name": "vit_base_patch16_clip_224.laion2b_ft_in12k",
        "size": "base",
        "patch_size": 16,
        "resolution": 224,
        "training_method": "clip",
        "pretraining_dataset": "laion2b",
        "fine_tuned": True,
        "fine_tuning_dataset": "in12k",
    },
    {
        "model_name": "vit_large_patch14_clip_224.laion2b_ft_in12k",
        "size": "large",
        "patch_size": 14,
        "resolution": 224,
        "training_method": "clip",
        "pretraining_dataset": "laion2b",
        "fine_tuned": True,
        "fine_tuning_dataset": "in12k",
    },
    {
        "model_name": "vit_huge_patch14_clip_224.laion2b_ft_in12k",
        "size": "huge",
        "patch_size": 14,
        "resolution": 224,
        "training_method": "clip",
        "pretraining_dataset": "laion2b",
        "fine_tuned": True,
        "fine_tuning_dataset": "in12k",
    },
]


colors_min_rand_max = [np.divide((255, 153, 51), 255), np.divide((160, 160, 160), 256),
              np.divide((51, 153, 255), 255)]

SUPPORTED_METRICS = [
    "cycle_knn",
    "mutual_knn",
    "lcs_knn",
    "cka",
    "unbiased_cka",
    "cknna",
    "svcca",
    "edit_distance_knn",
]

import argparse
from collections import namedtuple
def get_args():
    parser = argparse.ArgumentParser(description='extract activations from a model')
    parser.add_argument('vision_type', type=str,
                        default='clip')
    parser.add_argument('layer_method', type=str, default='prh')
    args = parser.parse_args()
    return args

def mock_get_args():
    mock_args = namedtuple('debug', ['vision_type', 'layer_method'])
    new_args = mock_args('clip', 'prh')
    return new_args

debug=False
if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    if debug:
        args = mock_get_args()
    else:
        args = get_args()
    vision_type = args.vision_type
    layer_method = args.layer_method

    dataset = 'wit_1024'
    subset = 'train'
    topk = 10
    precise = True
    method_k = 5  # cknna

    # vision_type='clip'  # in21k, mae, dinov2, clip, clip_ft_in12k
    #layer_method='prh' # 'prh' or 'last'
    # make below a if else statement with comments as the key
    if vision_type=='in21k':
        select_indices=[0,1,2,3] # in21k
    elif vision_type=='mae':
        select_indices=[4,5,6] # mae
    elif vision_type=='dinov2':
        select_indices = [7, 8, 9, 10]  #dinov2
    elif vision_type=='clip':
        select_indices = [11, 12, 13]  # clip
    elif vision_type=='clip_ft_in12k':
        select_indices = [14,15, 16]  #   clip ft in12k

    selected_models = [lvm_models[i]['model_name'] for i in select_indices]
    selected_attributes = [lvm_models[i] for i in select_indices]

    #%%
    best_layer=[]
    if layer_method=='prh':
        vision_model_list=[]
        for model_ in selected_models:
            save_path = to_feature_filename(
                platonic_path, dataset, subset, model_,
                pool='cls', prompt=None, caption_idx=None,
            )
            vision_model_list.append(save_path)
        alignment_scores_full, alignment_indices_full = compute_alignment(vision_model_list,
                                                                          vision_model_list,
                                                                          SUPPORTED_METRICS[method_k], topk=topk,
                                                                          precise=precise)
        # select unique values in column of alignment_indices_full[:,;,0], that are not 0
        layers=[list(set(x)-set([0])) for x in alignment_indices_full[:,:,0].T]
        # asssert that the size of each element in best_layer is 1
        if all(len(x) == 1 for x in best_layer):
            best_layer=np.asarray(layers).squeeze()
        else:
            print("Not all elements in best_layer are of size 1")
            best_layer=[]
            layer_method='last'



    #%%
    extract_mode = 'redux'
    activations_list = []
    layers_list = []
    for model_ in selected_models:
        save_path = to_feature_filename(
            platonic_path, dataset, subset, model_,
            pool='cls', prompt=None, caption_idx=None,
        )
        act_=torch.load(save_path, map_location=device)['feats']

        if len(best_layer)==len(selected_models):
            layer_id= int(best_layer[selected_models.index(model_)])
            act_=act_[:,layer_id-1,:]
        else:
            layer_id = act_.shape[1]
            act_=act_[:,layer_id-1,:]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)

    feature_map_all = [x['activations'] for x in activations_list]

    for idx in range(len(feature_map_all)):
        X = feature_map_all[idx]
        X = torch.tensor(X).to(float_version)
        column_means = torch.mean(X, dim=0)
        centered_X = X - column_means
        # centered_X/=centered_X.norm(p='fro')
        feature_map_all[idx] = centered_X.to(device)
    # compute a forbenious norm aacross all models

    #%% compute max pad
    max_pad=max(x.shape[1] for x in feature_map_all)
    #%% do some zero-padding here
    #max_pad= 5920
    x_model = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]
    #%% do the norming
    normalize = lambda x: x / torch.sqrt(torch.trace(torch.mm(x.T, x)))
    x_model = [x.requires_grad_(False) for x in x_model]
    # do norm
    x_model = [normalize(x) for x in x_model]

    #%% compute the model procrustes first and then do model to brain alginment
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'fb'  # or 'streaming' , 'full_batch' is the default

    adjust_mode = 'zp'  # 'pca' or 'none' or 'zero_pad',
    svd_solver = 'gesvd'  # 'gesvd' or 'svd', or 'lowrank'
    tolerance = 1e-12
    steps= 500
    verbose = True
    prev_objective=10e10
    n_init=4
    X_bar_model_final=None
    aligned_Xs_model_final=None
    proc_file=Path(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/shape_metric/procrustes/proc_{vision_type}_{dataset}_{grp}_{method}_{svd_solver}_{adjust_mode}_tol_{tolerance}_layer_{layer_method}_norm.pkl')
    if proc_file.exists():
        print(f'File {proc_file} exists. Loading from it.')
        with open(proc_file.__str__(), 'rb') as f:
            results_dict = pkl.load(f)
        X_bar_model_final = results_dict['X_bar_model_final']
        aligned_Xs_model_final = results_dict['aligned_Xs_model_final']
        print('Loaded X_bar_model_final and aligned_Xs_model_final from file.')
        # exit the script
    else:
    # print configuration
        print(f'grp: {grp}, method: {method}, adjust_mode: {adjust_mode}, svd_solver: {svd_solver}, tolerance: {tolerance} \n')
        for k in range(n_init):
            # print the current iteration
            print(f'iteration: {k}')
            with torch.no_grad():
                if method == 'full_batch':
                    method_key='full_batch'
                else:
                    method_key='streaming'
                if adjust_mode=='zp':
                    adjust_mode_key='zero_pad'
                else:
                    adjust_mode_key='pca'
                X_bar_model, aligned_Xs_model = pt_frechet_mean(x_model, group=grp, method=method_key, return_aligned_Xs=True,#warmstart=X_init,
                                                          max_iter=steps,verbose=verbose, tol=tolerance,svd_solver=svd_solver)

            X_diff = [X - X_bar_model for X in aligned_Xs_model]
            X_diff = torch.stack(X_diff)
            X_var_model = X_diff.norm(dim=-1, p='fro')
            objective=X_var_model.norm()
            print(f'objective: {objective}')
            if objective<prev_objective:
                X_bar_model_final=X_bar_model
                aligned_Xs_model_final=aligned_Xs_model
                prev_objective=objective
        # save the result dict
        results_dict = dict(X_bar_model_final=X_bar_model_final, aligned_Xs_model_final=aligned_Xs_model_final)
        with open(proc_file.__str__(), 'wb') as f:
            pkl.dump(results_dict, f)


    #%%
    aligned_Xbar_model = []
    for idx in tqdm(range(len(x_model))):
        x = x_model[idx]
        aligned_Xbar_model.append(x @ pt_align(x, X_bar_model_final, group="orth", svd_solver=svd_solver))

    X_diff_model = [x - X_bar_model_final for x in aligned_Xs_model_final]
    X_diff_model = torch.stack(X_diff_model)
    X_var_model = torch.linalg.vector_norm(X_diff_model, ord=2, dim=-1)

    #X_var_model.norm(p='fro', dim=0, keepdim=True)
    X_var_dispersion = X_var_model.norm(p='fro', dim=0, keepdim=True)
    # rank order the X_var_model.norm(p='fro', dim=0, keepdim=True)
    key_samples_indices_dispersion = torch.argsort(X_var_dispersion, dim=1, descending=False).squeeze()
    key_samples_indices_dispersion=key_samples_indices_dispersion.cpu().numpy()
    #X_var_dispersion.cpu().numpy().squeeze()[key_samples_indices_dispersion]


    #%%
    scaler = StandardScaler()
    #X_standardized = scaler.fit_transform(X_var_model.T.cpu().numpy())
    X_standardized = X_var_model.T.cpu().numpy()
    X_standardized -= X_standardized.mean(axis=0, keepdims=True)
    pca = PCA(n_components=X_standardized.shape[1])
    X_pca = pca.fit_transform(X_standardized)
    pc1_scores = X_pca[:, 0]
    pc2_scores = X_pca[:, 1]
    # key_samples_indices = np.argsort(np.abs(pc1_scores))[::-1]
    key_samples_indices_pca = np.argsort(pc1_scores)
    #%%
    datas = load_dataset('minhuh/prh', revision=dataset, split='train')
    n_samples = 100
    min_rand_max_var_list = {}
    for selection_method in ['pca', 'rank']:
        if selection_method == 'pca':
            pc1_scores_sorted = pc1_scores[key_samples_indices_pca]
            # get variance explained by each component
            var_explained = pca.explained_variance_ratio_
            # get the components
            # compute the variance explained by each component in percentage
            var_explained_perc = var_explained * 100
            # print variance explained in human readable format

            variance_idx = key_samples_indices_pca
            min_var_idx = sorted(key_samples_indices_pca[:n_samples])
            min_var=pc1_scores_sorted[ :n_samples]
            max_var_idx = sorted(key_samples_indices_pca[-n_samples:])
            max_var=pc1_scores_sorted[-n_samples:]
            rand_var_idx = sorted(np.random.choice(variance_idx, n_samples, replace=False))
            min_rand_max_var_list[selection_method] = [min_var_idx,rand_var_idx,max_var_idx]
            [print(f"{x:.2f}%") for x in var_explained_perc]
            # plot X_pca_0 and X_pca_1
            fig_width_inches = 11
            fig_height_inches = 8.5
            fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
            ax = fig.add_axes([.1, .1, .3, .3])

            # choose colors so that the scale with the pca1 score, with max being red and min being blue

            colors = plt.cm.Reds(np.linspace(0.0, 1, len(key_samples_indices_pca)))
            x = pc1_scores[key_samples_indices_pca]
            y = pc2_scores[key_samples_indices_pca]
            ax.scatter(x, y, color=colors, edgecolor='none', s=15)
            # Add labels and title
            ax.set_xlabel(f'PC1, var explained:{var_explained_perc[0]:.2f}%', fontdict={'fontsize': 7})
            ax.set_ylabel(f'PC2, var explained:{var_explained_perc[1]:.2f}%', fontdict={'fontsize': 7})
            # plot the min points as hollow circles with colors from color_min_rand_max
            ax.scatter(pc1_scores[min_var_idx], pc2_scores[min_var_idx], facecolors=colors_min_rand_max[0],
                       edgecolors='black', s=15, label='min variance')
            # plot the max points as hollow circles with colors from color_min_rand_max
            ax.scatter(pc1_scores[rand_var_idx], pc2_scores[rand_var_idx], facecolors=colors_min_rand_max[1],
                       edgecolors='black', s=15, label='random variance')
            ax.scatter(pc1_scores[max_var_idx], pc2_scores[max_var_idx], facecolors=colors_min_rand_max[2]
                       , edgecolors='black', s=15, label='max variance')
            # plot the random points as hollow circles with colors from color_min_rand_max

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # add the origin lines
            # ax.axhline(0, color='black', linewidth=1)
            # ax.axvline(0, color='black', linewidth=1)
            min_val = min(np.concatenate([x, y], axis=0))
            max_val = max(np.concatenate([x, y], axis=0))
            min_val = min_val - 0.05 * min_val
            max_val = max_val + 0.05 * max_val
            ax.set_xlim([min_val, max_val])
            ax.set_ylim([min_val, max_val])
            # set fond size for the axis labels
            ax.tick_params(axis='both', which='major', labelsize=7)
            # do plot in the top left corner of pca plot to show variance explained per pca component
            ax1 = fig.add_axes([0.15, 0.35, 0.05, 0.05])
            ax1.bar(range(len(var_explained_perc)), var_explained_perc, color='red')
            # ax1.set_xticks(np.arange(len(var_explained_perc)).astype(int))
            ax1.set_xticks([])
            ax1.set_title('Variance explained')
            ax1.set_ylim([0, np.ceil(var_explained_perc[0])])
            ax1.set_yticks(np.arange(0, np.ceil(var_explained_perc[0]) + 1, 20))
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            # set all fonts in ax1 to be 7

            # Display the plot
            fig.show()
            anylsis_path = Path(
                platonic_path) / 'analysis' / 'procrustes' / f'pca_procrustes_{vision_type}_{grp}_{method}_{adjust_mode}_layer_{layer_method}.pdf'
            if not os.path.exists(os.path.dirname(anylsis_path)):
                os.makedirs(os.path.dirname(anylsis_path))
            fig.savefig(anylsis_path.__str__(), bbox_inches='tight', dpi=300)

        elif selection_method == 'rank':
            min_var_idx=sorted(key_samples_indices_dispersion[:n_samples])
            max_var_idx=sorted(key_samples_indices_dispersion[-n_samples:])
            rand_var_idx=sorted(key_samples_indices_dispersion[np.random.choice(key_samples_indices_dispersion.shape[0], n_samples, replace=False)])
            min_rand_max_var_list[selection_method] = [min_var_idx, rand_var_idx, max_var_idx]
            # plot a figure with values points of dispersion on y axis and stimnulus order on x axis, and highlight the min, max and random points
            fig_width_inches = 11
            fig_height_inches = 8.5
            fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
            ax = fig.add_axes([.1, .1, .3, .3])
            # choose colors so that the scale with the pca1 score, with max being red and min being blue
            colors = plt.cm.Reds(np.linspace(0.0, 1, len(key_samples_indices_dispersion)))
            x = np.arange(len(key_samples_indices_dispersion))
            y = X_var_dispersion.cpu().numpy().squeeze()
            ax.scatter(x,  y[key_samples_indices_dispersion], color=colors, edgecolor='none', s=15)
            # Add labels and title
            ax.set_xlabel('Stimulus Order', fontdict={'fontsize': 7})
            ax.set_ylabel('Dispersion', fontdict={'fontsize': 7})
            # plot the min points as hollow circles with colors from color_min_rand_max
            x_min=np.sort([np.argwhere(key_samples_indices_dispersion==x) for x in min_var_idx]).squeeze()
            ax.scatter(x_min, y[min_var_idx], facecolors='none',
                         edgecolors=colors_min_rand_max[0], s=18, label='min variance')
            # plot the max points as hollow circles with colors from color_min_rand_max
            x_rand= np.sort([np.argwhere(key_samples_indices_dispersion==x) for x in rand_var_idx]).squeeze()
            ax.scatter(x_rand, X_var_dispersion.cpu().numpy().squeeze()[rand_var_idx], facecolors='none',
                            edgecolors=colors_min_rand_max[1], s=18, label='random variance')
            x_max= np.sort([np.argwhere(key_samples_indices_dispersion==x) for x in max_var_idx]).squeeze()
            ax.scatter(x_max, X_var_dispersion.cpu().numpy().squeeze()[max_var_idx], facecolors='none'
                         , edgecolors=colors_min_rand_max[2], s=18, label='max variance')
            # plot the random points as hollow circles with colors from color_min_rand_max
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # add the origin lines
            fig.show()
            anylsis_path = Path(
                platonic_path) / 'analysis' / 'procrustes' / f'ranking_procrustes_{vision_type}_{grp}_{method}_{adjust_mode}_layer_{layer_method}.pdf'
            if not os.path.exists(os.path.dirname(anylsis_path)):
                os.makedirs(os.path.dirname(anylsis_path))
            fig.savefig(anylsis_path.__str__(), bbox_inches='tight', dpi=300)


    #%% compute prh alignment
    # clear min_var_idx, rand_var_idx, max_var_idx
    min_var_idx=[]
    rand_var_idx=[]
    max_var_idx=[]

    for selection_method in ['pca', 'rank']:

        min_var_idx = []
        rand_var_idx = []
        max_var_idx = []
        min_var_idx, rand_var_idx, max_var_idx = min_rand_max_var_list[selection_method]
        print(f'selection_method: {selection_method}')
        vlm_model_paths_random = []
        vlm_model_paths_min = []
        vlm_model_paths_max = []



        for model_ in selected_models:
            save_path = to_feature_filename(
                platonic_path, dataset, subset, model_,
                pool='cls', prompt=None, caption_idx=None,
            )

            act_=torch.load(save_path, map_location=device)
            new_path=save_path.replace('/train','/procrustes')
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            act_random={ 'feats': act_['feats'][rand_var_idx,:],'num_params': act_['num_params']}
            act_min={ 'feats': act_['feats'][min_var_idx,:],'num_params': act_['num_params']}
            act_max={ 'feats': act_['feats'][max_var_idx,:],'num_params': act_['num_params']}
            with open(new_path.replace('.pt', f'_random_{selection_method}.pt'), 'wb') as f:
                torch.save(act_random, f)
            vlm_model_paths_random.append(new_path.replace('.pt', f'_random_{selection_method}.pt'))
            with open(new_path.replace('.pt', f'_min_{selection_method}.pt'), 'wb') as f:
                torch.save(act_min, f)
            vlm_model_paths_min.append(new_path.replace('.pt', f'_min_{selection_method}.pt'))
            with open(new_path.replace('.pt', f'_max_{selection_method}.pt'), 'wb') as f:
                torch.save(act_max, f)
            vlm_model_paths_max.append(new_path.replace('.pt', f'_max_{selection_method}.pt'))


        # given that you are comparing vision models to langauge model, you need to save the same thing for langauge models



        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
        ]

        llm_model_path = []
        llm_model_paths_random=[]
        llm_model_paths_min=[]
        llm_model_paths_max=[]
        for model_ in llm_models:
            save_path = to_feature_filename(
                platonic_path, dataset, subset, model_, pool='avg', prompt=False, caption_idx=None)
            # assert path exist
            act_=torch.load(save_path, map_location=device)
            new_path=save_path.replace('/train','/procrustes')
            # create a dictionary with the same variables as act_
            act_random={}
            act_random['feats'] = act_['feats'][rand_var_idx,:, :]
            act_random['num_params'] = act_['num_params']
            act_random['mask'] = act_['mask'][rand_var_idx, :]

            act_min={}
            act_min['feats'] = act_['feats'][min_var_idx,:, :]
            act_min['num_params'] = act_['num_params']
            act_min['mask'] = act_['mask'][min_var_idx, :]

            act_max={}
            act_max['feats'] = act_['feats'][max_var_idx,:, :]
            act_max['num_params'] = act_['num_params']
            act_max['mask'] = act_['mask'][max_var_idx, :]

            with open(new_path.replace('.pt', f'_random_{selection_method}.pt'), 'wb') as f:
                torch.save(act_random, f)
            llm_model_paths_random.append(new_path.replace('.pt', f'_random_{selection_method}.pt'))
            with open(new_path.replace('.pt', f'_min_{selection_method}.pt'), 'wb') as f:
                torch.save(act_min, f)
            llm_model_paths_min.append(new_path.replace('.pt', f'_min_{selection_method}.pt'))

            with open(new_path.replace('.pt', f'_max_{selection_method}.pt'), 'wb') as f:
                torch.save(act_max, f)
            llm_model_paths_max.append(new_path.replace('.pt', f'_max_{selection_method}.pt'))



        #%% compute the alignment in PRH
        alignment_scores_min=[]
        alignment_scores_rand=[]
        alignment_scores_max=[]

        alignment_scores_rand, alignment_indices_rand = compute_alignment(vlm_model_paths_random, llm_model_paths_random,
                                                                          SUPPORTED_METRICS[method_k], topk=topk,
                                                                          precise=precise)

        alignment_scores_min, alignment_indices_min = compute_alignment(vlm_model_paths_min, llm_model_paths_min,
                                                                        SUPPORTED_METRICS[method_k], topk=topk,
                                                                        precise=precise)

        alignment_scores_max, alignment_indices_max = compute_alignment(vlm_model_paths_max, llm_model_paths_max,
                                                                        SUPPORTED_METRICS[method_k], topk=topk,
                                                                        precise=precise)




        #%% plot the results
        llm_models_short = [m.split('/')[-1] for m in llm_models]
        # get the following from selected models, "vit_base_patch16_clip_224.laion2b_ft_in12k" --> base, clip, liaon2b, in12k, and put it in a list of list

        alignments_all = np.stack([alignment_scores_min, alignment_scores_rand, alignment_scores_max])
        stimulus_label= ['min', 'random', 'max']

        import matplotlib.pyplot as plt
        import numpy as np
        import textwrap

        n_rows=1
        n_cols=len(selected_models)
        num_selected_models=len(selected_models)
        NUM_STIMULUS_SETS = alignments_all.shape[0]  # Number of stimulus sets (3 in this case)
        y_min = alignments_all[:, :num_selected_models, :].min()
        y_max = alignments_all[:, :num_selected_models, :].max()
        y_buffer = (y_max - y_min) * 0.05  # Add a 5% buffer
        y_lim_min = y_min - y_buffer
        y_lim_max = y_max + y_buffer

        # Dynamically calculate figsize. Give each subplot a width of ~4 inches and a height of ~5.
        fig_width_inches = 11
        fig_height_inches = 8.5
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))

        # --- Manual Layout Calculation ---
        # Define margins and spacing in fractions of the figure size
        left_margin = 0.07
        right_margin = 0.03
        bottom_margin = 0.20  # Increased bottom margin for vertical labels
        top_margin = 0.15
        hspace = 0.08  # Horizontal space between plots
        # Calculate the total available width and height for all plots
        total_plot_width = 1.0 - left_margin - right_margin
        total_plot_height = 1.0 - bottom_margin - top_margin
        # Calculate the width of a single subplot
        subplot_width = (total_plot_width - (4 - 1) * hspace) / 4
        # Calculate the height to make the plot square, accounting for the figure's aspect ratio
        subplot_height = subplot_width * (fig_width_inches / fig_height_inches)
        llm_indices = np.arange(len(llm_models_short))
        for i in range(num_selected_models):
            # Calculate the 'left' position of the current subplot
            left = left_margin + i * (subplot_width + hspace)
            bottom = bottom_margin

            # Add the axes to the figure at the calculated position
            ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

            # Plot one line for each stimulus set
            for j in range(NUM_STIMULUS_SETS):
                # Extract the scores for the current ViT model (i) and stimulus set (j)
                scores_for_llms = alignments_all[j, i, :]
                ax.plot(llm_indices, scores_for_llms, marker='o', linestyle='-', label=f'{stimulus_label[j]}', color=colors_min_rand_max[j])

            # --- Subplot Customization ---
            # Wrap the long model name for better display
            model_title = textwrap.fill(selected_attributes[i]['model_name'], width=1000)
            ax.set_title(f'{model_title}\n{SUPPORTED_METRICS[method_k]},k={topk}', fontsize=7)
            ax.set_xlabel('Language Models', fontsize=8)

            # Only show y-label on the first plot
            if i == 0:
                ax.set_ylabel('Alignment Score', fontsize=8)
            ax.set_ylim(y_lim_min, y_lim_max)
            # Set x-ticks and rotate labels vertically
            ax.set_xticks(llm_indices)
            ax.set_xticklabels(llm_models_short, rotation='vertical', fontsize=7)

            # Set y-tick label size
            ax.tick_params(axis='y', which='major', labelsize=7)

            #ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=7)

            # Add a main title for the entire figure
        fig.suptitle('ViT Model Alignment with LLMs across Stimulus Sets', fontsize=16)

        # Display the plot
        fig.show()
        anylsis_path= Path(platonic_path) / 'analysis' / 'procrustes' / f'alignment_to_{vision_type}_{dataset}_samples_{n_samples}_{SUPPORTED_METRICS[method_k]}_topk_{topk}_proc_{grp}_{method}_{adjust_mode}_sample_{selection_method}_layer_{layer_method}.pdf'
        if not os.path.exists(os.path.dirname(anylsis_path)):
            os.makedirs(os.path.dirname(anylsis_path))
        fig.savefig(anylsis_path.__str__(), bbox_inches='tight', dpi=300)




    #%% save images
    # analysis_path=Path(platonic_path) / 'analysis' / 'procrustes' / f'alignment_to_{vision_type}_{dataset}_samples_{n_samples}_{SUPPORTED_METRICS[method_k]}_topk_{topk}_proc_{grp}_{method}_{adjust_mode}_sample_{selection_method}'
    # if not os.path.exists(analysis_path):
    #     os.makedirs(analysis_path)
    # for i in range(5):  # Save the first 5 images
    #     image = datas[i]['image']
    #     text = datas[i]['text']  # Assuming there's a 'text' column as well
    #
    #     # You can use information from the dataset (like text) to name your images
    #     # Be careful with special characters in filenames.
    #     filename_base = f"image_{i}"
    #
    #     # Clean up text for filename if needed (e.g., remove problematic characters)
    #     # A simple approach:
    #     # cleaned_text = "".join(c for c in text if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    #     # filename_base = f"image_{i}_{cleaned_text[:20]}" # Take first 20 chars of cleaned text
    #
    #     image_path = os.path.join(analysis_path, f"{filename_base}.png")  # Save as PNG
    #
    #     image.save(image_path)
    #     print(f"Image {i + 1} saved to {image_path}")