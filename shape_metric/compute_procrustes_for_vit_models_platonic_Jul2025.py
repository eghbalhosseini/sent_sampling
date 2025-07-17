
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
if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    selected_models = [             "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",]

    models_sh = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']
    # read image path
    dataset='wit_1024'
    subset='train'

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
        layer_id=act_.shape[1]
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
    # create a set of random matrix with same size as the model
    #%%
    # make them not require grad
    #%% compute the model procrustes first and then do model to brain alginment
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    svd_solver = 'gesvd'  # 'gesvd' or 'svd', or 'lowrank'
    tolerance = 1e-12
    steps= 100
    verbose = True
    prev_objective=10e10
    n_init=5
    X_bar_model_final=None
    aligned_Xs_model_final=None
    # print configuration
    print(f'grp: {grp}, method: {method}, adjust_mode: {adjust_mode}, svd_solver: {svd_solver}, tolerance: {tolerance} \n')
    for k in range(n_init):
        # print the current iteration
        print(f'iteration: {k}')
        with torch.no_grad():

            X_bar_model, aligned_Xs_model = pt_frechet_mean(x_model, group=grp, method=method, return_aligned_Xs=True,#warmstart=X_init,
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

    # align subjects to the mean model
    # safe final x_bar_model and aligned_Xs_model
    file=Path(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/shape_metric_highres_vision_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.pkl')
    results_dict=dict(X_bar_model_final=X_bar_model_final,aligned_Xs_model_final=aligned_Xs_model_final)
    with open(file.__str__(), 'wb') as f:
        pickle.dump(results_dict, f)


    #%%

    scaler = StandardScaler()
    # X_standardized = scaler.fit_transform(X_var_model.T.cpu().numpy())
    X_standardized = X_var_model.T.cpu().numpy()
    X_standardized -= X_standardized.mean(axis=0, keepdims=True)
    pca = PCA(n_components=X_standardized.shape[1])
    X_pca = pca.fit_transform(X_standardized)
    pc1_scores = X_pca[:, 0]
    pc2_scores = X_pca[:, 1]
    # key_samples_indices = np.argsort(np.abs(pc1_scores))[::-1]
    key_samples_indices = np.argsort(pc1_scores)
    pc1_scores_sorted= pc1_scores[key_samples_indices]
    # get variance explained by each component
    var_explained = pca.explained_variance_ratio_
    # get the components
    # compute the variance explained by each component in percentage
    var_explained_perc = var_explained * 100
    # print variance explained in human readable format

    [print(f"{x:.2f}%") for x in var_explained_perc]
    # plot X_pca_0 and X_pca_1
    fig, ax = plt.subplots(figsize=(8, 8))
    # choose colors so that the scale with the pca1 score, with max being red and min being blue

    colors = plt.cm.Reds(np.linspace(0, 1, len(key_samples_indices)))
    x = pc1_scores[key_samples_indices]
    y = pc2_scores[key_samples_indices]
    ax.scatter(x, y, color=colors, edgecolor='black', s=50)
    # Add labels and title
    ax.set_xlabel(f'PC1, var explained:{var_explained_perc[0]:.2f}%', fontdict={'fontsize': 20})
    ax.set_ylabel(f'PC2, var explained:{var_explained_perc[1]:.2f}%', fontdict={'fontsize': 20})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add the origin lines
    #ax.axhline(0, color='black', linewidth=1)
    #ax.axvline(0, color='black', linewidth=1)
    min_val = min(np.concatenate([x, y], axis=0))
    max_val = max(np.concatenate([x, y], axis=0))
    min_val = min_val - 0.05 * min_val
    max_val = max_val + 0.05 * max_val
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    # do plot in the top left corner of pca plot to show variance explained per pca component
    ax1 = fig.add_axes([0.2, 0.8, 0.15, 0.1])
    ax1.bar(range(len(var_explained_perc)), var_explained_perc, color='red')
    ax1.set_xticks(range(len(var_explained_perc)))
    ax1.set_xticklabels(range(1, len(var_explained_perc) + 1))
    ax1.set_title('Variance explained')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Display the plot
    fig.show()
    #%%
    datas = load_dataset('minhuh/prh', revision='wit_1024', split='train')
    n_samples = 100
    variance_idx = key_samples_indices
    min_var_idx = sorted(key_samples_indices[:n_samples])
    min_var=pc1_scores_sorted[ :n_samples]
    max_var_idx = sorted(key_samples_indices[-n_samples:])
    max_var=pc1_scores_sorted[-n_samples:]
    rand_var_idx = sorted(np.random.choice(variance_idx, n_samples, replace=False))


    for model_ in selected_models:
        save_path = to_feature_filename(
            platonic_path, dataset, subset, model_,
            pool='cls', prompt=None, caption_idx=None,
        )
        act_=torch.load(save_path, map_location=device)
        new_path=save_path.replace('/train','/procrustes')
        act_random={ 'feats': act_['feats'][rand_var_idx,:],'num_params': act_['num_params']}
        act_min={ 'feats': act_['feats'][min_var_idx,:],'num_params': act_['num_params']}
        act_max={ 'feats': act_['feats'][max_var_idx,:],'num_params': act_['num_params']}
        with open(new_path.replace('.pt', f'_random.pt'), 'wb') as f:
            torch.save(act_random, f)
        with open(new_path.replace('.pt', f'_min.pt'), 'wb') as f:
            torch.save(act_min, f)
        with open(new_path.replace('.pt', f'_max.pt'), 'wb') as f:
            torch.save(act_max, f)


    # given that you are comparing vision models to langauge model, you need to save the same thing for langauge models

    llm_models = [
        "huggyllama/llama-7b",
        "huggyllama/llama-13b",

    ]
    llm_model_path = []
    for model_ in llm_models:
        save_path = to_feature_filename(
            platonic_path, dataset, subset, model_, pool='last', prompt=True, caption_idx=None)
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

        with open(new_path.replace('.pt', f'_random.pt'), 'wb') as f:
            torch.save(act_random, f)
        with open(new_path.replace('.pt', f'_min.pt'), 'wb') as f:
            torch.save(act_min, f)

        with open(new_path.replace('.pt', f'_max.pt'), 'wb') as f:
            torch.save(act_max, f)





