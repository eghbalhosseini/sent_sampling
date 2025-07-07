from netrep.metrics import LinearMetric
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import cross_validate
from netrep.multiset import pairwise_distances, frechet_mean, pt_frechet_mean
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool
from netrep.utils import align, pt_align

from scipy.spatial.distance import pdist
from scipy.io import savemat
import torch
import torch.nn.functional as F
#matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 3,'font.weight':'normal'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
from scipy.spatial.distance import pdist
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
'''ANN result across models'''
model_layers = [('roberta-base', 'encoder.layer.1'),
                ('xlnet-large-cased', 'encoder.layer.23'),
                ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                ('gpt2-xl', 'encoder.h.43'),
                ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                ('ctrl', 'h.46'),]

import sys
from netrep.utils import align, pt_align
sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
from scipy.stats import median_abs_deviation as mad
from benchmarks import NSDBenchmark, NSDSampleBenchmark
from deepjuice._backends.cupyfy import convert_to_tensor
from sampling.get_DeepJuice_model_score_for_optimized_Ds_set import get_cross_validated_benchmarking_results
from deepjuice.alignment import compute_pearson_rdm

import multiprocessing
import os
import seaborn as sns
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
os.environ['OMP_NUM_THREADS'] = '4'
import matplotlib.pyplot as plt
import platform

# Check operating system
if platform.system() == 'Darwin':  # Darwin is the system name for macOS
    # Check if MPS (Metal Performance Shaders) backend is available, for Apple Silicon Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS on supported Macs
        device = torch.device("cpu")  # Fallback to CPU if MPS is not available
    else:
        device = torch.device("cpu")  # Fallback to CPU if MPS is not available
else:
    # For non-macOS, you can default to CPU or check for CUDA (NVIDIA GPU) availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

'''ANN result across models'''

import pickle
from glob import glob

def procrustes_dist(x,y):
    tr_xtx = torch.trace(torch.mm(x.T, x))
    tr_yty = torch.trace(torch.mm(y.T, y))
    #U, S, Vh = torch.linalg.svd(torch.mm(x.T, y))
    S=torch.linalg.svdvals(torch.mm(x.T, y),driver='gesvd')
    procrustes_ = tr_xtx + tr_yty - 2 * sum(S)
    return procrustes_

def bures_dist(x,y):
    xxt = torch.mm(x, x.t())
    lam, U = torch.linalg.eigh(xxt)
    xxt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T
    # Compute YY^T
    yyt = torch.mm(y, y.t())
    xy_interim = torch.mm(torch.mm(xxt_sqrt, yyt), xxt_sqrt)
    lam, U = torch.linalg.eigh(xy_interim)
    xyt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T
    # compute bures,
    bures_ = torch.trace(xxt) + torch.trace(yyt) - 2 * torch.trace(xyt_sqrt)
    return bures_

def bures_dist_epsilon(x, y, epsilon=1e-8):
    # Compute XX^T
    xxt = torch.mm(x, x.t())
    lam, U = torch.linalg.eigh(xxt)
    xxt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T

    # Compute YY^T
    yyt = torch.mm(y, y.t())

    # Compute the interim matrix and add regularization
    xy_interim = torch.mm(torch.mm(xxt_sqrt, yyt), xxt_sqrt)
    xy_interim += torch.eye(xy_interim.size(0)).to(x.device) * epsilon

    try:
        lam, U = torch.linalg.eigh(xy_interim)
    except torch._C._LinAlgError as e:
        print("Encountered an error with linalg.eigh:", e)
        return None

    xyt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T

    # Compute Bures distance
    bures_ = torch.trace(xxt) + torch.trace(yyt) - 2 * torch.trace(xyt_sqrt)
    return bures_


if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    # load act_leftout
    selected_models = ['torchvision_alexnet_imagenet1k_v1',
                       'torchvision_regnet_x_800mf_imagenet1k_v2',
                       'openclip_vit_b_32_laion2b_e16',
                       'timm_swinv2_cr_tiny_ns_224',
                       'torchvision_efficientnet_b1_imagenet1k_v2',
                       'timm_convnext_large_in22k',
                       ]

    models_sh = ['AlexNet', 'RegNet', 'ViT', 'Swin', 'EfficientNet',  'ConvNext']
    image_paths = '/om2/user/ehoseini/MyData/DeepJuice/NSD_image_paths.pkl'
    # read image path
    with open(image_paths, 'rb') as f:
        image_paths = pickle.load(f)

    #%%
    extract_mode = 'redux'
    activations_list = []
    layers_list = []

    deepjuice_ws_path = '/om2/user/ehoseini/MyData/neural_nlp_bench/activations/DeepJuice_DsParametricfMRI/'
    for model_ in selected_models:
        save_file = f'{deepjuice_ws_path}/{model_}*{extract_mode}.pkl'
        original_files = glob(save_file)
        # open the file
        with open(original_files[0], 'rb') as f:
            original = pickle.load(f)
        layer_id = original[0]
        act_ = original[1]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)

    feature_map_all = [x['activations'] for x in activations_list]

    for idx in range(len(feature_map_all)):
        X=feature_map_all[idx]
        X=torch.tensor(X).to(torch.float64)
        column_means = torch.mean(X, dim=0)
        centered_X = X - column_means
        #centered_X/=centered_X.norm(p='fro')
        feature_map_all[idx]=centered_X.to(device)

    benchmark_ = NSDBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/')
    x_fmri = (convert_to_tensor(benchmark_.response_data.to_numpy()).to(dtype=torch.float64, device=device)).T
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices['OTC'].values()]
    for subject_id in tqdm(range(len(fmri_roi_sub_x))):
        x_sub=fmri_roi_sub_x[subject_id]
        x_sub=x_sub-x_sub.mean(dim=0)
        #x_sub/=x_sub.norm(p='fro')
        # create a new list with all the models and the x_sub
        feature_map_all_brain=feature_map_all+[x_sub]
        # assert
        len(feature_map_all_brain)==7

        #%% perform mulitset distance
        grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
        method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
        adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
        tolerance = 1e-7
        steps= 2000
        verbose = True
        # warmstart= make it to previous X_bar
        if adjust_mode == 'zero_pad':
            X_shape = [x.shape[-1] for x in feature_map_all_brain]
            max_shape = max(X_shape)
            # pad each X with zeros to make it max_shape
            X_pad = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all_brain]
        else:
            X_pad = feature_map_all_brain
        assert(len(X_pad)==7)
        X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True, max_iter=steps,verbose=verbose,tol=tolerance)


        #%%
        X_diff=[X-X_var_all for X in aligned_Xs_all]
        X_diff=torch.stack(X_diff)
        #X_diff[:,0,:].norm(dim=1)
        X_var=X_diff.norm(dim=-1,p='fro')
        # compute the variance along the rows
        #%%
        # pick the last column of x_var as the brain
        brain_xvar=X_var[-1,:]
        model_xvar=X_var[:-1,:]
        x=model_xvar.mean(dim=0).cpu().numpy()
        y=brain_xvar.cpu().numpy()
        # plot a scatter on x-axis mean of model var and y axis brain var
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, color='red', edgecolor='black', s=50)
        # Plot the identity line
        min_val = min(min(x), min(y))-10
        max_val = max(max(x), max(y))+10
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
        # Add labels and title
        ax.set_xlabel('average distance all models to X_bar')
        ax.set_ylabel('distance brain to x_bar')

        # Show grid
        # Set limits
        plt.xlim([min_val, max_val])
        plt.ylim([min_val, max_val])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Display the plot
        # save the figure
        plt.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures//brain_vs_model_procurses_sub_{subject_id}.png', dpi=300)
        # create a figure with 6 suplots and in each subplot show the distance from model i to mean, vs brain to model i

        #%%
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs = axs.ravel()
        x_brain=aligned_Xs_all[-1]
        for id_model in range(6):
            x_model=aligned_Xs_all[id_model]
            ax=axs[id_model]
            dist_brain_to_model=(x_model-x_brain).norm(dim=-1,p='fro')
            dist_model_to_mean=(x_model-X_var_all).norm(dim=-1,p='fro')
            ax.scatter(dist_model_to_mean.cpu().numpy(),dist_brain_to_model.cpu().numpy(), color='red', edgecolor='black', s=50)
            # Plot the identity line
            min_val = min(min(dist_model_to_mean.cpu().numpy()), min(dist_brain_to_model.cpu().numpy()))-10
            max_val = max(max(dist_model_to_mean.cpu().numpy()), max(dist_brain_to_model.cpu().numpy()))+10
            ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
            # Add labels and title
            if id_model==0:
                axs[id_model].set_xlabel(f'distance model to x_bar')
                axs[id_model].set_ylabel('distance model to brain')
            ax.set_title(f'{models_sh[id_model]}')
            ax.set_xlim([min_val, max_val])
            ax.set_ylim([min_val, max_val])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        # save figure
        plt.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures//model_i_brain_vs_mean_procurses_sub_{subject_id}.png', dpi=300)

    #%% compute the model procrustes first and then do model to brain alginment
    max_pad=max([x.shape[1] for x in fmri_roi_sub_x])
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-7
    steps= 2000
    verbose = True
    # warmstart= make it to previous X_bar
        # pad each X with zeros to make it max_shape
    X_pad = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]
    x_sub_fmri = [F.pad(x , pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in fmri_roi_sub_x]
    # do a division by the norm
    #X_pad = [x/x.norm(p='fro') for x in X_pad]
    #x_sub_fmri = [x/x.norm(p='fro') for x in x_sub_fmri]
    prev_objective=1e10
    X_var_all_final=None
    aligned_Xs_model_final=None
    for k in range(10):
        X_var_all, aligned_Xs_model = pt_frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True,
                                                      max_iter=steps,
                                                      verbose=verbose, tol=tolerance)

        X_diff = [X - X_var_all for X in aligned_Xs_model]
        X_diff = torch.stack(X_diff)
        # X_diff[:,0,:].norm(dim=1)
        X_var_model = X_diff.norm(dim=-1, p='fro')
        objective=X_var_model.norm()
        print(f'objective: {objective}')
        if objective<prev_objective:
            X_var_all_final=X_var_all
            aligned_Xs_model_final=aligned_Xs_model



    aligned_Xbar_sub = [x @ pt_align(x, X_var_all_final, group="orth") for x in x_sub_fmri]
    preload = True
    if preload==True:
        data_for_plot=pd.read_pickle('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/data_for_alignment_between_models_and_brains_all_subjects.pkl')
        X_var_all_final=data_for_plot['X_var_all_final']
        aligned_Xs_model_final=data_for_plot['aligned_Xs_model_final']
        aligned_Xbar_sub=data_for_plot['aligned_Xbar_sub']
        X_var_sub=data_for_plot['X_var_sub']

    X_diff = [X - X_var_all_final for X in aligned_Xs_model_final]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_model = X_diff.norm(dim=-1, p='fro')

    X_diff = [X - X_var_all_final for X in aligned_Xbar_sub]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_sub = X_diff.norm(dim=-1, p='fro')

    x = X_var_model.mean(dim=0).cpu().numpy()
    y = X_var_sub.mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='red', edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y)) - 10
    max_val = max(max(x), max(y)) + 10
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    # Add labels and title
    ax.set_xlabel('Average distance all models to X_bar',fontdict={'fontsize': 20})
    ax.set_ylabel('Average distance brain to X_bar',fontdict={'fontsize': 20})

    # Show grid
    # Set limits
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Display the plot
    fig.show()
    # save the data for prodcuing these resutls
    data_for_plot={'X_var_all_final':X_var_all_final,
                   'aligned_Xs_model_final':aligned_Xs_model_final,
                   'aligned_Xbar_sub':aligned_Xbar_sub,
                   'X_var_sub':X_var_sub}
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/alignment_between_models_and_brains_all_subjects.png',
        dpi=300)
    # save eps
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/alignment_between_models_and_brains_all_subjects.eps',
        format='eps')
    # save the data
    with open(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/data_for_alignment_between_models_and_brains_all_subjects.pkl','wb') as f:
        pkl.dump(data_for_plot,f)

    # do it for individuaal models
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.ravel()
    for id_model in range(6):
        x_model = X_var_model[id_model].cpu().numpy()
        y = X_var_model.mean(dim=0).cpu().numpy()
        ax = axs[id_model]


        ax.scatter(x_model, y, color='red', edgecolor='black',
                   s=50)
        # Plot the identity line
        min_val = min(min(x_model), min(y)) - 10
        max_val = max(max(x_model), max(y)) + 10
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
        # Add labels and title
        if id_model == 0:
            axs[id_model].set_xlabel(f'distance model to x_bar')
            axs[id_model].set_ylabel('average distance brain to x_bar')
        ax.set_title(f'{models_sh[id_model]}')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.show()
    #%%
    x_pca=X_var_model.cpu().numpy()
    x_pca=x_pca-x_pca.mean(axis=0)
    pca = PCA(n_components=6)
    pca.fit(x_pca)
    # get variance explained by each component
    var_explained = pca.explained_variance_ratio_
    # get the components
    components = pca.components_
    # get the loadings
    loadings = pca.components_.T
    # compute the variance explained by each component in percentage
    var_explained_perc = var_explained * 100
    # print variance explained in human readable format

    [print(f"{x:.2f}%") for x in var_explained_perc]





    #%%
    prev_objective = 1e10
    X_var_sub_final = None
    aligned_Xs_sub_final = None
    for k in range(5):
        X_var_sub, aligned_Xs_sub = pt_frechet_mean(x_sub_fmri, group=grp, method=method, return_aligned_Xs=True,
                                                      max_iter=steps,
                                                      verbose=verbose, tol=tolerance)

        X_diff = [X - X_var_sub for X in aligned_Xs_sub]
        X_diff = torch.stack(X_diff)
        # X_diff[:,0,:].norm(dim=1)
        X_var_ = X_diff.norm(dim=-1, p='fro')
        objective = X_var_.norm()
        print(f'objective: {objective}')
        if objective < prev_objective:
            X_var_sub_final = X_var_sub
            aligned_Xs_sub_final = aligned_Xs_sub

    aligned_Xbar_models = [x @ pt_align(x, X_var_sub_final, group="orth") for x in X_pad]

    X_diff = [X - X_var_sub_final for X in aligned_Xs_sub_final]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_sub = X_diff.norm(dim=-1, p='fro')

    X_diff = [X - X_var_sub_final for X in X_pad]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_model = X_diff.norm(dim=-1, p='fro')

    x = X_var_model.mean(dim=0).cpu().numpy()
    y = X_var_sub.mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='blue', edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y)) - 10
    max_val = max(max(x), max(y)) + 10
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    # Add labels and title
    ax.set_xlabel('Average distance all models to X_bar for brain', fontdict={'fontsize': 20})
    ax.set_ylabel('Average distance brain to X_bar for brain', fontdict={'fontsize': 20})

    # Show grid
    # Set limits
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Display the plot
    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/alignment_between_models_and_brains_for_x_bar_brain.png',
        dpi=300)
    # save eps
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/alignment_between_models_and_brains_for_x_bar_brain.eps',
        format='eps')




