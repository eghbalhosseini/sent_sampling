
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
sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
from scipy.stats import median_abs_deviation as mad
from benchmarks import NSDBenchmark, NSDSampleBenchmark
from deepjuice._backends.cupyfy import convert_to_tensor
import multiprocessing
import os
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
os.environ['OMP_NUM_THREADS'] = '4'
import matplotlib.pyplot as plt
# Check operating system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import pickle
from glob import glob

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
    # compute a forbenious norm aacross all models
    #model_norm=torch.stack(feature_map_all).norm( p='fro')
    # devide all feature_maps by the norm
    #feature_map_all=[x/model_norm for x in feature_map_all]
    #%%
    benchmark_ = NSDBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/')
    x_fmri = (convert_to_tensor(benchmark_.response_data.to_numpy()).to(dtype=torch.float64, device=device)).T
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices['OTC'].values()]
    max_pad = max([x.shape[1] for x in fmri_roi_sub_x])
    x_sub_fmri = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in fmri_roi_sub_x]
    #sub_norm = torch.stack(x_sub_fmri).norm(p='fro')
    #x_sub_fmri = [x / sub_norm for x in x_sub_fmri]
    #%% do some zero-padding here
    x_sub_fmri = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in fmri_roi_sub_x]
    x_model = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]
    #%% do the norming
    #x_model = [x/x.norm(p='fro') for x in x_model]
    #x_sub_fmri = [x/x.norm(p='fro') for x in x_sub_fmri]
    #%% compute the model procrustes first and then do model to brain alginment
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-8
    steps= 2000
    verbose = True
    prev_objective=1e10
    X_bar_model_final=None
    aligned_Xs_model_final=None
    for k in range(10):
        X_bar_model, aligned_Xs_model = pt_frechet_mean(x_model, group=grp, method=method, return_aligned_Xs=True,
                                                      max_iter=steps,verbose=verbose, tol=tolerance)

        X_diff = [X - X_bar_model for X in aligned_Xs_model]
        X_diff = torch.stack(X_diff)
        # X_diff[:,0,:].norm(dim=1)
        X_var_model = X_diff.norm(dim=-1, p='fro')
        objective=X_var_model.norm()
        print(f'objective: {objective}')
        if objective<prev_objective:
            X_bar_model_final=X_bar_model
            aligned_Xs_model_final=aligned_Xs_model

    # align subjects to the mean model
    aligned_Xbar_sub = [x @ pt_align(x, X_bar_model_final, group="orth") for x in x_sub_fmri]

    X_bar_sub_from_models=torch.stack(aligned_Xbar_sub).mean(dim=0)

    X_diff = [X - X_bar_model_final for X in aligned_Xs_model_final]
    X_diff = torch.stack(X_diff)
    X_var_model = X_diff.norm(dim=-1, p='fro')

    X_diff = [X - X_bar_model_final for X in aligned_Xbar_sub]
    X_diff = torch.stack(X_diff)
    X_var_sub = X_diff.norm(dim=-1, p='fro')

    x = X_var_model.mean(dim=0).cpu().numpy()
    y = X_var_sub.mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='red', edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    min_val = min_val - 0.05 * min_val
    max_val = max_val + 0.05 * max_val
    # Add labels and title
    ax.set_xlabel('Average distance all models to X_bar_model',fontdict={'fontsize': 20})
    ax.set_ylabel('Average distance brain to X_bar_model',fontdict={'fontsize': 20})

    # Show grid
    # Set limits
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Display the plot
    fig.show()
    # save the data for prodcuing these resutls
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_and_subjects.png',dpi=300)
    # save eps
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_and_subjects.eps',format='eps')

    #%% plot the distance between models and subjects#%%
    # do it for individuaal models
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.ravel()
    for id_model in range(6):
        x_var_model = X_var_model[id_model].cpu().numpy()
        y = X_var_sub.mean(dim=0).cpu().numpy()
        ax = axs[id_model]


        ax.scatter(x_var_model, y, color='red', edgecolor='black',
                   s=50)
        # Plot the identity line
        min_val = min(min(x_var_model), min(y))
        max_val = max(max(x_var_model), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
        min_val = min_val - 0.05 * min_val
        max_val = max_val + 0.05 * max_val
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
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_distance_to_x_bar_models_and_subjects.png',
        dpi=300)
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
    X_bar_sub_final = None
    aligned_Xs_sub_final = None
    for k in range(10):
        X_bar_sub, aligned_Xs_sub = pt_frechet_mean(x_sub_fmri, group=grp, method=method, return_aligned_Xs=True,
                                                      max_iter=steps,
                                                      verbose=verbose, tol=tolerance)
        X_diff = [X - X_bar_sub for X in aligned_Xs_sub]
        X_diff = torch.stack(X_diff)
        # X_diff[:,0,:].norm(dim=1)
        X_var_ = X_diff.norm(dim=-1, p='fro')
        objective = X_var_.norm()
        print(f'objective: {objective}')
        if objective < prev_objective:
            X_bar_sub_final = X_bar_sub
            aligned_Xs_sub_final = aligned_Xs_sub

    aligned_X_bar_models = [x @ pt_align(x, X_bar_sub_final, group="orth") for x in x_model]

    X_diff = [X - X_bar_sub_final for X in aligned_Xs_sub_final]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_sub = X_diff.norm(dim=-1, p='fro')

    X_diff = [X - X_bar_sub_final for X in aligned_X_bar_models]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_model = X_diff.norm(dim=-1, p='fro')

    x = X_var_model.mean(dim=0).cpu().numpy()
    y = X_var_sub.mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color=[.3,.3,1], edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    min_val = min_val - 0.2 * min_val
    max_val = max_val + 0.05 * max_val
    # Add labels and title
    ax.set_xlabel('Average distance all models to X_bar_brain', fontdict={'fontsize': 20})
    ax.set_ylabel('Average distance all brains to X_bar_brain', fontdict={'fontsize': 20})

    # Show grid
    # Set limits
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Display the plot
    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_brain_distance_to_models_and_subjects.png',
        dpi=300)
    # save eps
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_brain_distance_to_models_and_subjects.eps',
        format='eps')
    #%%
    x_pca=X_var_sub.cpu().numpy()
    x_pca=x_pca-x_pca.mean(axis=0)
    pca = PCA(n_components=4)
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
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.ravel()
    for id_model in range(6):
        x_var_model = X_var_model[id_model].cpu().numpy()
        y = X_var_sub.mean(dim=0).cpu().numpy()
        ax = axs[id_model]

        ax.scatter(x_var_model, y, color=[.3,.3,1], edgecolor='black',
                   s=50)
        # Plot the identity line
        min_val = min(min(x_var_model), min(y))
        max_val = max(max(x_var_model), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
        min_val = min_val - 0.5 * min_val
        max_val = max_val + 0.05 * max_val
        # Add labels and title
        if id_model == 0:
            axs[id_model].set_xlabel(f'distance model to x_bar_brain')
            axs[id_model].set_ylabel('average distance brain to x_bar_brain')
        ax.set_title(f'{models_sh[id_model]}')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_distance_to_x_bar_brain_and_subjects_unnorm.png',
        dpi=300)


    #%%
    X_diff = [X - X_bar_sub_final for X in aligned_Xs_sub_final]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_sub = X_diff.norm(dim=-1, p='fro')

    X_diff = [X - X_bar_model_final for X in aligned_Xs_model_final]
    X_diff = torch.stack(X_diff)
    # X_diff[:,0,:].norm(dim=1)
    X_var_model = X_diff.norm(dim=-1, p='fro')

    x = X_var_model.mean(dim=0).cpu().numpy()
    y = X_var_sub.mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color=[.5, .5, .5], edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    min_val = min_val - 0.25 * min_val
    max_val = max_val + 0.05 * max_val# Add labels and title
    ax.set_xlabel('Average distance all models to X_bar_model', fontdict={'fontsize': 20})
    ax.set_ylabel('Average distance all brains to X_bar_brain', fontdict={'fontsize': 20})

    # Show grid
    # Set limits
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Display the plot
    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_vs_x_bar_brain_to_subjects.png',
        dpi=300)
    # save eps
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_vs_x_bar_brain_to_subjects.eps',
        format='eps')


    #%%
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.ravel()
    for id_model in range(6):
        x_var_model = X_var_model[id_model].cpu().numpy()
        y = X_var_sub.mean(dim=0).cpu().numpy()
        ax = axs[id_model]

        ax.scatter(x_var_model, y, color=[.5,.5,.5], edgecolor='black',
                   s=50)
        # Plot the identity line
        min_val = min(min(x_var_model), min(y))
        max_val = max(max(x_var_model), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
        min_val = min_val - 0.5 * min_val
        max_val = max_val + 0.05 * max_val
        # Add labels and title
        if id_model == 0:
            axs[id_model].set_xlabel(f'distance model to x_bar_brain')
            axs[id_model].set_ylabel('average distance brain to x_bar_brain')
        ax.set_title(f'{models_sh[id_model]}')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_distance_to_x_bar_model_and_subjects_to_x_bar_brain.png',
        dpi=300)


