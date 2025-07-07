
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
import getpass
# import svd from scipy
import scipy
from scipy.linalg import svd
from scipy.linalg import orthogonal_procrustes
if getpass.getuser() == 'ehoseini':
    sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
    image_paths = '/om2/user/ehoseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/om2/user/ehoseini/MyData/neural_nlp_bench/activations/DeepJuice_DsParametricfMRI/'
    benchmark_path = '/om2/user/ehoseini/MyData/DeepJuice/nsd_data/'
else:
    sys.path.append('/Users/eghbalhosseini/MyCodes/DeepJuiceDev/')
    image_paths = '/Users/eghbalhosseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/Users/eghbalhosseini/MyData/DeepJuice/workspace/nsd/'
    benchmark_path = '/Users/eghbalhosseini/MyData/DeepJuice/nsd_data/'
from scipy.stats import median_abs_deviation as mad
from benchmarks import NSDBenchmark, NSDSampleBenchmark
from deepjuice._backends.cupyfy import convert_to_tensor
from deepjuice._backends.cupyfy import convert_to_tensor
from sampling.get_DeepJuice_model_score_for_optimized_Ds_set import get_cross_validated_benchmarking_results
from deepjuice.alignment import compute_pearson_rdm
from netrep.utils import align, pt_align, pt_orthogonal_procrustes

import multiprocessing
import os
from sklearn.preprocessing import StandardScaler
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
    selected_models = ['torchvision_alexnet_imagenet1k_v1',
                       'torchvision_regnet_x_800mf_imagenet1k_v2',
                       'openclip_vit_b_32_laion2b_e16',
                       'timm_swinv2_cr_tiny_ns_224',
                       'torchvision_efficientnet_b1_imagenet1k_v2',
                       'timm_convnext_large_in22k',
                       ]

    models_sh = ['AlexNet', 'RegNet', 'ViT', 'Swin', 'EfficientNet',  'ConvNext']
    # read image path
    with open(image_paths, 'rb') as f:
        image_paths = pickle.load(f)

    #%%
    extract_mode = 'redux'
    activations_list = []
    layers_list = []
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
        X = feature_map_all[idx]
        X = torch.tensor(X).to(float_version)
        column_means = torch.mean(X, dim=0)
        centered_X = X - column_means
        # centered_X/=centered_X.norm(p='fro')
        feature_map_all[idx] = centered_X.to(device)

    #%%
    benchmark_ = NSDBenchmark(path_dir=benchmark_path)
    x_fmri = (convert_to_tensor(benchmark_.response_data.to_numpy()).to(dtype=float_version, device=device)).T
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    rois = roi_indices.keys()
    roi = 'OTC'
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices[roi].values()]
    # for each roi get the largest size and pad the rest
    max_pad = max([x.shape[1] for x in fmri_roi_sub_x])
    x_sub_fmri = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in fmri_roi_sub_x]

    # read image path
    #%% do some zero-padding here
    #max_pad= 5920
    #torch.norm(Xbar - X0) / torch.sqrt(torch.tensor(Xbar.numel(), dtype=torch.float))
    #
    x_sub_fmri = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in fmri_roi_sub_x]
    x_model = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]
    #%% do the norming
    # drop the required grad
    normalize = lambda x: x / torch.sqrt(torch.trace(torch.mm(x.T, x)))
    x_model = [x.requires_grad_(False) for x in x_model]
    x_sub_fmri = [x.requires_grad_(False) for x in x_sub_fmri]
    # do norm
    x_model = [normalize(x) for x in x_model]
    x_sub_fmri = [normalize(x) for x in x_sub_fmri]
    ##
    [torch.trace(torch.mm(x.T, x)) for x in x_model]
    [torch.trace(torch.mm(x.T, x)) for x in x_sub_fmri]
    #%% compute the model procrustes first and then do model to brain alginment
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    svd_solver = 'gesvd'  # 'gesvd' or 'svd', or 'lowrank'
    tolerance = 1e-16
    steps = 100
    verbose = True
    n_init = 5
    prev_objective = 1e10
    X_bar_model_final = None
    aligned_Xs_model_final = None
    file=Path(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/shape_metric_highres_vision_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.pkl')
    # load file
    with open(file.__str__(), 'rb') as f:
        results_dict = pd.read_pickle(f)
    # align subjects to the mean model

    X_bar_model=results_dict['X_bar_model_final']
    torch.trace(torch.mm(X_bar_model.T,X_bar_model))
    X_bar_model=normalize(X_bar_model)
    #aligned_Xbar_model=[normalize(x) for x in results_dict['aligned_Xs_model_final']]
    #X_bar_model_norm= torch.stack(aligned_Xbar_model).mean(dim=0)

    #X_bar_model_norm = normalize_1(results_dict['X_bar_model_final'])
    # run subject fmri data through alignment
    aligned_Xbar_model=[]
    for idx in tqdm(range(len(x_model))):
        x=x_model[idx]
        aligned_Xbar_model.append(x @ pt_align(x, X_bar_model, group="orth",svd_solver=svd_solver))

    aligned_Xbar_sub=[]
    for idx in tqdm(range(len(x_sub_fmri))):
        x=x_sub_fmri[idx]
        aligned_Xbar_sub.append(x @ pt_align(x, X_bar_model, group="orth",svd_solver=svd_solver))

    # create a shuffle index based on x_sub_fmri[0].shape[0]

    # aligned_Xbar_sub_shuffle = []
    # for idx in tqdm(range(len(x_sub_fmri))):
    #     x = x_sub_fmri_shuffle[idx]
    #     # create arandom matrixwith the same size as x
    #     x_rand = torch.randn_like(x)
    #     x_rand = normalize(x_rand)
    #     aligned_Xbar_sub_shuffle.append(x_rand @ pt_align(x_rand, X_bar_model_norm, group="orth", svd_solver=svd_solver))
    #     #aligned_Xbar_sub_shuffle.append(x_rand)

    # normalize the data
    X_diff_model = [x - X_bar_model for x in aligned_Xbar_model]
    X_diff_model = torch.stack(X_diff_model)
    X_var_model = torch.linalg.vector_norm(X_diff_model, ord=2,dim=-1)

    X_diff_sub = [X - X_bar_model for X in aligned_Xbar_sub]
    X_diff_sub = torch.stack(X_diff_sub)
    X_var_sub  = torch.linalg.vector_norm(X_diff_sub, ord=2,dim=-1)

    x_bar_mean=torch.linalg.vector_norm(X_bar_model,ord=2,dim=-1)


    x = X_var_model.mean(dim=0).cpu().numpy()
    y = X_var_sub.mean(dim=0).cpu().numpy()
    #y_shuffle = x_bar_mean.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='red', edgecolor='black', s=50)
    #ax.scatter(x, y_shuffle, color='gray', edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    min_val = min_val - 0.05 * min_val
    max_val = max_val + 0.05 * max_val
    ax.set_xlabel('Average distance all models to X_bar_model',fontdict={'fontsize': 20})
    ax.set_ylabel('Average distance brain to X_bar_model',fontdict={'fontsize': 20})
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    corr_matrix = np.corrcoef(x, y)
    ax.set_title(f'{roi}, corr: {corr_matrix[0, 1]:.2f}')
    fig.show()
    # save the data for prodcuing these resutls
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.png',dpi=300)
    # save eps
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.eps',format='eps')
#%%
    x = X_var_model.std(dim=0).cpu().numpy()
    y = X_var_sub.std(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='red', edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    min_val = min_val - 0.05 * min_val
    max_val = max_val + 0.05 * max_val
    ax.set_xlabel('variance of distance all models to X_bar_model', fontdict={'fontsize': 20})
    ax.set_ylabel('variance of distance brain to X_bar_model', fontdict={'fontsize': 20})
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    corr_matrix = np.corrcoef(x, y)
    ax.set_title(f'{roi}, corr: {corr_matrix[0, 1]:.2f}')
    fig.show()
    # save the data for prodcuing these resutls
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_variance_distance_to_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_norm.png',
        dpi=300)
    # save eps
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_norm.eps',
        format='eps')

    #%% plot the distance between models and subjects#%%
    # do it for individuaal models
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.ravel()
    for id_model in range(6):
        x_var_model = X_var_model[id_model].cpu().numpy()
        y = X_var_sub.mean(dim=0).cpu().numpy()
        #y_shuffle = X_var_sub_shuffle.mean(dim=0).cpu().numpy()
        ax = axs[id_model]


        ax.scatter(x_var_model, y, color='red', edgecolor='black',
                   s=50)
        #ax.scatter(x_var_model, y_shuffle, color='gray', edgecolor='black',
        #           s=50)
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

        corr_matrix = np.corrcoef(x_var_model, y)
        ax.set_title(f'{models_sh[id_model]}, corr: {corr_matrix[0, 1]:.2f}')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_distance_to_x_bar_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.png',
        dpi=300)
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_distance_to_x_bar_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.eps')
    #%%
    #%% plot the distance between models and subjects#%%
    # do it for individuaal models
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.ravel()
    for id_sub in range(4):
        x = X_var_model.mean(dim=0).cpu().numpy()
        y = X_var_sub[id_sub].cpu().numpy()
        ax = axs[id_sub]
        ax.scatter(x, y, color='red', edgecolor='black',
                   s=50)
        # Plot the identity line
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
        min_val = min_val - 0.05 * min_val
        max_val = max_val + 0.05 * max_val
        # Add labels and title
        if id_sub == 0:
            axs[id_sub].set_xlabel(f'average distance all model to x_bar')
            axs[id_sub].set_ylabel('average distance subject brain to x_bar')
        ax.set_title(f'sub {id_sub}')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        corr_matrix = np.corrcoef(x, y)
        # Extract the correlation coefficient
        corr_coefficient = corr_matrix[0, 1]
        ax.set_title(f'subj {id_sub}, corr: {corr_matrix[0, 1]:.2f}')

    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_subject_distance_to_x_bar_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_norm.png',
        dpi=300)
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_subject_distance_to_x_bar_models_and_subjects_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_norm.eps')
    #%%
    scaler = StandardScaler()
    #X_standardized = scaler.fit_transform(X_var_model.T.cpu().numpy())
    X_standardized = X_var_model.T.cpu().numpy()
    X_standardized-=X_standardized.mean(axis=0,keepdims=True)
    pca = PCA(n_components=X_standardized.shape[1])
    X_pca = pca.fit_transform(X_standardized)
    pc1_scores = X_pca[:, 0]
    pc2_scores = X_pca[:, 1]
    #key_samples_indices = np.argsort(np.abs(pc1_scores))[::-1]
    key_samples_indices = np.argsort(pc1_scores)
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
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    min_val = min(np.concatenate([x,y],axis=0))
    max_val = max(np.concatenate([x,y],axis=0))
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
    # save figure
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_pca_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.png',dpi=300)
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_pca_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.eps',format='eps')

    # do the same for the subjects
    #X_standardized_sub = scaler.fit_transform(X_var_sub.T.cpu().numpy())
    X_standardized_sub = X_var_sub.T.cpu().numpy()
    X_standardized_sub-=X_standardized_sub.mean(axis=0,keepdims=True)
    pca_sub = PCA(n_components=X_standardized_sub.shape[1])
    X_pca_sub = pca_sub.fit_transform(X_standardized_sub)
    pc1_scores_sub = X_pca_sub[:, 0]
    key_samples_indices_sub = np.argsort(np.abs(pc1_scores_sub))[::-1]
    # get variance explained by each component
    var_explained_sub = pca_sub.explained_variance_ratio_
    # get the components
    # compute the variance explained by each component in percentage
    var_explained_perc_sub = var_explained_sub * 100
    # print variance explained in human readable format

    [print(f"{x:.2f}%") for x in var_explained_perc_sub]
    #%%
    new_path='/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/nsd_data/stimulus/NSD_shared1000/'
    org_path='/Users/eghbalhosseini/MyCodes/DeepJuiceDev/benchmarks/nsd_sample/stimulus/NSD_shared1000/'
    # repalce org_path with new_path in image_paths
    image_paths=[x.replace(org_path,new_path) for x in image_paths]
    # create new forldername shape_metric_vision_orth_full_batch_zero_pad_1e-13_norm
    new_folder_name=f'shape_metric_vision_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm_image_order_1st_pc'
    # make sure the folder exists
    new_path=f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/results/{new_folder_name}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # if the path exists then remove all the files
    else:
        os.system(f'rm -r {new_path + "/*"}')
    # add a number based on new_image_order in front file name and save it into new_folder_name
    for i,im in tqdm(enumerate(key_samples_indices)):
        new_im_path=image_paths[im]
        new_im_name=new_im_path.split('/')[-1]
        new_im_name=f'{i}_{new_im_name}'
        new_im_path=f'{new_path}/{new_im_name}'
        os.system(f'cp {image_paths[im]} {new_im_path}')

    #%% create a color that goes from white to red and is the length of the number of images
    colors = plt.cm.Reds(np.linspace(0, 1, len(key_samples_indices)))
    # rorder them based on new_images_order
    #colors = colors[new_images_order]
    x = X_var_model.mean(dim=0).cpu().numpy()[key_samples_indices]
    #x = pc1_scores[key_samples_indices]
    y = X_var_sub.mean(dim=0).cpu().numpy()[key_samples_indices]
    #y = pc1_scores_sub[key_samples_indices]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color=colors, edgecolor='black', s=50)
    # Plot the identity line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=3)
    min_val = min_val - 0.05 * min_val
    max_val = max_val + 0.05 * max_val
    # Add labels and title
    ax.set_xlabel('Average distance all models to X_bar_model', fontdict={'fontsize': 20})
    ax.set_ylabel('Average distance brain to X_bar_model', fontdict={'fontsize': 20})

    # Show grid
    # Set limits
    plt.xlim([min_val, .5*max_val])
    plt.ylim([min_val, max_val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    corr_matrix = np.corrcoef(x, y)
    ax.set_title(f'{roi}, corr: {corr_matrix[0, 1]:.2f}')
    # Display the plot
    fig.show()
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_and_subjects_color_by_PC_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.png',dpi=300)
    # save eps
    fig.savefig(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_bar_model_distance_to_models_and_subjects_color_by_PC_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.eps',format='eps')

    #%%
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.ravel()
    for id_model in range(6):
        x_var_model = X_var_model[id_model].cpu().numpy()[key_samples_indices]
        y = X_var_sub.mean(dim=0).cpu().numpy()[key_samples_indices]
        ax = axs[id_model]

        ax.scatter(x_var_model, y, color=colors, edgecolor='black',
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

        corr_matrix = np.corrcoef(x_var_model, y)
        ax.set_title(f'{models_sh[id_model]}, corr: {corr_matrix[0, 1]:.2f}')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_distance_to_x_bar_models_and_subjects_color_by_PC_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_norm.png',
        dpi=300)
    fig.savefig(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_model_distance_to_x_bar_models_and_subjects_color_by_PC_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_norm.eps')
    # %%
    n_samples = 80

    variance_idx = key_samples_indices
    # take 80 samples from the first 100 and last 100
    min_var_idx=sorted(variance_idx[:80])
    max_var_idx=sorted(variance_idx[-80:])
    rand_var_idx=sorted(np.random.choice(variance_idx,80,replace=False))

    # %%
    benchmark_ = NSDBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/')
    benchmark_min = NSDSampleBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/',
                                       image_samples=min_var_idx)
    benchmark_max = NSDSampleBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/',
                                       image_samples=max_var_idx)
    benchmark_rand = NSDSampleBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/',
                                        image_samples=rand_var_idx)
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    x_fmri = convert_to_tensor(benchmark_.response_data.to_numpy()).T
    x_fmri_min = convert_to_tensor(benchmark_min.response_data.to_numpy()).T
    x_fmri_max = convert_to_tensor(benchmark_max.response_data.to_numpy()).T
    x_fmri_rand = convert_to_tensor(benchmark_rand.response_data.to_numpy()).T

    feature_map_min = [x[min_var_idx, :].to(torch.float32) for x in feature_map_all]
    feature_map_max = [x[max_var_idx, :].to(torch.float32) for x in feature_map_all]
    feature_map_rand = [x[rand_var_idx, :].to(torch.float32) for x in feature_map_all]

    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_min = [x_fmri_min[:, indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_max = [x_fmri_max[:, indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_rand = [x_fmri_rand[:, indx] for indx in roi_indices['OTC'].values()]
    #
    benchmark_min.build_rdms(compute_pearson_rdm)
    benchmark_max.build_rdms(compute_pearson_rdm)
    benchmark_rand.build_rdms(compute_pearson_rdm)
    benchmark_.build_rdms(compute_pearson_rdm)

    score_min = get_cross_validated_benchmarking_results(benchmark=benchmark_min, feature_extractor=feature_map_min,
                                                         extract_mode=extract_mode)

    score_rand = get_cross_validated_benchmarking_results(benchmark=benchmark_rand, feature_extractor=feature_map_rand,
                                                          extract_mode=extract_mode)
    score_max = get_cross_validated_benchmarking_results(benchmark=benchmark_max, feature_extractor=feature_map_max,
                                                         extract_mode=extract_mode)

    score_all = get_cross_validated_benchmarking_results(benchmark=benchmark_, feature_extractor=[x.to(torch.float32) for x in feature_map_all],
                                                         extract_mode=extract_mode)

    n_subs = len(np.unique(benchmark_rand.metadata.subj_id.values))
    # #
    all_regions = ['EVC', 'OTC', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4',
                   'FFA-1', 'FFA-2', 'OFA', 'EBA', 'FBA-1', 'FBA-2', 'OPA', 'PPA',
                   'VWFA-1', 'VWFA-2', 'OWFA']
    colors = [np.divide((255, 153, 51), 255), np.divide((160, 160, 160), 256),
              np.divide((51, 153, 255), 255)]



    plot_path = '/om2/user/ehoseini/MyData/DeepJuice/'

    for region in all_regions:
        x_min = [list(x['srpr'][x['srpr']['region'] == region]['score']) for x in score_min]
        x_min = np.squeeze(np.stack(x_min))
        # take the mean over last axis
        # x_min=np.mean(x_min,axis=-1)

        x_rand = [list(x['srpr'][x['srpr']['region'] == region]['score']) for x in score_rand]
        x_rand = np.squeeze(np.stack(x_rand))
        # x_rand = np.mean(x_rand, axis=-1)

        x_max = [list(x['srpr'][x['srpr']['region'] == region]['score']) for x in score_max]
        x_max = np.squeeze(np.stack(x_max))

        x_all = [list(x['srpr'][x['srpr']['region'] == region]['score']) for x in score_all]
        x_all = np.squeeze(np.stack(x_all))
        # x_max = np.mean(x_max, axis=-1)

        width = 0.15  # the width of the bars
        fig = plt.figure(figsize=(11, 8))
        # fig_length = 0.055 * len(models_scores)
        ax = plt.axes((.1, .4, .35, .35))
        x = np.arange(len(x_min))

        # model_name = model_sh

        rects2 = ax.bar(x - 0.3, np.median(x_all, axis=1), width, label='all', color='w', linewidth=.5,
                        edgecolor='k')
        ax.errorbar(x - 0.3, np.median(x_all, axis=1), yerr=mad(x_all, axis=1), linestyle='', color='k')
        rects2 = ax.bar(x - 0.1, np.median(x_min, axis=1), width, label='agree', color=colors[0], linewidth=.5,
                        edgecolor='k')
        ax.errorbar(x - 0.1, np.median(x_min, axis=1), yerr=mad(x_min, axis=1), linestyle='', color='k')
        # plot the second item
        rects2 = ax.bar(x + .1, np.median(x_rand, axis=1), width, label='random', color=colors[1], linewidth=.5,
                        edgecolor='k')
        ax.errorbar(x + .1, np.median(x_rand, axis=1), yerr=mad(x_rand, axis=1), linestyle='', color='k')
        # plot the third item
        rects3 = ax.bar(x + 0.3, np.median(x_max, axis=1), width, label='disagree', color=colors[2], linewidth=.5,
                        edgecolor='k')
        ax.errorbar(x + 0.3, np.median(x_max, axis=1), yerr=mad(x_max, axis=1), linestyle='', color='k')

        # ax.errorbar(x  , models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
        ax.axhline(y=0, color='k', linestyle='-')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Pearson correlation')
        ax.set_title(f' {region}')
        ax.set_xticks(x)
        ax.set_xticklabels(models_sh, rotation=45)
        ax.set_ylim((-.0, 0.5))
        ax.set_xlim((-.5, 6.5))
        ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.show()
        fig.savefig(
            f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_var_brain_fit_by_PC1_{region}_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.png',
            dpi=300)
        # save eps
        fig.savefig(
            f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/figures/x_var_brain_fit_by_PC1_{region}_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.eps',
            format='eps')
