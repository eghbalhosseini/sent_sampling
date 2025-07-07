from netrep.metrics import LinearMetric
from pathlib import Path
import pickle as pkl
import pandas as pd
from sklearn.model_selection import cross_validate
from netrep.multiset import pairwise_distances, frechet_mean, pt_frechet_mean
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib
import torch.nn.functional as F
from netrep.utils import align, pt_align
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
#matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 3,'font.weight':'normal'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sent_sampling.utils import make_shorthand
'''ANN result across models'''
selected_models = ['torchvision_alexnet_imagenet1k_v1',
                   'torchvision_regnet_x_800mf_imagenet1k_v2',
                   'openclip_vit_b_32_laion2b_e16',
                   'timm_swinv2_cr_tiny_ns_224',
                   'torchvision_efficientnet_b1_imagenet1k_v2',
                   'clip_rn50',
                   'timm_convnext_large_in22k',
                   ]

import multiprocessing
import os
import seaborn as sns
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
os.environ['OMP_NUM_THREADS'] = '4'
import matplotlib.pyplot as plt
from netrep.utils import align
from scipy.spatial.distance import pdist
from scipy.io import savemat
import pickle
from glob import glob
import torch
# Check operating system
import platform
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

if __name__ == '__main__':
    pre_pca=False
    extract_mode = 'redux'
    suffix='_jsd_mult_80'
    with_jsd='jsd' in suffix
    if with_jsd:
        ds_suffix='_jsd'
    else:
        ds_suffix=''
    n_samples = 80
    optim_id_min = f'coordinate_ascent_eh-obj=2-D_s{ds_suffix}-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_id_max = f'coordinate_ascent_eh-obj=D_s{ds_suffix}-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_id_rand = f'coordinate_ascent_eh-obj=D_s_rand-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    deepjuice_id = f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    selected_models = ['torchvision_alexnet_imagenet1k_v1',
                       'torchvision_regnet_x_800mf_imagenet1k_v2',
                       'openclip_vit_b_32_laion2b_e16',
                       'timm_swinv2_cr_tiny_ns_224',
                       'torchvision_efficientnet_b1_imagenet1k_v2',
                       'clip_rn50',
                       'timm_convnext_large_in22k',
                       ]
    models_sh = ['AlexNet', 'RegNet', 'ViT', 'Swin', 'EfficientNet', 'CLIP', 'ConvNext']
    image_paths = '/om2/user/ehoseini/MyData/DeepJuice/NSD_image_paths.pkl'
    # read image path
    with open(image_paths, 'rb') as f:
        image_paths = pickle.load(f)
    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
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

    feature_maps = [x['activations'] for x in activations_list]
    results_path = RESULTS_DIR
    (extract_short_hand, optim_short_hand_min) = make_shorthand(deepjuice_id, optim_id_min)
    ds_min_path=f'{RESULTS_DIR}/results_{extract_short_hand}_{optim_short_hand_min}_{extract_mode}{suffix}.pkl'
    with open(ds_min_path, 'rb') as f:
        results_ds_min = pickle.load(f)

    (extract_short_hand, optim_short_hand_max) = make_shorthand(deepjuice_id, optim_id_max)
    ds_max_path = f'{results_path}/results_{extract_short_hand}_{optim_short_hand_max}_{extract_mode}{suffix}.pkl'
    with open(ds_max_path, 'rb') as f:
        results_ds_max = pickle.load(f)

    (extract_short_hand, optim_short_rand) = make_shorthand(deepjuice_id, optim_id_rand)
    ds_rand_path = f'{results_path}/results_{extract_short_hand}_{optim_short_rand}_{extract_mode}.pkl'
    with open(ds_rand_path, 'rb') as f:
        results_ds_rand = pickle.load(f)
    ds_min_image_ids = results_ds_min['optimized_S']
    ds_max_image_ids = results_ds_max['optimized_S']
    ds_rand_image_ids = results_ds_rand['optimized_S']
    feature_map_min = [x[sorted(ds_min_image_ids), :] for x in feature_maps]
    feature_map_max = [x[sorted(ds_max_image_ids), :] for x in feature_maps]
    feature_map_rand = [x[sorted(ds_rand_image_ids), :] for x in feature_maps]
    sorted_image_ids = np.stack([sorted(ds_min_image_ids), sorted(ds_max_image_ids)]).flatten()
    selected_image_paths = [image_paths[x] for x in sorted_image_ids]
    # strip the path to the image name
    selected_image_names = [x.split('/')[-1] for x in selected_image_paths]
    # drop the extension
    selected_image_names = [x.split('.')[0] for x in selected_image_names]
#%%
    feature_map_all_ = []
    if pre_pca:
        for idx in range(len(feature_maps)):
            pass
    else:
        for idx in range(len(feature_maps)):
            # center
            X_min = feature_map_min[idx]
            column_means = np.mean(X_min, axis=0)
            centered_X_min = X_min - column_means

            X_max = feature_map_max[idx]
            column_means = np.mean(X_max, axis=0)
            centered_X_max = X_max - column_means

            X_rand = feature_map_rand[idx]
            column_means = np.mean(X_rand, axis=0)
            centered_X_rand = X_rand - column_means

            feature_map_min.append(centered_X_min)
            feature_map_max.append(centered_X_max)
            feature_map_rand.append(centered_X_rand)
        model_group_act = {'min': feature_map_min, 'rand': feature_map_rand, 'max': feature_map_max}
    #%%
    feature_map_min = [torch.tensor(x).to(device) for x in feature_map_min]
    feature_map_max = [torch.tensor(x).to(device) for x in feature_map_max]

    #%% perform mulitset distance
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-6
    max_iter=500
    verbose = True
    act_dir = deepjuice_ws_path
    file_name=f'multi_shape_distance_individual_DeepJuice_DsParametric{suffix}_{grp}_{adjust_mode}_{method}_pre_pca_{pre_pca}_centered'
    save_path = Path(f'{act_dir}/{file_name}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)

     #   #aligned_Xs_all=all_X_dict['aligned_all']
        #X_var_all=all_X_dict['var_al']
    X=feature_map_min
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in feature_map_min]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in X]
        X_pad_max = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_max]

    X_var_min, aligned_Xs_min = pt_frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True,max_iter=max_iter,
                                         verbose=verbose,tol=tolerance)
    X_var_max, aligned_Xs_max = pt_frechet_mean(X_pad_max, group=grp, method=method, return_aligned_Xs=True, max_iter=max_iter,
                                     verbose=verbose,tol=tolerance)
    #%%
    x_align_min_min = []
    x_align_max_min = []
    for idx, X_p in enumerate(X_pad):
        # first compute the alginment matrix
        T = pt_align(X_var_min, X_pad[idx], group=grp)
        x_align_min_min.append(torch.matmul(X_pad[idx], T))
        x_align_max_min.append(torch.matmul(X_pad_max[idx], T))

    x_align_min_max = []
    x_align_max_max = []
    for idx, X_p in enumerate(X_pad_max):
        # first compute the alginment matrix
        T = pt_align(X_var_max, X_pad_max[idx], group=grp)
        x_align_min_max.append(torch.matmul(X_pad[idx], T))
        x_align_max_max.append(torch.matmul(X_pad_max[idx], T))

    x_align_max_min = np.stack([x.cpu().numpy() for x in x_align_max_min])
    x_align_max_min = np.mean(x_align_max_min, axis=0)
    x_align_min_min = np.stack([x.cpu().numpy() for x in x_align_min_min])
    x_align_min_min = np.mean(x_align_min_min, axis=0)

    x_align_min_max = np.stack([x.cpu().numpy() for x in x_align_min_max])
    x_align_min_max = np.mean(x_align_min_max, axis=0)
    x_align_max_max = np.stack([x.cpu().numpy() for x in x_align_max_max])
    x_align_max_max = np.mean(x_align_max_max, axis=0)
    # average over dimension 0
    pca = PCA(n_components=2)
    # do a pca on x_align_min and then transform x_align_max
    #X = np.stack([X_var_min.cpu().numpy(), x_align_max_min])
    X = np.stack([x_align_min_min, x_align_max_min])
    X = np.concatenate(X, axis=0)
    X = pca.fit_transform(X)
    X_align_pca_min_min = X[:X_var_min.shape[0], :]
    X_align_pca_max_min = X[X_var_min.shape[0]:, :]
    print(pca.explained_variance_ratio_)
    #
    #X = np.stack([x_align_min_max, X_var_max.cpu().numpy()])
    X=np.stack([x_align_min_max, x_align_max_max])
    X = np.concatenate(X, axis=0)
    X = pca.fit_transform(X)
    X_align_pca_min_max = X[:x_align_min_max.shape[0], :]
    X_align_pca_max_max = X[x_align_min_max.shape[0]:, :]
    # print the variance explained
    print(pca.explained_variance_ratio_)


    #%%
    x_pca = np.concatenate((X_align_pca_min_min[:,:2],X_align_pca_max_min[:,:2] ), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_align_pca_min_min.shape[0]), np.repeat('max', X_align_pca_min_min.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    # Define your color palette for groups
    color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}
    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    # add the image_id as a text next to the point
    # Plot the marginals
    sns.histplot(data=df, x="x", hue="labels", palette=color_palette, ax=g.ax_marg_x, legend=False, binwidth=20,
                 element="step", fill=False)
    sns.histplot(data=df, y="y", hue="labels", palette=color_palette, ax=g.ax_marg_y, legend=False, binwidth=20,
                 element="step", fill=False)
    g.fig.show()
    g.savefig(os.path.join(act_dir, f'DsParamfMRI{suffix}_DeepJuice_Align_max_to_min_{file_name}.png'))
    # save eps

    save_path = Path(f'{act_dir}/multi_shape_distance_individual_DeepJuice_DsParametric{suffix}_max_to_min_{grp}_{adjust_mode}_{method}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids})


    #%%

    x_pca = np.concatenate((X_align_pca_min_max[:,:2],X_align_pca_max_max[:,:2] ), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_align_pca_min_max.shape[0]), np.repeat('max', X_align_pca_max_max.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids

    # Define your color palette for groups
    color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}
    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    # add the image_id as a text next to the point
    # Plot the marginals
    sns.histplot(data=df, x="x", hue="labels", palette=color_palette, ax=g.ax_marg_x, legend=False, binwidth=20,
                 element="step", fill=False)
    sns.histplot(data=df, y="y", hue="labels", palette=color_palette, ax=g.ax_marg_y, legend=False, binwidth=20,
                 element="step", fill=False)
    g.fig.show()
    g.savefig(os.path.join(act_dir, f'DsParamefMRI{suffix}_DeepJuice_Align_min_to_max_{file_name}.png'))

    save_path = Path(
        f'{act_dir}/multi_shape_distance_individual_DeepJuice_DsParametric{suffix}_min_to_max_{grp}_{adjust_mode}_{method}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids})

    #%%
    column_means = np.mean(X_var_min.cpu().numpy(), axis=0)
    centered_X_var_min = X_var_min.cpu().numpy() - column_means
    column_means = np.mean(X_var_max.cpu().numpy(), axis=0)
    centered_X_var_max = X_var_max.cpu().numpy() - column_means

    X_pca = pca.fit_transform(centered_X_var_min)
    X_max=pca.transform(centered_X_var_max)
    X_align_pca_min = X_pca
    X_align_pca_max = X_max

    x_pca = np.concatenate((X_align_pca_min[:, :2], X_align_pca_max[:, :2]), axis=0)
    # create labels max and min
    labels = np.concatenate(
        (np.repeat('min', X_align_pca_max.shape[0]), np.repeat('max', X_align_pca_max.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    # Define your color palette for groups
    color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}
    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    g.ax_joint.set_xlim(-300, +300)
    g.ax_joint.set_ylim(-300, +300)
    # add the image_id as a text next to the point
    # Plot the marginals
    sns.histplot(data=df, x="x", hue="labels", palette=color_palette, ax=g.ax_marg_x, legend=False, binwidth=20,
                 element="step", fill=False)
    g.ax_marg_x.set_xlim(-300, +300)


    sns.histplot(data=df, y="y", hue="labels", palette=color_palette, ax=g.ax_marg_y, legend=False, binwidth=20,
                 element="step", fill=False)

    g.ax_marg_y.set_ylim(-300, +300)
    g.figure.show()
    g.savefig(os.path.join(act_dir, f'DsParamfMRI{suffix}_DeepJuice_project_max_to_min_X_var_{file_name}.png'))
    # save eps
    save_path = Path(f'{act_dir}/MSD_DsParamfMRI{suffix}_DeepJuice_project_max_to_min_X_var_{file_name}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids})

    #%%

    X_align_pca_max = pca.fit_transform(centered_X_var_max)
    X_align_pca_min=pca.transform(centered_X_var_min)


    x_pca = np.concatenate((X_align_pca_min[:, :2], X_align_pca_max[:, :2]), axis=0)
    # create labels max and min
    labels = np.concatenate(
        (np.repeat('min', X_align_pca_max.shape[0]), np.repeat('max', X_align_pca_max.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    # Define your color palette for groups
    color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}
    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    # add the image_id as a text next to the point
    # Plot the marginals
    g.ax_joint.set_xlim(-300, +300)
    g.ax_joint.set_ylim(-300, +300)

    sns.histplot(data=df, x="x", hue="labels", palette=color_palette, ax=g.ax_marg_x, legend=False, binwidth=20,
                 element="step", fill=False)
    g.ax_marg_x.set_xlim(-300, +300)

    sns.histplot(data=df, y="y", hue="labels", palette=color_palette, ax=g.ax_marg_y, legend=False, binwidth=20,
                 element="step", fill=False)
    g.ax_marg_y.set_ylim(-300, +300)
    g.figure.show()
    g.savefig(os.path.join(act_dir, f'DsParamfMRI{suffix}_DeepJuice_project_min_to_max_X_var_{file_name}.png'))
    # save eps
    save_path = Path(f'{act_dir}/MSD_DsParamfMRI{suffix}_DeepJuice_project_min_to_max_X_var_{file_name}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids, })


    #%% do a version by taking 50% from min and 50% from max
    X_pad_min_max=[]
    for idx in range(len(X_pad)):
        X_pad_min_max.append(torch.cat((X_pad[idx], X_pad_max[idx]), dim=0))

    X_var_min_max, aligned_Xs_min_max = pt_frechet_mean(X_pad_min_max, group=grp, method=method, return_aligned_Xs=True,max_iter=max_iter,
                                         verbose=verbose,tol=tolerance)

    #%%
    pca = PCA(n_components=2)
    # do a pca on x_align_min and then transform x_align_max
    # X = np.stack([X_var_min.cpu().numpy(), x_align_max_min])
    X = X_var_min_max.cpu().numpy()
    X = pca.fit_transform(X)
    x_pca = X[:X_var_min_max.shape[0], :2]
    pca.explained_variance_ratio_
    # create labels max and min
    labels = np.concatenate(
        (np.repeat('min', X_align_pca_min_min.shape[0]), np.repeat('max', X_align_pca_min_min.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    # Define your color palette for groups
    color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}
    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    # add the image_id as a text next to the point
    # Plot the marginals
    sns.histplot(data=df, x="x", hue="labels", palette=color_palette, ax=g.ax_marg_x, legend=False, binwidth=20,
                 element="step", fill=False)
    sns.histplot(data=df, y="y", hue="labels", palette=color_palette, ax=g.ax_marg_y, legend=False, binwidth=20,
                 element="step", fill=False)
    g.fig.show()
    g.savefig(os.path.join(act_dir, f'DsParamfMRI{suffix}_DeepJuice_min_max_{file_name}.png'))
    # save eps

    save_path = Path(
        f'{act_dir}/multi_shape_distance_individual_DeepJuice_DsParametric{suffix}_min_max_{grp}_{adjust_mode}_{method}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids})
