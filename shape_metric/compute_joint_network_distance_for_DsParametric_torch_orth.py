from netrep.metrics import LinearMetric
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import cross_validate
from netrep.multiset import pairwise_distances, frechet_mean, pt_frechet_mean
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib
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
    else:
        device = torch.device("cpu")  # Fallback to CPU if MPS is not available
else:
    # For non-macOS, you can default to CPU or check for CUDA (NVIDIA GPU) availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")



if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    # load act_leftout
    pre_pca=False
    act_dir='/Users/eghbalhosseini/MyData/neural_nlp_bench/activations/DsParametricfMRI/'
    model_resp_leftout=[]
    model_resp_dsparametric=[]
    model_resp_all=[]
    for model_,layer in model_layers:
        model_name = model_
        save_path = Path(f'{act_dir}/act_leftout_dsparametric_{model_name}.pkl')
        # make sure parent exist
        # load from save path
        with open(save_path, 'rb') as f:
            sent_end_layer = pkl.load(f)
        model_resp_leftout.append(sent_end_layer)
        save_path = Path(f'{act_dir}/act_all_dsparametric_{model_name}.pkl')
        # make sure parent exist
        # load from save path
        with open(save_path, 'rb') as f:
            sent_end_layer = pkl.load(f)
        model_resp_all.append(sent_end_layer['act_all'])
        ds_min_ids = sent_end_layer['min_loc']
        ds_max_ids = sent_end_layer['max_loc']
        ds_rand_ids = sent_end_layer['rand_loc']
        all_sentences= sent_end_layer['sentences']


        save_path = Path(f'{act_dir}/act_dict_dsparametric_{model_name}.pkl')
        # make sure parent exist
        with open(save_path, 'rb') as f:
            sent_end_layer = pkl.load(f)
        model_resp_dsparametric.append(sent_end_layer)
    feature_map_min_full = []
    feature_map_max_full = []
    feature_map_rand_full = []
    for idx in range(len(model_resp_dsparametric)):
        assert (model_layers[idx][0] == model_resp_dsparametric[idx]['model_name'])
        feature_map_min_full.append(model_resp_dsparametric[idx]['act_min'])
        feature_map_max_full.append(model_resp_dsparametric[idx]['act_max'])
        feature_map_rand_full.append(model_resp_dsparametric[idx]['act_rand'])

    #%%
    if pre_pca:
        feature_map_min=[]
        feature_map_max=[]
        feature_map_rand=[]
        feature_map_leftout=[]
        feature_map_all=[]
        feature_map_joint=[]
        all_var_explained=[]
        for idx in range(len(model_resp_dsparametric)):
            # do a pca on explaining 90 percent of the variance
            pca = PCA(n_components=.95)
            pca.fit(model_resp_leftout[idx])
            var_explained = pca.explained_variance_ratio_
            all_var_explained.append(var_explained)
            combined_act=np.concatenate([model_resp_dsparametric[idx]['act_min'],model_resp_dsparametric[idx]['act_rand'],model_resp_dsparametric[idx]['act_max']])
            assert(model_layers[idx][0]==model_resp_dsparametric[idx]['model_name'])
            feature_map_min.append(pca.transform(model_resp_dsparametric[idx]['act_min']))
            feature_map_max.append(pca.transform(model_resp_dsparametric[idx]['act_max']))
            feature_map_rand.append(pca.transform(model_resp_dsparametric[idx]['act_rand']))
            feature_map_leftout.append(pca.transform(model_resp_leftout[idx]))
            feature_map_joint.append(pca.transform(combined_act))
            feature_map_all.append(pca.transform(model_resp_all[idx]))
        model_group_act = {'min': feature_map_min, 'rand': feature_map_rand, 'max': feature_map_max, 'leftout': feature_map_leftout}
    else:
        feature_map_min=[]
        feature_map_max=[]
        feature_map_rand=[]
        for idx in range(len(model_resp_dsparametric)):
            assert(model_layers[idx][0]==model_resp_dsparametric[idx]['model_name'])
            # center
            X_min= model_resp_dsparametric[idx]['act_min']
            column_means = np.mean(X_min, axis=0)
            centered_X_min = X_min - column_means

            X_max= model_resp_dsparametric[idx]['act_max']
            column_means = np.mean(X_max, axis=0)
            centered_X_max = X_max - column_means

            X_rand= model_resp_dsparametric[idx]['act_rand']
            column_means = np.mean(X_rand, axis=0)
            centered_X_rand = X_rand - column_means

            feature_map_min.append(centered_X_min)
            feature_map_max.append(centered_X_max)
            feature_map_rand.append(centered_X_rand)

            #feature_map_min.append(model_resp_dsparametric[idx]['act_min'])
            #feature_map_max.append(model_resp_dsparametric[idx]['act_max'])
            #feature_map_rand.append(model_resp_dsparametric[idx]['act_rand'])
    #%% move things to torch
    feature_map_min=[torch.tensor(x).to(device) for x in feature_map_min]
    feature_map_max=[torch.tensor(x).to(device) for x in feature_map_max]

    #%% perform mulitset distance
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'streaming'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-5
    verbose = True
    file_name=f'multi_shape_distance_individual_DsParametric_{grp}_{adjust_mode}_{method}_pre_pca_{pre_pca}_torch_centered'
    save_path = Path(f'{act_dir}/{file_name}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    X=feature_map_min
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in feature_map_min]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in X]
        X_pad_max = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_max]

    X_var_min, aligned_Xs_min = pt_frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True,max_iter=50,
                                             verbose=verbose,tol=tolerance)
    X_var_max, aligned_Xs_max = frechet_mean(X_pad_max, group=grp, method=method, return_aligned_Xs=True, max_iter=50,
                                     verbose=verbose,tol=tolerance)

    # make a dictionary of aligned_Xs and x_vars
    all_X_dict={'aligned_min':aligned_Xs_min,'aligned_max':aligned_Xs_max,'var_min':X_var_min,'var_max':X_var_max}
    with open(save_path, 'wb') as f:
        pkl.dump(all_X_dict, f)



    x_align_min_min=[]
    x_align_max_min=[]
    for idx in tqdm(range(len(aligned_Xs_min))):
        T=align(X_var_min,aligned_Xs_min[idx],group=grp)
        x_align_min_min.append(T.dot(X_pad[idx].T).T)
        x_align_max_min.append(T.dot(X_pad_max[idx].T).T)

    x_align_min_max=[]
    x_align_max_max=[]
    for idx in tqdm(range(len(aligned_Xs_max))):
        T=align(X_var_max,aligned_Xs_max[idx],group=grp)
        x_align_min_max.append(T.dot(X_pad[idx].T).T)
        x_align_max_max.append(T.dot(X_pad_max[idx].T).T)

    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in aligned_Xs_min]), 'correlation'))
    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in x_align_min_min]), 'correlation'))
    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in x_align_min_max]), 'correlation'))
    # for max
    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in aligned_Xs_max]), 'correlation'))
    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in x_align_max_min]), 'correlation'))
    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in x_align_max_max]), 'correlation'))

    # do it for feature_map_min_full and feature_map_max_full
    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in feature_map_min_full]), 'correlation'))

    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in feature_map_max_full]), 'correlation'))

    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in feature_map_min]), 'correlation'))
    2 - np.mean(pdist(np.stack([pdist(x, 'correlation') for x in feature_map_max]), 'correlation'))

    x_align_min_min=np.stack(x_align_min_min)
    x_align_min_min = np.mean(x_align_min_min, axis=0)
    x_align_max_min = np.stack(x_align_max_min)
    x_align_max_min = np.mean(x_align_max_min, axis=0)

    x_align_min_max=np.stack(x_align_min_max)
    x_align_min_max = np.mean(x_align_min_max, axis=0)
    x_align_max_max = np.stack(x_align_max_max)
    x_align_max_max = np.mean(x_align_max_max, axis=0)
    # average over dimension 0

    pca = PCA(n_components=2)
# do a pca on x_align_min and then transform x_align_max
    X=np.stack([X_var_min,x_align_max_min])
    X=np.concatenate(X,axis=0)
    X=pca.fit_transform(X)
    X_align_pca_min_min=X[:X_var_min.shape[0],:]
    X_align_pca_max_min=X[X_var_min.shape[0]:,:]

    #
    X=np.stack([x_align_min_max,X_var_max])
    X=np.concatenate(X,axis=0)
    X=pca.fit_transform(X)
    X_align_pca_min_max=X[:x_align_min_max.shape[0],:]
    X_align_pca_max_max=X[x_align_min_max.shape[0]:,:]



    #%%
    sorted_image_ids=np.stack([np.arange(model_resp_dsparametric[0]['sent_min'].shape[0]), np.arange(model_resp_dsparametric[0]['sent_max'].shape[0])]).flatten()
    sent_min=list(model_resp_dsparametric[0]['sent_min'])
    sent_max=list(model_resp_dsparametric[0]['sent_max'])
    sent_=np.stack([sent_min, sent_max]).flatten()
    x_pca = np.concatenate((X_align_pca_min_min[:,:2],X_align_pca_max_min[:,:2] ), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_align_pca_min_min.shape[0]), np.repeat('max', X_align_pca_min_min.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    df['sent'] = sent_
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
    g.savefig(os.path.join(act_dir, f'DsParamfMRI_Align_max_to_min_{file_name}.png'))
    # save eps

    save_path = Path(f'{act_dir}/multi_shape_distance_individual_DsParametric_{grp}_{adjust_mode}_{method}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids, 'sent': sent_})


    #%%

    x_pca = np.concatenate((X_align_pca_min_max[:,:2],X_align_pca_max_max[:,:2] ), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_align_pca_min_max.shape[0]), np.repeat('max', X_align_pca_max_max.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    df['sent'] = sent_
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
    g.savefig(os.path.join(act_dir, f'DsParamefMRI_Align_min_to_max_{file_name}.png'))


    #%%
    column_means = np.mean(X_var_min, axis=0)
    centered_X_var_min = X_var_min - column_means
    column_means = np.mean(X_var_max, axis=0)
    centered_X_var_max = X_var_max - column_means

    X_pca = pca.fit_transform(centered_X_var_min)
    X_max=pca.transform(centered_X_var_max)
    X_align_pca_min = X_pca
    X_align_pca_max = X_max

    sorted_image_ids = np.stack([np.arange(model_resp_dsparametric[0]['sent_min'].shape[0]),
                                 np.arange(model_resp_dsparametric[0]['sent_max'].shape[0])]).flatten()
    sent_min = list(model_resp_dsparametric[0]['sent_min'])
    sent_max = list(model_resp_dsparametric[0]['sent_max'])
    sent_ = np.stack([sent_min, sent_max]).flatten()
    x_pca = np.concatenate((X_align_pca_min[:, :2], X_align_pca_max[:, :2]), axis=0)
    # create labels max and min
    labels = np.concatenate(
        (np.repeat('min', X_align_pca_max.shape[0]), np.repeat('max', X_align_pca_max.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    df['sent'] = sent_
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

    g.savefig(os.path.join(act_dir, f'DsParamfMRI_project_max_to_min_X_var_{file_name}.png'))
    # save eps
    save_path = Path(f'{act_dir}/MSD_DsParamfMRI_project_max_to_min_X_var_{file_name}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids, 'sent': sent_})

    #%%

    X_align_pca_max = pca.fit_transform(centered_X_var_max)
    X_align_pca_min=pca.transform(centered_X_var_min)


    sorted_image_ids = np.stack([np.arange(model_resp_dsparametric[0]['sent_min'].shape[0]),
                                 np.arange(model_resp_dsparametric[0]['sent_max'].shape[0])]).flatten()
    sent_min = list(model_resp_dsparametric[0]['sent_min'])
    sent_max = list(model_resp_dsparametric[0]['sent_max'])
    sent_ = np.stack([sent_min, sent_max]).flatten()
    x_pca = np.concatenate((X_align_pca_min[:, :2], X_align_pca_max[:, :2]), axis=0)
    # create labels max and min
    labels = np.concatenate(
        (np.repeat('min', X_align_pca_max.shape[0]), np.repeat('max', X_align_pca_max.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    df['sent'] = sent_
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
    g.savefig(os.path.join(act_dir, f'DsParamfMRI_project_min_to_max_X_var_{file_name}.png'))
    # save eps
    save_path = Path(f'{act_dir}/MSD_DsParamfMRI_project_min_to_max_X_var_{file_name}.mat')
    # # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids, 'sent': sent_})


    #%% do a version by taking 50% from min and 50% from max
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-5
    verbose = True
    file_name=f'multi_shape_distance_min_max_DsParametric_{grp}_{adjust_mode}_{method}_pre_pca_{pre_pca}_centered'
    save_path = Path(f'{act_dir}/{file_name}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    X_min=feature_map_min
    X_max=feature_map_max
    # randomly select 40 sentence from X_min
    np.random.seed(0)
    rand_ids=np.random.choice(X_min[0].shape[0], 40, replace=False)
    X_min_rand=[x[rand_ids] for x in X_min]
    X_max_rand=[x[rand_ids] for x in X_max]
    # for each one combine X_min and X_max
    X_min_max=[np.concatenate([x_min,x_max],axis=0) for x_min,x_max in zip(X_min_rand,X_max_rand)]
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in X_min_max]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in X_min_max]

    X_var_min_max, aligned_Xs_min_max = frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True,max_iter=50,
                                             verbose=verbose,tol=tolerance)


    # make a dictionary of aligned_Xs and x_vars
    all_X_dict={'aligned_min_max':aligned_Xs_min_max,'var_min_max':X_var_min_max}
    with open(save_path, 'wb') as f:
        pkl.dump(all_X_dict, f)

    # do a pca on X_var_min_max
    pca = PCA(n_components=2)
    column_means = np.mean(X_var_min_max, axis=0)
    centered_X_var_min_max = X_var_min_max - column_means
    X_pca = pca.fit_transform(centered_X_var_min_max)

    sorted_image_ids = np.stack([np.arange(model_resp_dsparametric[0]['sent_min'].shape[0])[rand_ids],
                                 np.arange(model_resp_dsparametric[0]['sent_max'].shape[0])[rand_ids]]).flatten()
    sent_min = [list(model_resp_dsparametric[0]['sent_min'])[int(x)] for x in rand_ids]
    sent_max = [list(model_resp_dsparametric[0]['sent_max'])[int(x)] for x in rand_ids]

    sent_ = np.stack([sent_min, sent_max]).flatten()

    # create labels max and min
    labels = np.concatenate(
        (np.repeat('min', rand_ids.shape[0]), np.repeat('max', rand_ids.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(X_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    df['sent'] = sent_
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
    g.savefig(os.path.join(act_dir, f'DsParamfMRI_project_min_max_X_var_{file_name}.png'))