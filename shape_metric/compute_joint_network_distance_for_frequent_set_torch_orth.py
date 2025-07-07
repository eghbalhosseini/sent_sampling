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

from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project
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



if __name__ == '__main__':
    # compute the simliarty vs score
    dataset_id='coca_preprocessed_all_clean_100K_sample_1_estim_ds_min'
    extract_id = f'group=best_performing_pereira_1-dataset={dataset_id}_textNoPeriod-activation-bench=None-ave=False'
    optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset(splits=20)
    ext_obj()
    optim_obj = optim_pool[optim_id]()
    optim_obj.load_extractor(ext_obj)

    ##% do it for full
    feature_map_all=[]
    for idx, act_dict in tqdm(enumerate(optim_obj.activations)):
        X = torch.tensor(
            [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']])
        column_means = torch.mean(X, axis=0)
        centered_X = X - column_means
        feature_map_all.append(centered_X.to(device))
    #%% perform mulitset distance
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'streaming'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-5
    steps= 2000
    verbose = True
    act_dir = '/om2/user/ehoseini/MyData/neural_nlp_bench/activations/DsParametricfMRI/'
    file_name=f'multi_shape_distance_{dataset_id}_{grp}_{adjust_mode}_{method}_torch_centered_steps_{steps}'
    save_path = Path(f'{act_dir}/{file_name}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in feature_map_all]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad_all = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in
                     feature_map_all]

    if save_path.exists():
        with open(save_path, 'rb') as f:
            all_X_dict = pkl.load(f)
        aligned_Xs_all=all_X_dict['aligned_all']
        X_var_all=all_X_dict['var_al']
    else:
        X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad_all, group=grp, method=method, return_aligned_Xs=True, max_iter=steps,
                                            verbose=verbose,tol=tolerance)
        # make a dictionary of aligned_Xs and x_vars
        all_X_dict={'aligned_all':aligned_Xs_all,'var_al':X_var_all}
        with open(save_path, 'wb') as f:
            pkl.dump(all_X_dict, f)




    #%%  get the distance matrix
    feature_map_min_full = []
    feature_map_max_full = []
    feature_map_rand_full = []

    for model_ in ext_obj.model_spec:
        save_path = Path(f'{act_dir}/act_dict_dsparametric_{model_}.pkl')
        # make sure parent exist
        with open(save_path, 'rb') as f:
            sent_end_layer = pkl.load(f)
        assert (model_ == sent_end_layer['model_name'])
        X_min = sent_end_layer['act_min']
        column_means = np.mean(X_min, axis=0)
        centered_X_min = X_min - column_means

        X_max = sent_end_layer['act_max']
        column_means = np.mean(X_max, axis=0)
        centered_X_max = X_max - column_means

        X_rand = sent_end_layer['act_rand']
        column_means = np.mean(X_rand, axis=0)
        centered_X_rand = X_rand - column_means

        feature_map_min_full.append(centered_X_min)
        feature_map_max_full.append(centered_X_max)
        feature_map_rand_full.append(centered_X_rand)

    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in feature_map_min_full]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        feature_map_min_full = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in feature_map_min_full]
        feature_map_max_full = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in feature_map_max_full]

    #%%
    x_align_min=[]
    x_align_max=[]
    for idx,X_pad in enumerate(X_pad_all):
        # first compute the alginment matrix
        T = pt_align(X_var_all, X_pad, group=grp)
        x_align_min.append(torch.matmul(torch.tensor(feature_map_min_full[idx],device=device),T))
        x_align_max.append(torch.matmul(torch.tensor(feature_map_max_full[idx],device=device),T))

    x_var_min= torch.mean(torch.stack(x_align_min),dim=0)
    x_var_max= torch.mean(torch.stack(x_align_max),dim=0)
    x_var_min=x_var_min.cpu().detach().numpy()
    x_var_max=x_var_max.cpu().detach().numpy()
    # do a joint pca
    pca = PCA(n_components=2)
    pca.fit(X_var_all.cpu().detach().numpy())
    x_var_min = pca.transform(x_var_min)
    x_var_max = pca.transform(x_var_max)
    x_var_all = pca.transform(X_var_all.cpu().detach().numpy())
    #%% plot them

    sorted_image_ids = np.stack([np.arange(sent_end_layer['sent_min'].shape[0]),
                                 np.arange(sent_end_layer['sent_max'].shape[0])]).flatten()
    sent_min = list(sent_end_layer['sent_min'])
    sent_max = list(sent_end_layer['sent_max'])
    sent_ = np.stack([sent_min, sent_max]).flatten()
    x_pca = np.concatenate((x_var_min[:, :2], x_var_max[:, :2]), axis=0)
    # create labels max and min
    labels = np.concatenate(
        (np.repeat('min', x_var_min.shape[0]), np.repeat('max', x_var_min.shape[0])), axis=0)
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
    # do a scatter of x_var_all
    sns.scatterplot(x=x_var_all[:,0],y=x_var_all[:,1], color='g', ax=g.ax_joint)
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

