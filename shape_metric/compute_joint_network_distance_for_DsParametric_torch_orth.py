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
    #act_dir='/Users/eghbalhosseini/MyData/neural_nlp_bench/activations/DsParametricfMRI/'

    act_dir = '/om2/user/ehoseini/MyData/neural_nlp_bench/activations/DsParametricfMRI/'
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
    ##% do it for full
    feature_map_all=[]
    for idx in range(len(model_resp_all)):
        X=model_resp_all[idx]
        column_means = np.mean(X, axis=0)
        centered_X = X - column_means
        feature_map_all.append(torch.tensor(centered_X).to(device))
    #%% perform mulitset distance
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'streaming'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-5
    steps= 2000
    verbose = True
    file_name=f'multi_shape_distance_all_DsParametric_{grp}_{adjust_mode}_{method}_pre_pca_{pre_pca}_torch_centered_steps_{steps}'
    save_path = Path(f'{act_dir}/{file_name}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    X=feature_map_min
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in feature_map_min]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in X]
        X_pad_max = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_max]
        X_pad_all = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]

    #X_var_min, aligned_Xs_min = pt_frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True,max_iter=50,
    #                                         verbose=verbose,tol=tolerance)
    #X_var_max, aligned_Xs_max = pt_frechet_mean(X_pad_max, group=grp, method=method, return_aligned_Xs=True, max_iter=50,
    #                                 verbose=verbose,tol=tolerance)

    X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad_all, group=grp, method=method, return_aligned_Xs=True, max_iter=steps,
                                        verbose=verbose,tol=tolerance)
    # make a dictionary of aligned_Xs and x_vars
    all_X_dict={'aligned_all':aligned_Xs_all,'var_al':X_var_all}
    with open(save_path, 'wb') as f:
        pkl.dump(all_X_dict, f)



