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
from sent_sampling.utils.opt_exp_design import swap, LOGGER
from netrep.utils import align, pt_align
import time
from scipy.spatial.distance import pdist
from scipy.io import savemat
import torch
import torch.nn.functional as F
#matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 3,'font.weight':'normal'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
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

import multiprocessing
import os
import seaborn as sns
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
os.environ['OMP_NUM_THREADS'] = '4'
import matplotlib.pyplot as plt
import platform
import sys
sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
from torch.nn.functional import gumbel_softmax

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
from glob import glob
import pickle

if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    # load act_leftout
    # load act_leftout
    selected_models = ['torchvision_alexnet_imagenet1k_v1',
                       'torchvision_regnet_x_800mf_imagenet1k_v2',
                       'openclip_vit_b_32_laion2b_e16',
                       'timm_swinv2_cr_tiny_ns_224',
                       'torchvision_efficientnet_b1_imagenet1k_v2',
                       'timm_convnext_large_in22k',
                       ]
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
        X = feature_map_all[idx]
        X = torch.tensor(X).to(torch.float64)
        column_means = torch.mean(X, dim=0)
        centered_X = X - column_means
        feature_map_all[idx] = centered_X.to(device)

    #%%
    #% perform mulitset distance
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-5
    steps= 2000
    verbose = True
    # warmstart= make it to previous X_bar
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in feature_map_all]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad = [F.pad(x, pad=(0, max_shape - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]
    else:
        X_pad = feature_map_all

    #%%
    N_S=feature_map_all[0].shape[0]
    n_s=80
    S_full = set(np.arange(N_S))
    S = list(np.random.choice(N_S, n_s, replace=False))
    X_pad_sample = [X_pad[i][S] for i in range(len(X_pad))]
    X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad_sample, group=grp, method=method, return_aligned_Xs=True, max_iter=steps,
                                    verbose=verbose,tol=tolerance)
    X_diff = torch.stack([X - X_var_all for X in aligned_Xs_all])
    X_var = X_diff.norm(dim=-1)
    fS = X_var.sum(dim=1).sum()

    n_out = N_S - n_s
    S_out = list(S_full.difference(set(S)))
    n_iter=2
    t=0
    changed = True
    while t < n_iter and changed:
        changed = False
        t += 1
        time_start = time.perf_counter()
        # start with a random selection from s
        si_list = np.random.choice(S, size=n_s, replace=False)
        for si_idx, si in enumerate(S):
            so_list = np.random.choice(S_out, size=n_out, replace=False)
            so_idx = 0
            keep_swapping = True
            while keep_swapping:
                S_test = swap(S, si, so_list[so_idx])
                # time_start = time.perf_counter()
                X_pad_swap = [X_pad[i][S_test] for i in range(len(X_pad))]
                X_var_all_swap, aligned_Xs_all_swap = pt_frechet_mean(X_pad_sample, group=grp, method=method,warmstart=X_var_all
                                                            ,return_aligned_Xs=True, max_iter=steps,
                                                            verbose=False, tol=tolerance)
                X_diff_swap = torch.stack([X - X_var_all_swap for X in aligned_Xs_all_swap])
                X_var_swap = X_diff_swap.norm(dim=-1)
                f_swap = X_var_swap.sum(dim=1).sum()
                # time_elapsed = (time.perf_counter() - time_start)
                if f_swap < fS:
                    fS = f_swap
                    fS_loop = fS
                    S = S_test
                    S_out = swap(S_out, so_list[so_idx], si)
                    LOGGER.info('[%d/%d] [t = %d] id = %d, %d to %d after %d swaps,  f(S) = %.10f' % (
                    1, 1, t, si_idx, si, so_list[so_idx], so_idx, fS))
                    keep_swapping = False
                    changed = True
                else:
                    keep_swapping = True
                    so_idx += 1
                if so_idx == len(so_list):
                    LOGGER.info('[%d/%d] [t = %d] id = %d f(s) %d didnt change after all %d swaps,  f(S) = %.10f' % (
                     1, 1, t, si_idx, si, so_idx, fS))
                    keep_swapping = False



        X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad_sample, group=grp, method=method, return_aligned_Xs=True,
                                                    max_iter=steps,
                                                    verbose=verbose, tol=tolerance)

    ## do a final pca
    pca = PCA(n_components=2)
    # do a pca on x_align_min and then transform x_align_max
    #X = pca.fit_transform(X_var_all)
    X = pca.fit_transform(aligned_Xs_all[6])

    # get the variance explained
    print(pca.explained_variance_ratio_)

    #%%
    probs = torch.nn.Parameter(torch.ones(N_S) / torch.ones(N_S).sum(), requires_grad=True)
    optimizer = torch.optim.Adam([probs], lr=0.001)

    for _ in tqdm(range(10)):  # Number of optimization steps
        optimizer.zero_grad()

        # Sample 20 indices according to the probabilities
        logits = torch.log(probs)
        gumbel_samples = gumbel_softmax(logits, tau=1.0, hard=True)
        S = torch.multinomial(gumbel_samples, n_s, replacement=False)
        X_pad_sample = [X_pad[i][S] for i in range(len(X_pad))]
        # Compute the objective
        X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad_sample, group=grp, method=method, return_aligned_Xs=True,
                                                    max_iter=steps,
                                                    verbose=False, tol=tolerance)
        aligned_Xs_all=torch.stack(aligned_Xs_all).requires_grad_()
        X_diff = aligned_Xs_all - X_var_all
        X_var = X_diff.norm(dim=-1)
        loss = X_var.sum(dim=1).sum()
        # print loss
        print('loss:',loss.item())


        # Ensure the loss requires grad
        # Compute gradients and update probabilities
        loss.backward()
        optimizer.step()

        # Ensure probabilities remain valid
        with torch.no_grad():
            probs.clamp_(0, 1)
            probs.div_(probs.sum())


