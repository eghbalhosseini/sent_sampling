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
        device = torch.device("cpu")  # Fallback to CPU if MPS is not available
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
    extract_id = "group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=False"
    extractor_obj = extract_pool[extract_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples=25-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    # extract ev sentences
    # find location of ev sentences in sentences
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)

    #%%
    feature_map_all=[]
    for idx in range(len(extractor_obj.model_group_act)):
        X=extractor_obj.model_group_act[idx]['activations']
        X=np.asarray([a[0] for a in X])
        X=torch.tensor(X)
        column_means = torch.mean(X, dim=0)
        centered_X = X - column_means
        centered_X = centered_X/centered_X.norm()
        feature_map_all.append(centered_X.to(device))

    #%% perform mulitset distance
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
    S_full = set(np.arange(optimizer_obj.N_S))
    S = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False))
    X_pad_sample = [X_pad[i][S] for i in range(len(X_pad))]
    X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad_sample, group=grp, method=method, return_aligned_Xs=True, max_iter=steps,
                                    verbose=verbose,tol=tolerance)
    X_diff = torch.stack([X - X_var_all for X in aligned_Xs_all])
    X_var = X_diff.norm(dim=-1)
    fS = X_var.sum(dim=1).sum()

    n_out = optimizer_obj.N_S - optimizer_obj.N_s
    S_out = list(S_full.difference(set(S)))
    n_iter=2
    t=0
    changed = True
    while t < n_iter and changed:
        changed = False
        t += 1
        time_start = time.perf_counter()
        # start with a random selection from s
        si_list = np.random.choice(S, size=optimizer_obj.N_s, replace=False)
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
    X_diff=[torch.norm(torch.tensor(X)-torch.tensor(X_var_all),p=1,dim=1) for X in aligned_Xs_all]
    X_diff=torch.stack(X_diff)

    #row_means = X_diff.mean(dim=1, keepdim=True)
    # Compute the standard deviation of each row
    #row_stds = X_diff.std(dim=1, keepdim=True)
    # Compute the z-scores for each row
    #z_scores = (X_diff - row_means) / row_stds
    # compute the variance along the rows
    var_alginment = X_diff.var(dim=0)
    # rank order the variance and get the index of high and low variance
    variance_idx = torch.argsort(var_alginment)

    #%%
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples=75-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    # extract ev sentences
    # find location of ev sentences in sentences
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    low_resolution = False
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=False,save_results=False)
    S = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False))
    optimizer_obj.gpu_object_function_debug(list(variance_idx[-75:]))
    optimizer_obj.gpu_object_function_debug(list(variance_idx[:75]))
    optimizer_obj.gpu_object_function_debug(S)





