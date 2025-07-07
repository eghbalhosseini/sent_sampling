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
        feature_map_all[idx]=centered_X.to(device)

    #%% perform mulitset distance
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-7
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

    X_var_all, aligned_Xs_all = pt_frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True, max_iter=steps,
                                            verbose=verbose,tol=tolerance)


    #%%
    n_samples = 80
    X_diff=[X-X_var_all for X in aligned_Xs_all]
    X_diff=torch.stack(X_diff)

    #X_diff[:,0,:].norm(dim=1)
    X_var=X_diff.norm(dim=-1)
    # compute the variance along the rows
    var_alginment = X_var.norm(dim=0)
    # rank order the variance and get the index of high and low variance
    variance_idx = torch.argsort(var_alginment).tolist()
    # take 80 samples from the first 100 and last 100
    min_var_idx=sorted(variance_idx[:80])
    max_var_idx=sorted(variance_idx[-80:])
    rand_var_idx=sorted(np.random.choice(variance_idx,80,replace=False))




    #%%
    #benchmark_ = NSDSampleBenchmark(image_samples=sorted(ds_min_image_ids))
    benchmark_ = NSDBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/')
    benchmark_min=NSDSampleBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/',image_samples=min_var_idx)
    benchmark_max=NSDSampleBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/',image_samples=max_var_idx)
    benchmark_rand=NSDSampleBenchmark(path_dir='/om2/user/ehoseini/MyData/DeepJuice/nsd_data/',image_samples=rand_var_idx)
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    x_fmri = (convert_to_tensor(benchmark_.response_data.to_numpy())
         .to(dtype=torch.float64, device=device)).T
    x_fmri_min = (convert_to_tensor(benchmark_min.response_data.to_numpy())
            .to(dtype=torch.float64, device=device)).T
    x_fmri_max = (convert_to_tensor(benchmark_max.response_data.to_numpy())
            .to(dtype=torch.float64, device=device)).T
    x_fmri_rand = (convert_to_tensor(benchmark_rand.response_data.to_numpy())
            .to(dtype=torch.float64, device=device)).T

#%%

    for image_id in min_var_idx:
        image_path=image_paths[image_id]
        # add a ds_min infront of the image_id in the destination hwne saving it
        image_id=image_path.split('/')[-1]
        shutil.copy(image_path, ds_min_image_path)
        # change the name of the image to ds_min_image_id
        os.rename(f'{ds_min_image_path}{image_id}', f'{ds_min_image_path}ds_min_{image_id}')
        #shutil.copy(image_path, ds_min_image_path)
    for image_id in ds_max_image_ids:
        image_path=image_paths[image_id]
        image_id = image_path.split('/')[-1]
        shutil.copy(image_path, ds_max_image_path)
        os.rename(f'{ds_max_image_path}/{image_id}', f'{ds_max_image_path}ds_max_{image_id}')
    for image_id in ds_rand_image_ids:
        image_path=image_paths[image_id]
        image_id = image_path.split('/')[-1]
        shutil.copy(image_path, ds_rand_image_path)
        os.rename(f'{ds_rand_image_path}{image_id}', f'{ds_rand_image_path}ds_rand_{image_id}')


    fmri_roi_sub_x=[x_fmri[:,indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_min=[x_fmri_min[:,indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_max=[x_fmri_max[:,indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_rand=[x_fmri_rand[:,indx] for indx in roi_indices['OTC'].values()]
    #
    all_model_pro_dist=[]
    all_model_bures_dist=[]
    for modl_idx in tqdm(range(len(X_pad))):
        y_model=X_pad[modl_idx].to(torch.float64)
        y_model_min=X_pad[modl_idx][min_var_idx,:].to(torch.float64)
        y_model_max=X_pad[modl_idx][max_var_idx,:].to(torch.float64)
        y_model_rand=X_pad[modl_idx][rand_var_idx,:].to(torch.float64)
        modl_pro_dist=[]
        modl_bures_dist=[]
        for sub_idx in range(len(fmri_roi_sub_x)):
            x_sub = fmri_roi_sub_x[sub_idx]
            x_sub_min = fmri_roi_sub_x_min[sub_idx]
            x_sub_max = fmri_roi_sub_x_max[sub_idx]
            x_sub_rand = fmri_roi_sub_x_rand[sub_idx]
            #
            x_sub -= x_sub.mean(dim=0)
            x_sub_min -= x_sub_min.mean(dim=0)
            x_sub_max -= x_sub_max.mean(dim=0)
            x_sub_rand -= x_sub_rand.mean(dim=0)
            ###
            x_sub_all = [x_sub_min, x_sub_rand, x_sub_max]

            pro_dists=[]
            bures_dists=[]
            for idx,y in enumerate([y_model_min,y_model_rand,y_model_max]):
                x=x_sub_all[idx]
                pro_ = procrustes_dist(x, y)
                bures_ = bures_dist_epsilon(x, y)
                if bures_ is None:
                    # replace it with nan
                    bures_ = torch.nan
                pro_dists.append(pro_)
                bures_dists.append(bures_)
            modl_pro_dist.append(torch.tensor(pro_dists))
            modl_bures_dist.append(torch.tensor(bures_dists))
        all_model_pro_dist.append(modl_pro_dist)
        all_model_bures_dist.append(modl_bures_dist)

    # save the results to a file
    save_file = '/om2/user/ehoseini/MyData/DeepJuice/NSD_procrustes_bures_dist.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump([all_model_pro_dist, all_model_bures_dist], f)

#%% only procrustrates
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_min = [x_fmri_min[:, indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_max = [x_fmri_max[:, indx] for indx in roi_indices['OTC'].values()]
    fmri_roi_sub_x_rand = [x_fmri_rand[:, indx] for indx in roi_indices['OTC'].values()]
    #
    all_model_pro_dist = []
    for modl_idx in tqdm(range(len(X_pad))):
        y_model = X_pad[modl_idx].to(torch.float64).to(device)
        y_model_min = X_pad[modl_idx][min_var_idx, :].to(torch.float64).to(device)
        y_model_max = X_pad[modl_idx][max_var_idx, :].to(torch.float64).to(device)
        y_model_rand = X_pad[modl_idx][rand_var_idx, :].to(torch.float64).to(device)
        modl_pro_dist = []
        modl_bures_dist = []
        for sub_idx in tqdm(range(len(fmri_roi_sub_x))):
            x_sub = fmri_roi_sub_x[sub_idx]
            x_sub_min = fmri_roi_sub_x_min[sub_idx]
            x_sub_max = fmri_roi_sub_x_max[sub_idx]
            x_sub_rand = fmri_roi_sub_x_rand[sub_idx]
            #
            x_sub -= x_sub.mean(dim=0)
            x_sub_min -= x_sub_min.mean(dim=0)
            x_sub_max -= x_sub_max.mean(dim=0)
            x_sub_rand -= x_sub_rand.mean(dim=0)
            ###
            x_sub_all = [x_sub_min, x_sub_rand, x_sub_max,x_sub]
            pro_dists = []
            for idx, y in enumerate([y_model_min, y_model_rand, y_model_max,y_model]):
                x = x_sub_all[idx]
                pro_ = procrustes_dist(x, y)
                pro_dists.append(pro_)
            modl_pro_dist.append(torch.tensor(pro_dists))
        all_model_pro_dist.append(modl_pro_dist)

    # save the results to a file
    save_file = '/om2/user/ehoseini/MyData/DeepJuice/NSD_procrustes_dist_OTC.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(all_model_pro_dist, f)

    #%% load the data
    save_file = '/om2/user/ehoseini/MyData/DeepJuice/NSD_procrustes_dist_OTC.pkl'
    with open(save_file, 'rb') as f:
        all_model_pro_dist = pickle.load(f)

    num_subject=len(all_model_pro_dist[0])
    procrustes_mean=torch.stack([torch.stack(x).mean(dim=0) for x in all_model_pro_dist])
    #bures_mean=torch.stack([torch.stack(x).mean(dim=0) for x in all_model_bures_dist])
    procrustes_std=torch.stack([torch.stack(x).std(dim=0)/np.sqrt(num_subject) for x in all_model_pro_dist])
    bures_std=torch.stack([torch.stack(x).std(dim=0)/np.sqrt(num_subject) for x in all_model_bures_dist])

    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
                np.divide((255, 255, 255), 256)]

    models_scores = torch.stack([procrustes_mean,procrustes_std],dim=-1)
    #models_scores=torch.stack([bures_mean,bures_std],dim=-1)
    width = 0.1  # the width of the bars
    fig = plt.figure(figsize=(11, 8))
    # fig_length = 0.055 * len(models_scores)
    #ax = plt.axes((.1, .4, .35, .35))
    x = np.arange(models_scores.shape[0])

    model_name = models_sh
    # create 6 sbuplots and each one plot min, rand, and max
    for kk in range(models_scores.shape[0]):
        ax = plt.subplot(2, 3, kk + 1)
        y = models_scores[kk, :, 0]
        y_err = models_scores[kk, :, 1]
        y=y / torch.tensor([80, 80, 80, 1000])
        y_err = y_err / torch.tensor([80, 80, 80, 1000])

        x=np.asarray([0])
        rects2 = ax.bar(x - 0.3, y[0], width, label='min', color=colors[0], linewidth=.5,
                        edgecolor='k')
        ax.errorbar(x - 0.3, y[0], yerr=y_err[0], linestyle='', color='k')
        # plot the second item
        rects2 = ax.bar(x-.2, y[1], width, label='rand', color=colors[1], linewidth=.5, edgecolor='k')
        ax.errorbar(x-.2, y[1], yerr=y_err[1], linestyle='', color='k')
        #
        rects3 = ax.bar(x -.1, y[2], width, label='max', color=colors[2], linewidth=.5, edgecolor='k')
        ax.errorbar(x -.1, y[2], yerr=y_err[2], linestyle='', color='k')
        #
        rects3 = ax.bar(x , y[3], width, label='all', color=colors[3], linewidth=.5, edgecolor='k')
        ax.errorbar(x , y[3], yerr=y_err[3], linestyle='', color='k')

        ax.axhline(y=0, color='k', linestyle='-')
        ax.set_ylabel('procrustes distance/#samples')
        ax.set_title(f'{model_name[kk]}')
        ax.set_xticks(x)
        if kk ==0:
            ax.legend()
            ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #ax.set_xticklabels(['min', 'rand', 'max', 'all'])
        #ax.set_ylim((-.1, .1))
        #ax.legend()
    plt.tight_layout()
    fig.show()
    # save figure
    fig.savefig('/om2/user/ehoseini/MyData/DeepJuice/NSD_procrustes_dist_OTC.pdf',bbox_inches='tight')
    # rects2 = ax.bar(x - 0.25, models_scores[:, 0, 0], width, label='min', color=colors[0], linewidth=.5, edgecolor='k')
    # ax.errorbar(x - 0.25, models_scores[:, 0, 0], yerr=models_scores[:, 0, 1], linestyle='', color='k')
    # # plot the second item
    # rects2 = ax.bar(x, models_scores[:, 1, 0], width, label='rand', color=colors[1], linewidth=.5, edgecolor='k')
    # ax.errorbar(x, models_scores[:, 1, 0], yerr=models_scores[:, 1, 1], linestyle='', color='k')
    # # plot the third item
    # rects3 = ax.bar(x + 0.25, models_scores[:, 2, 0], width, label='max', color=colors[2], linewidth=.5, edgecolor='k')
    # ax.errorbar(x + 0.25, models_scores[:, 2, 0], yerr=models_scores[:, 2, 1], linestyle='', color='k')
    #
    # rects3 = ax.bar(x + 0.35, models_scores[:, 3, 0], width, label='max', color=colors[3], linewidth=.5, edgecolor='k')
    # ax.errorbar(x + 0.35, models_scores[:, 3, 0], yerr=models_scores[:, 3, 1], linestyle='', color='k')
    #
    # # ax.errorbar(x  , models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    # ax.axhline(y=0, color='k', linestyle='-')
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('procrustes distance')

    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=45)
    #ax.set_ylim((-.175, 0.175))
    ax.set_xlim((-.5, 5.5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()

    #%% do it for all regions
    rois=list(roi_indices.keys())
    all_roi_pro_dist=dict()
    for roi in tqdm(rois):
        fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices[roi].values()]
        fmri_roi_sub_x_min = [x_fmri_min[:, indx] for indx in roi_indices[roi].values()]
        fmri_roi_sub_x_max = [x_fmri_max[:, indx] for indx in roi_indices[roi].values()]
        fmri_roi_sub_x_rand = [x_fmri_rand[:, indx] for indx in roi_indices[roi].values()]
        #
        all_model_pro_dist = []
        for modl_idx in tqdm(range(len(X_pad))):
            y_model = X_pad[modl_idx].to(torch.float64).to(device)
            y_model_min = X_pad[modl_idx][min_var_idx, :].to(torch.float64).to(device)
            y_model_max = X_pad[modl_idx][max_var_idx, :].to(torch.float64).to(device)
            y_model_rand = X_pad[modl_idx][rand_var_idx, :].to(torch.float64).to(device)
            modl_pro_dist = []
            modl_bures_dist = []
            for sub_idx in tqdm(range(len(fmri_roi_sub_x))):
                x_sub = fmri_roi_sub_x[sub_idx]
                x_sub_min = fmri_roi_sub_x_min[sub_idx]
                x_sub_max = fmri_roi_sub_x_max[sub_idx]
                x_sub_rand = fmri_roi_sub_x_rand[sub_idx]
                #
                x_sub -= x_sub.mean(dim=0)
                x_sub_min -= x_sub_min.mean(dim=0)
                x_sub_max -= x_sub_max.mean(dim=0)
                x_sub_rand -= x_sub_rand.mean(dim=0)
                ###
                x_sub_all = [x_sub_min, x_sub_rand, x_sub_max, x_sub]
                pro_dists = []
                for idx, y in enumerate([y_model_min, y_model_rand, y_model_max, y_model]):
                    x = x_sub_all[idx]
                    pro_ = procrustes_dist(x, y)
                    pro_dists.append(pro_)
                modl_pro_dist.append(torch.tensor(pro_dists))
            all_model_pro_dist.append(modl_pro_dist)

        all_roi_pro_dist[roi]=all_model_pro_dist
        # save the results to a file
    save_file = f'/om2/user/ehoseini/MyData/DeepJuice/NSD_procrustes_dist_all_rois.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(all_roi_pro_dist, f)


    #%%
    save_file = '/om2/user/ehoseini/MyData/DeepJuice/NSD_procrustes_dist_all_rois.pkl'
    with open(save_file, 'rb') as f:
        all_roi_pro_dist = pickle.load(f)


    num_subject = len(all_roi_pro_dist['OTC'][0])
    for roi in list(all_roi_pro_dist.keys()):
        all_model_pro_dist = all_roi_pro_dist[roi]
        procrustes_mean = torch.stack([torch.stack(x).mean(dim=0) for x in all_model_pro_dist])
        # bures_mean=torch.stack([torch.stack(x).mean(dim=0) for x in all_model_bures_dist])
        procrustes_std = torch.stack([torch.stack(x).std(dim=0) / np.sqrt(num_subject) for x in all_model_pro_dist])
        bures_std = torch.stack([torch.stack(x).std(dim=0) / np.sqrt(num_subject) for x in all_model_bures_dist])

        colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
                  np.divide((255, 255, 255), 256)]

        models_scores = torch.stack([procrustes_mean, procrustes_std], dim=-1)
        # models_scores=torch.stack([bures_mean,bures_std],dim=-1)
        width = 0.1  # the width of the bars
        fig = plt.figure(figsize=(11, 8))
        # fig_length = 0.055 * len(models_scores)
        # ax = plt.axes((.1, .4, .35, .35))
        x = np.arange(models_scores.shape[0])

        model_name = models_sh
        # create 6 sbuplots and each one plot min, rand, and max
        for kk in range(models_scores.shape[0]):
            ax = plt.subplot(2, 3, kk + 1)
            y = models_scores[kk, :, 0]
            y_err = models_scores[kk, :, 1]
            y = y / torch.tensor([80, 80, 80, 1000])
            y_err = y_err / torch.tensor([80, 80, 80, 1000])
            x = np.asarray([0])
            rects2 = ax.bar(x - 0.3, y[0], width, label='min', color=colors[0], linewidth=.5,edgecolor='k')
            ax.errorbar(x - 0.3, y[0], yerr=y_err[0], linestyle='', color='k')
            # plot the second item
            rects2 = ax.bar(x - .2, y[1], width, label='rand', color=colors[1], linewidth=.5, edgecolor='k')
            ax.errorbar(x - .2, y[1], yerr=y_err[1], linestyle='', color='k')
            #
            rects3 = ax.bar(x - .1, y[2], width, label='max', color=colors[2], linewidth=.5, edgecolor='k')
            ax.errorbar(x - .1, y[2], yerr=y_err[2], linestyle='', color='k')
            #
            rects3 = ax.bar(x, y[3], width, label='all', color=colors[3], linewidth=.5, edgecolor='k')
            ax.errorbar(x, y[3], yerr=y_err[3], linestyle='', color='k')
            ax.axhline(y=0, color='k', linestyle='-')
            ax.set_ylabel('procrustes distance/#samples')
            ax.set_title(f'{model_name[kk]}')
            ax.set_xticks(x)
            if kk == 0:
                ax.legend()
                ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.tight_layout()
        # add a supertitle with ROI name
        fig.suptitle(f'{roi}',fontsize=16)
        #fig.show()
        # save figure
        fig.savefig(f'/om2/user/ehoseini/MyData/DeepJuice/NSD_procrustes_dist_{roi}.pdf', bbox_inches='tight')

    #%%
    feature_map_min = [x[sorted(min_var_idx), :].to(torch.float32) for x in feature_map_all]
    feature_map_max = [x[sorted(max_var_idx), :].to(torch.float32) for x in feature_map_all]
    feature_map_rand = [x[sorted(rand_var_idx), :].to(torch.float32) for x in feature_map_all]
    feature_all = [x.to(torch.float32) for x in feature_map_all]
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

    score_all = get_cross_validated_benchmarking_results(benchmark=benchmark_, feature_extractor=feature_all,
                                                            extract_mode=extract_mode)
    n_subs = len(np.unique(benchmark_rand.metadata.subj_id.values))
    # #
    all_regions = ['EVC', 'OTC', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4',
                   'FFA-1', 'FFA-2', 'OFA', 'EBA', 'FBA-1', 'FBA-2', 'OPA', 'PPA',
                   'VWFA-1', 'VWFA-2', 'OWFA']
    colors = [np.divide((255, 153, 51), 255), np.divide((160, 160, 160), 256),
              np.divide((51, 153, 255), 255)]

    plot_path='/om2/user/ehoseini/MyData/DeepJuice/'

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

        width = 0.15 # the width of the bars
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
        # ax.set_ylim((-.0, 0.4))
        ax.set_xlim((-.5, 6.5))
        ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)



        fig.savefig(os.path.join(plot_path,
                                 f'regression_score_DsParametricVision_by_voxel_procrustes_alignment_{region}.png'),
                    dpi=250, format='png',
                    metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

        # fig.savefig(os.path.join(plot_path,
        #                          f'regression_score_DsParametricVision_by_voxel_procrustes_alignment__{region}.eps'),
        #             format='eps', metadata=None,
        #             bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

        # save x_min x_rand and x_max as a dictionary to a file


    # deepjuice_path = '/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
    # extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    # optim_id_min = f'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    # optim_id_max = f'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    # ext_obj = extract_pool[extract_id]()
    # deepjuice_identifier = f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    # ext_obj.identifier = deepjuice_identifier
    # activations_list=[]
    # for model_ in tqdm(selected_models):
    #     save_file = f'{deepjuice_path}/nsd/{model_}*{extract_mode}.pkl'
    #     original_files = glob(save_file)
    #     # open the file
    #     with open(original_files[0], 'rb') as f:
    #         original = pickle.load(f)
    #     layer_id = original[0]
    #     act_ = original[1]
    #     activation = dict(model_name=model_, layer=layer_id, activations=act_)
    #     activations_list.append(activation)
    #     layers_list.append(layer_id)
    #
    #
    # optim_obj = optim_pool[optim_id_min]()
    # optim_obj.N_S = 1000
    # optim_obj.extract_type = 'activation'
    # optim_obj.activations = activations_list
    # optim_obj.extractor_obj = ext_obj
    # optim_obj.early_stopping = False
    # optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,save_results=False)
    #
    # # make a random set of 75
    # S = np.random.choice(optim_obj.N_S, optim_obj.N_s, replace=False)
    # # extract ev sentences
    # # find location of ev sentences in sentences
    # 2-optim_obj.gpu_object_function_debug(variance_idx[-75:])[0]
    # 2-optim_obj.gpu_object_function_debug(variance_idx[:75])[0]
    # 2-optim_obj.gpu_object_function_debug(S)[0]
    #
    #
    #
    #
    #
    #
    #
    #
    #
