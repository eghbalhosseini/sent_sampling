
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
import multiprocessing
import os
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
    # compute a forbenious norm aacross all models
    #model_norm=torch.stack(feature_map_all).norm( p='fro')
    # devide all feature_maps by the norm
    #feature_map_all=[x/model_norm for x in feature_map_all]
    #%%
    benchmark_ = NSDBenchmark(path_dir=benchmark_path)
    x_fmri = (convert_to_tensor(benchmark_.response_data.to_numpy()).to(dtype=float_version, device=device)).T
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    rois = roi_indices.keys()
    roi = 'OTC'
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices[roi].values()]
    # for each roi get the largest size and pad the rest
    max_pad = max([x.shape[1] for x in fmri_roi_sub_x])

    #%% do some zero-padding here
    #max_pad= 5920
    x_sub_fmri = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in fmri_roi_sub_x]
    x_model = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]
    #%% do the norming
    # drop the required grad
    # write a lambda function for doing the norm to apply it tot he list of tensors
    # norm = lambda x: x/(torch.norm(x,p=fro) / torch.sqrt(torch.tensor(x.numel())
    #normalize = lambda x: x / (torch.norm(x, p='fro') / torch.sqrt(torch.tensor(x.numel(), dtype=float_version)))
    #%% do the norming
    x=x_model[0]
    torch.sqrt(torch.trace(torch.mm(x.T, x)))
    normalize = lambda x: x / torch.sqrt(torch.trace(torch.mm(x.T, x)))
    x_model = [x.requires_grad_(False) for x in x_model]
    x_sub_fmri = [x.requires_grad_(False) for x in x_sub_fmri]
    # do norm
    x_model = [normalize(x) for x in x_model]
    x_sub_fmri = [normalize(x) for x in x_sub_fmri]
    # create a set of random matrix with same size as the model
    #%%
    # make them not require grad
    #%% compute the model procrustes first and then do model to brain alginment
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'streaming'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    svd_solver = 'gesvd'  # 'gesvd' or 'svd', or 'lowrank'
    tolerance = 1e-10
    steps= 500
    verbose = True
    n_init=1
    prev_objective=1e10
    X_bar_model_final=None
    aligned_Xs_model_final=None
    # print configuration
    print(f'grp: {grp}, method: {method}, adjust_mode: {adjust_mode}, svd_solver: {svd_solver}, tolerance: {tolerance} \n')
    for k in range(n_init):
        # print the current iteration
        print(f'iteration: {k}')
        with torch.no_grad():

            X_bar_model, aligned_Xs_model = pt_frechet_mean(x_model, group=grp, method=method, return_aligned_Xs=True,
                                                      max_iter=steps,verbose=verbose, tol=tolerance,svd_solver=svd_solver)

        X_diff = [X - X_bar_model for X in aligned_Xs_model]
        X_diff = torch.stack(X_diff)
        X_var_model = X_diff.norm(dim=-1, p='fro')
        objective=X_var_model.norm()
        print(f'objective: {objective}')
        if objective<prev_objective:
            X_bar_model_final=X_bar_model
            aligned_Xs_model_final=aligned_Xs_model

    # align subjects to the mean model
    # safe final x_bar_model and aligned_Xs_model
    file=Path(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/shape_metric_highres_vision_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.pkl')
    results_dict=dict(X_bar_model_final=X_bar_model_final,aligned_Xs_model_final=aligned_Xs_model_final)
    with open(file.__str__(), 'wb') as f:
        pickle.dump(results_dict, f)

