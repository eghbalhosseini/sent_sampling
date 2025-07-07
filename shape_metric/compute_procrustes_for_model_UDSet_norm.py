
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
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool
from sent_sampling.utils.opt_exp_design import swap, LOGGER
import numpy as np
from sklearn.manifold import MDS

if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
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
    # createa n list of empty tensors
    feature_map_all=[None]*len(extractor_obj.model_group_act)

    for idx in range(len(extractor_obj.model_group_act)):
        X=extractor_obj.model_group_act[idx]['activations']
        X=np.asarray([a[0] for a in X])
        X = torch.tensor(X).to(torch.float64)
        column_means = torch.mean(X, dim=0)
        centered_X = X - column_means
        # centered_X/=centered_X.norm(p='fro')
        feature_map_all[idx] = centered_X.to(device)
    # compute a forbenious norm aacross all models
    #model_norm=torch.stack(feature_map_all).norm( p='fro')
    # devide all feature_maps by the norm
    #feature_map_all=[x/model_norm for x in feature_map_all]
    #%%
    max_pad=max([x.shape[-1] for x in feature_map_all])
    #%% do some zero-padding here
    #max_pad= 5920
    x_model = [F.pad(x, pad=(0, max_pad - x.shape[-1], 0, 0), mode='constant', value=0) for x in feature_map_all]
    #%% do the norming
    # drop the required grad
    # write a lambda function for doing the norm to apply it tot he list of tensors
    # norm = lambda x: x/(torch.norm(x,p=fro) / torch.sqrt(torch.tensor(x.numel())
    #normalize = lambda x: x / (torch.norm(x, p='fro') / torch.sqrt(torch.tensor(x.numel(), dtype=float_version)))
    #%% do the norming
    normalize = lambda x: x / torch.sqrt(torch.trace(torch.mm(x.T, x)))
    x_model = [x.requires_grad_(False) for x in x_model]
    # do norm
    x_model = [normalize(x) for x in x_model]
    # create a set of random matrix with same size as the model
    #%%
    # make them not require grad
    #%% compute the model procrustes first and then do model to brain alginment
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    svd_solver = 'gesvd'  # 'gesvd' or 'svd', or 'lowrank'
    tolerance = 1e-12
    steps= 100
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

            X_bar_model, aligned_Xs_model = pt_frechet_mean(x_model, group=grp, method=method, return_aligned_Xs=True,warmstart=None,
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
    file=Path(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/shape_metric_highres_language_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.pkl')
    results_dict=dict(X_bar_model_final=X_bar_model_final,aligned_Xs_model_final=aligned_Xs_model_final)
    with open(file.__str__(), 'wb') as f:
        pickle.dump(results_dict, f)
    #%%
    mds = MDS(n_components=100, random_state=42, dissimilarity='euclidean', n_jobs=-1)
    X=feature_map_all[0].cpu().numpy()
    X_mds = mds.fit_transform(X)



