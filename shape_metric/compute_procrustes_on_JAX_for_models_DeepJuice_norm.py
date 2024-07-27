import sys
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.scipy.linalg import svd
from typing import Literal
from jax.config import config
import time
from netrep.utils import align, pt_align
import getpass
from pathlib import Path
from sent_sampling.utils.shape_utils import jax_frechet_mean,jax_align,pad_jax
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

from benchmarks import NSDBenchmark, NSDSampleBenchmark
from deepjuice._backends.cupyfy import convert_to_tensor
import multiprocessing
import os
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
import pickle
from glob import glob
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
    float_version = jnp.float64
    device = jax.devices('gpu')[0]
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
        X = jnp.array(feature_map_all[idx], dtype=float_version)
        column_means = jnp.mean(X, axis=0)
        centered_X = X - column_means
        # centered_X = centered_X / jnp.linalg.norm(centered_X, ord='fro')  # Uncomment if normalization is needed
        feature_map_all[idx] = jax.device_put(centered_X, device)
    #%%
    benchmark_ = NSDBenchmark(path_dir=benchmark_path)
    x_fmri = jnp.array((convert_to_tensor(benchmark_.response_data.to_numpy())).T)
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    rois = roi_indices.keys()
    roi = 'OTC'
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices[roi].values()]
    # for each roi get the largest size and pad the rest
    max_pad = max([x.shape[1] for x in fmri_roi_sub_x])
    x_sub_fmri = [pad_jax(x,  max_pad ) for x in fmri_roi_sub_x]

    #%% do the norming
    normalize = lambda x: x / (jnp.linalg.norm(x, ord='fro') / 1)
    # do norm
    x_model = [normalize(x) for x in feature_map_all]
    x_sub_fmri = [normalize(x) for x in x_sub_fmri]

    #%%
    # pad x_model up to max_pad
    x_model = [pad_jax(x, max_pad) for x in x_model]
    x_model = [jax.device_put(x, device=jax.devices('gpu')[0])for x in x_model]
    # normalize x_model_sazmple
    x_model = [normalize(x) for x in x_model]
    # normalize
    #%% compute the model procrustes first and then do model to brain alginment
    group = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'streaming'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    tol = 1e-10
    max_iter= 200
    verbose = True
    n_init=2
    prev_objective=1e9
    X_bar_model_final=None
    aligned_Xs_model_final=None
    # print configuration
    print(f'grp: {group}, method: {method}, adjust_mode: {adjust_mode}, tolerance: {tol} \n')
    for k in range(n_init):
        # print the current iteration
        print(f'iteration: {k}')
        #######
        # time the operation
        start = time.time()
        X_bar_model, aligned_Xs_model = jax_frechet_mean(x_model, group=group, method=method, return_aligned_Xs=True,max_iter=max_iter,verbose=verbose, tol=tol,svd_solver=None)
        end = time.time()
        print(f'Elapsed time for full set: {end - start}')
        #######
        # compute the objective
        X_diff_model = jnp.array([x - X_bar_model for x in aligned_Xs_model])
        X_var_model = jnp.linalg.norm(X_diff_model, ord=2, axis=-1)
        objective=jnp.linalg.norm(X_var_model)
        print(f'objective: {objective}')
        if objective<prev_objective:
            X_bar_model_final=X_bar_model
            aligned_Xs_model_final=aligned_Xs_model

    # align subjects to the mean model
    # safe final x_bar_model and aligned_Xs_model
    # Compute A^T * A
    # Transpose the matrix A
    A = jax.device_put(jnp.array(x_model[0], dtype=jnp.float64), device=jax.devices('gpu')[0])
    A_T = A.T
    # Compute the product A^T * A
    ATA = jnp.dot(A_T, A)
    # Compute the trace of A^T * A
    trace_ATA = jnp.trace(ATA)
    # Print the result
    print("Trace of A^T * A:", trace_ATA)
    Q_algin=jax_align(A,X_bar_model)
    # compute norm of Q along columns
    Q_algin_fro_norm = jnp.linalg.norm(Q_algin, axis=0)
    A_align = jnp.dot(A, Q_algin)

    ATA_align = jnp.dot(A_align.T, A_align)
    # Compute the trace of A^T * A
    trace_ATA_align = jnp.trace(ATA_align)
    print("Trace of A^T_align * A_align:", trace_ATA_align)

    file=Path(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/shape_metric_jax_highres_vision_{group}_{method}_{adjust_mode}_{tol}_{n_init}_norm.pkl')
    results_dict=dict(X_bar_model_final=X_bar_model_final,aligned_Xs_model_final=aligned_Xs_model_final)
    with open(file.__str__(), 'wb') as f:
        pickle.dump(results_dict, f)



