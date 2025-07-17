
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
    platonic_path='/rdma/vast-rdma/vast/evlab/ehoseini/MyData/shape_metric/'
else:
    sys.path.append('/Users/eghbalhosseini/MyCodes/DeepJuiceDev/')
    image_paths = '/Users/eghbalhosseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/Users/eghbalhosseini/MyData/DeepJuice/workspace/nsd/'
    benchmark_path = '/Users/eghbalhosseini/MyData/DeepJuice/nsd_data/'

import sys
import os
# Add the root directory of the repo to sys.path
sys.path.append(os.path.abspath('/om2/user/ehoseini/platonic-rep'))
from utils import to_feature_filename
from measure_alignment import prepare_features,compute_score, compute_alignment
from extract_features import extract_llm_features, extract_lvm_features
from sklearn.preprocessing import StandardScaler
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
SUPPORTED_METRICS = [
        "cycle_knn",
        "mutual_knn",
        "lcs_knn",
        "cka",
        "unbiased_cka",
        "cknna",
        "svcca",
        "edit_distance_knn",
    ]

from collections import namedtuple
def mock_get_args():
    mock_args = namedtuple('debug', ['output_dir', 'metric','topk','precise'])
    new_args = mock_args('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/shape_metric/alignment', 'unbiased_cka', 10, True)
    return new_args


if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    selected_models = ["vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",]

    models_sh = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']
    # read image path
    dataset='wit_1024'
    subset='train'

    #%%
    extract_mode = 'redux'
    activations_list = []
    layers_list = []
    vlm_model_paths=[]
    for model_ in selected_models:
        save_path = to_feature_filename(
            platonic_path, dataset, subset, model_,
            pool='cls', prompt=None, caption_idx=None,
        )
        vlm_model_paths.append(save_path)

    llm_models = [
        "huggyllama/llama-7b",
        "huggyllama/llama-13b",

    ]
    llm_model_path=[]
    for model_ in llm_models:
        save_path = to_feature_filename(
            platonic_path, dataset, subset, model_,pool='last', prompt=True, caption_idx=None)
        # assert path exist
        llm_model_path.append(save_path)



    topk=10
    precise=True

    args=mock_get_args()
    alignment_scores, alignment_indices = compute_alignment(vlm_model_paths,llm_model_path, args.metric,args.topk,args.precise)



    #%% compute the values for procustes output
    vlm_model_paths_random = []
    vlm_model_paths_min = []
    vlm_model_paths_max = []
    for model_ in selected_models:
        save_path = to_feature_filename(
            platonic_path, dataset, subset, model_,
            pool='cls', prompt=None, caption_idx=None,
        )
        new_path=save_path.replace('/train','/procrustes')
        random_path=new_path.replace('.pt','_random.pt')
        vlm_model_paths_random.append(random_path)

        min_path=new_path.replace('.pt','_min.pt')
        vlm_model_paths_min.append(min_path)

        max_path=new_path.replace('.pt','_max.pt')
        vlm_model_paths_max.append(max_path)

    llm_model_paths_random=[]
    llm_model_paths_min=[]
    llm_model_paths_max=[]

    for model_ in llm_models:
        save_path = to_feature_filename(
            platonic_path, dataset, subset, model_,pool='last', prompt=True, caption_idx=None)
        # assert path exist
        new_path=save_path.replace('/train','/procrustes')
        random_path=new_path.replace('.pt','_random.pt')
        llm_model_paths_random.append(random_path)
        min_path=new_path.replace('.pt','_min.pt')
        llm_model_paths_min.append(min_path)
        max_path=new_path.replace('.pt','_max.pt')
        llm_model_paths_max.append(max_path)


    method_k=5
    SUPPORTED_METRICS[method_k]
    alignment_scores_rand, alignment_indices_rand = compute_alignment(vlm_model_paths_random,llm_model_paths_random, SUPPORTED_METRICS[method_k],args.topk,args.precise)

    alignment_scores_min, alignment_indices_min = compute_alignment(vlm_model_paths_min, llm_model_paths_min,
                                                                      SUPPORTED_METRICS[method_k], args.topk, args.precise)

    alignment_scores_max, alignment_indices_max = compute_alignment(vlm_model_paths_max, llm_model_paths_max,
                                                                    SUPPORTED_METRICS[method_k], args.topk, args.precise)


