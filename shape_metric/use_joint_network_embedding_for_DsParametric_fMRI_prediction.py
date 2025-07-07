import copy

from netrep.metrics import LinearMetric
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import cross_validate
from netrep.multiset import pairwise_distances, frechet_mean
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib
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
from netrep.utils import align
from scipy.spatial.distance import pdist
from scipy.io import savemat
from brainscore.metrics import Score
from neural_nlp import models
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.benchmarks.neural import apply_aggregate
import copy
import xarray as xr
from scipy.stats import median_abs_deviation as median_absolute_deviation
if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    # load act_leftout
    pre_pca=False
    act_dir='/Users/eghbalhosseini/MyData/neural_nlp_bench/activations/DsParametricfMRI/'
    model_resp_leftout=[]
    model_resp_dsparametric=[]
    model_resp_all=[]
    feature_map_min=[]
    stimulus_id_min=[]
    feature_map_max=[]
    stimulus_id_max=[]
    model_resp_list=[]
    for model_,layer in model_layers:
        model_name = model_
        save_path = Path(f'{act_dir}/{model_name}_DsParametricfMRI.pkl')
        # make sure parent exist
        # load from save path
        model_resp=pd.read_pickle(save_path)
        model_resp_list.append(model_resp)
        model_resp_min=model_resp[model_resp.stim_group=='min']
        model_resp_max=model_resp[model_resp.stim_group=='max']
        stimulus_id_min.append(model_resp_min.stimulus_id.values)
        stimulus_id_max.append(model_resp_max.stimulus_id.values)
        X_min=model_resp_min.values
        column_means = np.mean(X_min, axis=0)
        centered_X_min = X_min - column_means
        feature_map_min.append(centered_X_min)
        X_max=model_resp_max.values
        column_means = np.mean(X_max, axis=0)
        centered_X_max = X_max - column_means
        feature_map_max.append(centered_X_max)

    # make sure all rows of stimulus_id_mins are the same
    assert all([np.all(stimulus_id_min[0]==x) for x in stimulus_id_min])
    assert all([np.all(stimulus_id_max[0]==x) for x in stimulus_id_max])

    #%%
    grp = 'perm'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    sig_corr=True
    tolerance = 1e-5
    verbose = True
    file_name = f'neural_nlp_multi_shape_distance_individual_DsParametric_{grp}_{adjust_mode}_{method}_pre_pca_{pre_pca}_centered'
    save_path = Path(f'{act_dir}/{file_name}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    X = feature_map_min
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in feature_map_min]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in X]
        X_pad_max = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in feature_map_max]

    X_var_min, aligned_Xs_min = frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True, max_iter=50,
                                             verbose=verbose, tol=tolerance)
    X_var_max, aligned_Xs_max = frechet_mean(X_pad_max, group=grp, method=method, return_aligned_Xs=True, max_iter=50,
                                             verbose=verbose, tol=tolerance)


    all_X_dict={'aligned_min':aligned_Xs_min,'aligned_max':aligned_Xs_max,'var_min':X_var_min,'var_max':X_var_max}
    with open(save_path, 'wb') as f:
        pkl.dump(all_X_dict, f)


    


    #%%
    model_resp=model_resp_list[0]
    model_resp_temp=copy.deepcopy(model_resp_list[5])
    model_resp_min_temp = model_resp_temp[model_resp_temp.stim_group == 'min']
    model_resp_max_temp = model_resp_temp[model_resp_temp.stim_group == 'max']
    model_align_resp = copy.deepcopy(model_resp_min_temp)
    model_align_resp.values = X_var_min

    benchmark_name = 'DsParametricfMRI-first-reliable-min-Encoding'
    benchmark=benchmark_pool[benchmark_name]
    model_align_resp['stimulus_id'].values = benchmark._target_assembly['stimulus_id'].values
    cross_scores = benchmark._cross(benchmark._target_assembly,
                                    apply=lambda cross_assembly: benchmark._apply_cross(model_align_resp,
                                                                                        cross_assembly))
    raw_scores = cross_scores.raw
    raw_neuroids = apply_aggregate(lambda values: values.mean('split'), raw_scores)
    language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
    # score = self._aggregate_no_ceiling(language_neuroids, ceiling=[], subject_column='subject')
    x0 = benchmark._aggregate_no_ceiling(language_neuroids, subject_column='subject')

    benchmark_name = 'DsParametricfMRI-second-reliable-min-Encoding'
    benchmark=benchmark_pool[benchmark_name]

    model_align_resp['stimulus_id'].values = benchmark._target_assembly['stimulus_id'].values
    cross_scores = benchmark._cross(benchmark._target_assembly,
                                    apply=lambda cross_assembly: benchmark._apply_cross(model_align_resp,
                                                                                        cross_assembly))
    raw_scores = cross_scores.raw
    raw_neuroids = apply_aggregate(lambda values: values.mean('split'), raw_scores)
    language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
    # score = self._aggregate_no_ceiling(language_neuroids, ceiling=[], subject_column='subject')
    x1 = benchmark._aggregate_no_ceiling(language_neuroids, subject_column='subject')


    x0_raw = x0.raw.raw
    x1_raw = x1.raw.raw
    if sig_corr:
        sig_corr_val_ratio=x0_raw.repetition_corr_ratio>0.95
        sig_corr_val = x0_raw.repetition_corr > 0.1
        sig_select= sig_corr_val_ratio & sig_corr_val
        x0_raw=x0_raw[:,sig_select]
        x1_raw=x1_raw[:,sig_select]
    # drop subject 1042
    subj_valid=x0_raw['subject']!=''
    x0_raw=x0_raw[:,subj_valid]
    x1_raw=x1_raw[:,subj_valid]
    # make sure x0_raw and x1_raw have the same voxel order
    assert all(x0_raw['neuroid_id'].values == x1_raw['neuroid_id'].values)
    # stack x0_raw and x1_raw into a repettion deminsion
    xr01=xr.concat([x0_raw, x1_raw], dim='repetition')
    xr01=xr01.mean('repetition')
    # group by
    subject_scores = xr01.groupby('subject').median()
    center = subject_scores.median('subject').values
    subject_values = np.nan_to_num(subject_scores.values,
                                   nan=0)
    subject_axis = subject_scores.dims.index(subject_scores['subject'].dims[0])
    error = median_absolute_deviation(subject_values, axis=subject_axis)
    # select the layer
    np.mean(center)





