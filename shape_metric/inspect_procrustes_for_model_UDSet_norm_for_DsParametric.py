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
import torch
from sent_sampling.utils import extract_pool
from netrep.utils import align, pt_align, pt_orthogonal_procrustes
if __name__ == '__main__':
    normalize = lambda x: x / torch.sqrt(torch.trace(torch.mm(x.T, x)))
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'full_batch'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'zero_pad'  # 'pca' or 'none' or 'zero_pad'
    svd_solver = 'gesvd'  # 'gesvd' or 'svd', or 'lowrank'
    tolerance = 1e-12
    steps = 100
    verbose = True
    n_init = 1
    prev_objective = 1e10
    X_bar_model_final = None
    aligned_Xs_model_final = None
    # print configuration
    print(
        f'grp: {grp}, method: {method}, adjust_mode: {adjust_mode}, svd_solver: {svd_solver}, tolerance: {tolerance} \n')
    file = Path(
        f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/DeepJuice/shape_metric_highres_language_{grp}_{method}_{svd_solver}_{adjust_mode}_{tolerance}_{n_init}_{steps}_norm.pkl')
    with open(file.__str__(), 'rb') as f:
        results_dict=pd.read_pickle( f)
    #%%
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    # read the excel that contains the selected sentences
    # %%  RUN SANITY CHECKS
    ds_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN/ds_parametric/ANNSET_DS_MIN_MAX_from_100ev_eh_FINAL.csv')
    # read also the actuall experiment stimuli
    stim_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN//ds_parametric/fMRI_final/stimuli_order_ds_parametric.csv',
                           delimiter='\t')
    # find unique conditions
    unique_cond = np.unique(stim_csv.Condition)
    # for each unique_cond find sentence transcript
    unique_cond_transcript = [stim_csv.Stim_transcript[stim_csv.Condition == x].values for x in unique_cond]
    # remove duplicate sentences in unique_cond_transcript
    unique_cond_transcript = [list(np.unique(x)) for x in unique_cond_transcript]
    ds_min_list = unique_cond_transcript[1]
    ds_max_list = unique_cond_transcript[0]
    ds_rand_list = unique_cond_transcript[2]
    # extract the ds_min sentence that are in min_included column
    ds_min_ = ds_csv.DS_MIN_edited[(ds_csv['min_include'] == 1)]
    ds_max_ = ds_csv.DS_MAX_edited[(ds_csv['max_include'] == 1)]
    ds_rand_ = ds_csv.DS_RAND_edited[(ds_csv['rand_include'] == 1)]
    # check if ds_min_ and ds_min_list have the same set of sentences regardless of the order
    assert len([ds_min_list.index(x) for x in ds_min_]) == len(ds_min_)
    assert len([ds_max_list.index(x) for x in ds_max_]) == len(ds_max_)
    assert len([ds_rand_list.index(x) for x in ds_rand_]) == len(ds_rand_)
    # %% MORE SANITY CHECKS FOR THE ACTIVATIONS
    # get the
    ds_min_sent = ds_csv.DS_MIN[(ds_csv['min_include'] == 1)]
    ds_max_sent = ds_csv.DS_MAX[(ds_csv['max_include'] == 1)]
    ds_rand_sent = ds_csv.DS_RAND[(ds_csv['rand_include'] == 1)]
    # laod the extractor
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    ext_obj()
    # find location of sentences in ext_obj.model_group_act
    ds_min_list = []
    ds_max_list = []
    ds_rand_list = []
    for idx, act_dict in enumerate(ext_obj.model_group_act):
        True
        sentences = [x[1] for x in act_dict['activations']]
        # find the location of ds_min_sent in sentences
        ds_min_loc = [sentences.index(x) for x in ds_min_sent]
        ds_max_loc = [sentences.index(x) for x in ds_max_sent]
        ds_rand_loc = [sentences.index(x) for x in ds_rand_sent]
        ds_min_list.append(ds_min_loc)
        ds_max_list.append(ds_max_loc)
        ds_rand_list.append(ds_rand_loc)

    ds_min_list = np.asarray(ds_min_list).transpose()
    ds_max_list = np.asarray(ds_max_list).transpose()
    ds_rand_list = np.asarray(ds_rand_list).transpose()
    # make sure the row are the same in ds_min_list
    assert np.all([np.all(x == x[0]) for x in ds_min_list])
    assert np.all([np.all(x == x[0]) for x in ds_max_list])
    assert np.all([np.all(x == x[0]) for x in ds_rand_list])
    ds_min_id = [x[0] for x in ds_min_list]
    ds_max_id = [x[0] for x in ds_max_list]
    ds_rand_id = [x[0] for x in ds_rand_list]




    #%%
    results_dict.keys()
    X_bar_model=results_dict['X_bar_model_final']
    torch.trace(torch.mm(X_bar_model.T,X_bar_model))
    X_bar_model=normalize(X_bar_model)

    x_model_aligned= results_dict['aligned_Xs_model_final']
    aligned_Xbar_model=[]
    for idx in tqdm(range(len(x_model_aligned))):
        x=x_model_aligned[idx]
        aligned_Xbar_model.append(x @ pt_align(x, X_bar_model, group="orth",svd_solver=svd_solver))

    pca = PCA(n_components=2)
# do a pca on x_align_min and then transform x_align_max

    X=pca.fit_transform(X_bar_model.cpu().numpy())

    X_align_pca_min= X[ds_min_id,:]
    X_align_pca_max= X[ds_max_id,:]

    #%%
    sorted_image_ids=np.stack([np.arange(len(ds_min_id)), np.arange(len(ds_max_id))]).flatten()
    sent_min=[sentences[x] for x in ds_min_id]
    sent_max=[sentences[x] for x in ds_max_id]
    sent_=np.stack([sent_min, sent_max]).flatten()
    x_pca = np.concatenate((X_align_pca_min,X_align_pca_max), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_align_pca_min.shape[0]), np.repeat('max', X_align_pca_min.shape[0])), axis=0)
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



    #%%
    X_diff_model = [x - X_bar_model for x in aligned_Xbar_model]
    X_diff_model = torch.stack(X_diff_model)
    X_var_model = torch.linalg.vector_norm(X_diff_model, ord=2,dim=-1)


    # X_standardized = scaler.fit_transform(X_var_model.T.cpu().numpy())
    X_standardized = X_var_model.T.cpu().numpy()
    X_standardized -= X_standardized.mean(axis=0, keepdims=True)
    pca = PCA(n_components=X_standardized.shape[1])
    X_pca = pca.fit_transform(X_standardized)
    pc1_scores = X_pca[:, 0]
    pc2_scores = X_pca[:, 1]
    # key_samples_indices = np.argsort(np.abs(pc1_scores))[::-1]
    key_samples_indices = np.argsort(pc1_scores)
    # get variance explained by each component
    var_explained = pca.explained_variance_ratio_
    # get the components
    # compute the variance explained by each component in percentage
    var_explained_perc = var_explained * 100
    # print variance explained in human readable format

    [print(f"{x:.2f}%") for x in var_explained_perc]

    fig, ax = plt.subplots(figsize=(8, 8))
    # choose colors so that the scale with the pca1 score, with max being red and min being blue

    colors = plt.cm.Reds(np.linspace(0, 1, len(key_samples_indices)))
    x = pc1_scores[key_samples_indices]
    y = pc2_scores[key_samples_indices]
    ax.scatter(x, y, color=colors, edgecolor='none', s=50)
    ax.scatter(x[ds_min_id], y[ds_min_id], color='none', edgecolor='black', s=50, label='min')

    ax.scatter(x[ds_max_id], y[ds_max_id], color='none', edgecolor='blue', s=50, label='min')
    # Add labels and title
    ax.set_xlabel(f'PC1, var explained:{var_explained_perc[0]:.2f}%', fontdict={'fontsize': 20})
    ax.set_ylabel(f'PC2, var explained:{var_explained_perc[1]:.2f}%', fontdict={'fontsize': 20})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add the origin lines
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    min_val = min(np.concatenate([x, y], axis=0))
    max_val = max(np.concatenate([x, y], axis=0))
    min_val = min_val - 0.05 * min_val
    max_val = max_val + 0.05 * max_val
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    # do plot in the top left corner of pca plot to show variance explained per pca component
    ax1 = fig.add_axes([0.2, 0.8, 0.15, 0.1])
    ax1.bar(range(len(var_explained_perc)), var_explained_perc, color='red')
    ax1.set_xticks(range(len(var_explained_perc)))
    ax1.set_xticklabels(range(1, len(var_explained_perc) + 1))
    ax1.set_title('Variance explained')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Display the plot
    fig.show()

