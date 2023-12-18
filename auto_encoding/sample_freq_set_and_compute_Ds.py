import os
import numpy as np
import sys
from pathlib import Path
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
from matplotlib.pyplot import GridSpec
import pandas as pd
from pathlib import Path
import torch
from sent_sampling.utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 10,'font.weight':'bold'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
from scipy.stats import ttest_ind

if __name__ == '__main__':
    modelnames = ['roberta-base', 'xlnet-large-cased', 'bert-large-uncased', 'xlm-mlm-en-2048', 'gpt2-xl',
                  'albert-xxlarge-v2', 'ctrl']
    dataset_id = ['coca_preprocessed_all_clean_100K_sample_1_2_ds_max_est_n_10K',
                  'coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K']
    extract_id = [f'group=best_performing_pereira_1-dataset={dataset_id[0]}_textNoPeriod-activation-bench=None-ave=False',
                  f'group=best_performing_pereira_1-dataset={dataset_id[1]}_textNoPeriod-activation-bench=None-ave=False']
    optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'

    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    ext_obj()
    optimizer_obj = optim_pool[optim_id]()
    optimizer_obj.load_extractor(ext_obj)
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=True,
                                             save_results=False)
    ds_max_est = []
    RDM_max_est = []
    for k in tqdm(enumerate(range(1000))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        d_s_r, RDM_r = optimizer_obj.gpu_object_function_debug(sent_random)
        ds_max_est.append(2-d_s_r)
        RDM_max_est.append(2-RDM_r)

    ext_obj=extract_pool[extract_id[1]]()
    ext_obj.load_dataset()
    ext_obj()
    optimizer_obj = optim_pool[optim_id]()
    optimizer_obj.load_extractor(ext_obj)
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=True,
                                             save_results=False)
    ds_min_est = []
    RDM_min_est = []
    for k in tqdm(enumerate(range(1000))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        d_s_r, RDM_r = optimizer_obj.gpu_object_function_debug(sent_random)
        ds_min_est.append(2-d_s_r)
        RDM_min_est.append(2-RDM_r)

    colors = [np.divide((51, 153, 255), 255), np.divide((255, 153, 51), 255)]

    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio=8/11
    ax = plt.axes((.2, .7, .08, .25))
    ax.scatter(.04 * np.random.normal(size=(np.asarray(len(ds_min_est)))) + 0, np.asarray(ds_min_est),
               color=colors[0], s=2, alpha=.3)
    rand_mean = np.asarray(ds_min_est).mean()
    ax.scatter(0, rand_mean, color=colors[0], s=50,
               label=f'min estimated= {rand_mean:.4f}', edgecolor='k',zorder=100)

    ax.scatter(.04 * np.random.normal(size=(np.asarray(len(ds_max_est)))) + 0.5, np.asarray(ds_max_est),
               color=colors[1], s=2, alpha=.3)
    rand_mean = np.asarray(ds_max_est).mean()
    ax.scatter(.5, rand_mean, color=colors[1], s=50,
               label=f'max estimated= {rand_mean:.4f}', edgecolor='k',zorder=100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.4, 0.9))
    ax.set_ylim((1.1, 1.4))
    ax.set_ylabel('Alignment')
    ax.set_xticks([])
    ax.set_xticklabels([])
    # check if ds_min and ds_max are significantly different

    t, p = ttest_ind(ds_min_est, ds_max_est)

    RDM_min = torch.stack(RDM_min_est).mean(dim=0)
    RDM_max = torch.stack(RDM_max_est).mean(dim=0)


    rdm_max_vec = RDM_min[np.triu_indices(RDM_max.shape[0], k=1)].cpu().numpy()
    rdm_min_vec = RDM_max[np.triu_indices(RDM_min.shape[0], k=1)].cpu().numpy()
    # plot rdm vectors connecting points from rdom_rand to rdm max to rdm min
    # fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.1, .05, .15, .3 * pap_ratio))

    rdm_vec = np.vstack((rdm_min_vec, rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2,], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2], rdm_vec[:, i], color=colors, s=10, marker='o', alpha=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['ds_min\nestimated', 'ds_max\nestimated'], fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim((0, 1.3))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 2.25))
    ax.set_ylim([.8, 2])
    # ax.violinplot([0,1,2],rdm_vec.transpose(),showmeans=True,showextrema=False,showmedians=False)

    #fig.show()
    fig.savefig(f'{ANALYZE_DIR}/ds_min_max_estimation_from_frequency.pdf', dpi=300)

