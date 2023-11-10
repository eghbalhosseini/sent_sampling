import glob
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG, RESULTS_DIR, UD_PARENT, SAVE_DIR,load_obj,save_obj,ANALYZE_DIR
from utils import extract_pool
import pickle
from neural_nlp.models import model_pool, model_layers
import fnmatch
import re
from utils.extract_utils import model_extractor_parallel
from utils.optim_utils import optim_pool
import matplotlib.pyplot as plt
import torch
import matplotlib
import numpy as np
if __name__ == '__main__':
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=False'
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=False'


    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()

    # first extract ev sentences
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))

    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]
    ev_sentences

    ext_obj()

    sentences = [x[1] for x in ext_obj.model_group_act[0]['activations']]
    # find location of ev sentences in sentences
    ev_sentence_ids = []
    for ev_sent in ev_sentences:
        # remove the period
        ev_sent = ev_sent[:-1]
        ev_sentence_ids.append(sentences.index(ev_sent))

    optim_id='coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=200-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    optim_obj = optim_pool[optim_id]()

    optim_obj.load_extractor(ext_obj)

    low_resolution= False
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=False,
                                                 save_results=False)


    DS_max,RDM_max=optim_obj.gpu_object_function_debug(ev_sentence_ids)
    DS_max=2-DS_max

    if  isinstance(RDM_max, torch.Tensor):
        RDM_max = RDM_max.cpu().numpy()
    RDM_max=2-RDM_max
    ds_rand = []
    RDM_rand = []
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optim_obj.N_S, optim_obj.N_s))
        d_s_r, RDM_r = optim_obj.gpu_object_function_debug(sent_random)
        ds_rand.append(d_s_r)
        RDM_rand.append(RDM_r)
    ds_rand = 2 - np.asarray(ds_rand)
    RDM_rand = [2 -x for x in  RDM_rand]
    ## reset defaults
    plt.rcdefaults()

    ## Set up LaTeX fonts
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 6,
    })
    y_lim=(.6,2)
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.2, .7, .08, .25))
    ax.scatter(.02 * np.random.normal(size=(np.asarray(len(ds_rand)))) + 0,np.asarray(ds_rand),
               color=(.6, .6, .6), s=2, alpha=.3)
    rand_mean = np.asarray(ds_rand).mean()
    ax.scatter(0, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
               label=f'random= {rand_mean:.4f}', edgecolor='k')
    ax.scatter(0, DS_max, color=np.divide((255, 128, 0), 255), s=50, label=f'Ds_max={DS_max:.4f}', edgecolor='k')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.4, 0.4))
    ax.set_ylim(y_lim)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1.1, .2), frameon=True)
    ax.set_ylabel(r'$D_s$')
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    mask = np.triu(np.ones_like(RDM_max, dtype=np.bool))
    # change True to nan and false to 1
    mask = np.where(mask, np.nan, 1)

    ax = plt.axes((.6, .73, .25, .25))
    RDM_rand_mean = torch.stack(RDM_rand).mean(0).cpu().numpy()
    RDM_rand_mean=RDM_rand_mean.T
    RDM_rand_mean = np.multiply(RDM_rand_mean, mask)
    #im = ax.imshow(RDM_rand_mean, cmap='viridis', vmax=np.nanmax(RDM_max),vmin=0)
    im = ax.imshow(RDM_rand_mean, cmap='viridis', vmax=1.6, vmin=.6)
    # add values to image plot
    for i in range(RDM_rand_mean.shape[0]):
        for j in range(RDM_rand_mean.shape[1]):
            text = ax.text(j, i, f"{RDM_rand_mean[i, j]:.2f}",
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_title('RDM_rand')
    # set ytick labels to ext_obj.model_spec
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax = plt.axes((.6, .4, .25, .25))
    # transpose the RDM_max and set the upper triangle to nan
    # if RDM_max is on on torch move to cpu
    if isinstance(RDM_max, torch.Tensor):
        RDM_max = RDM_max.cpu().numpy()

    RDM_max = np.triu(RDM_max, k=1).T + np.triu(RDM_max, k=1)
    # change the diagonal to nan
    np.fill_diagonal(RDM_max, np.nan)
    # create an upper triangle mask
    # multiply the RDM_max with the mask
    RDM_max = RDM_max * mask

    # change the upper triangle to nan
    # add values to image plot
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white', 1.)
    #im = ax.imshow(RDM_max, cmap=cmap, vmin=0,vmax=np.nanmax(RDM_max))
    im = ax.imshow(RDM_max, cmap=cmap, vmin=.6, vmax=1.6)
    # add lower triangle of RDM_max to image plot
    for i in range(RDM_max.shape[0]):
        for j in range(RDM_max.shape[1]):
            text = ax.text(j,i, f'{RDM_max[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax.set_title('RDM_max')
    ax = plt.axes((.9, .05, .01, .25))
    plt.colorbar(im, cax=ax)

    rdm_rand_vec = RDM_rand_mean[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_max_vec = RDM_max[np.tril_indices(RDM_max.shape[0], k=-1)]

    # plot rdm vectors connecting points from rdom_rand to rdm max to rdm min
    # fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.1, .05, .1, .25))
    color_set = [ np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255)]
    rdm_vec = np.vstack(( rdm_rand_vec, rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5, zorder=1)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2], rdm_vec[:, i], color=color_set, s=10, marker='o', alpha=.8, zorder=2)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels([ 'ds_rand', 'ds_max'], fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim(y_lim)
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 2.25))

    fig.show()
    save_path = Path(ANALYZE_DIR)


    save_loc = Path(save_path.__str__(),  f'ANNSet1_Ds_{extract_id}_v2.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(),  f'ANNSet1_Ds_{extract_id}_v2.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='white',
                edgecolor='white')

