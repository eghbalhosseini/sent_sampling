import warnings
import numpy as np
import os
os.chdir('/om/user/ehoseini/sent_sampling')
os.getcwd()
import torch.cuda
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import re
from utils import extract_pool , gpt2_xl_grp_config
from utils.extract_utils import model_extractor, model_extractor_parallel, SAVE_DIR
from utils.data_utils import SENTENCE_CONFIG,  RESULTS_DIR, save_obj, load_obj, ANALYZE_DIR
import utils.optim_utils
import importlib
importlib.reload(utils.optim_utils)

from utils.optim_utils import optim, optim_pool, pt_create_corr_rdm_short, optim_group
import argparse
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# rapids stuff
import cudf, cuml
from cuml.neighbors import NearestNeighbors
from cuml import PCA
from cuml.decomposition import PCA
from sklearn.cluster import SpectralClustering
import pandas as pd
from pathlib import Path
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # first load ev sentenes:
    # first find ev sentences
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'

    d_metric = 'correlation'
    ave_flag='False'
    ##
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))
    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]
    # load one extractor
    extractor_name='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False'
    extractor_name=extractor_name.replace('ave=False',f'ave={ave_flag}')
    extractor_obj = extract_pool[extractor_name]()
    extractor_obj.load_dataset()
    extractor_obj()
    model_names = [x['model_name'] for x in extractor_obj.model_group_act]
    sentences = [x['text'] for x in extractor_obj.data_]
    # find location of ev sentences in the set
    ev_selected_idx=[sentences.index(x) for x in ev_sentences]
    all_modl_dat = [model_dat['activations'] for model_dat in extractor_obj.model_group_act]
    all_modl_dat[4] = [x[0] for x in all_modl_dat[4]]
    #optim_ids = 'coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=200-n_init=1-run_gpu=True'
    #optimizer_obj = optim_pool[optim_ids]()
    #optimizer_obj.load_extractor(extractor_obj)
    #optimizer_obj.precompute_corr_rdm_on_gpu(low_dim=True, low_resolution=True, cpu_dump=True)
    # select a reference model for finind neighbors:
    # first reduce the dimensionality of the data for each model
    all_modl_act = [torch.Tensor(x).to(device) for x in all_modl_dat]
    [x.shape for x in all_modl_act]
    del all_modl_dat
    var_explained=[]
    all_modl_act_ld=[]
    for id, x in tqdm(enumerate(all_modl_act)):
        pca_=PCA(n_components=700)
        pca_.fit(x)
        var_explained.append(pca_.explained_variance_ratio_)
        x_ld = pca_.fit_transform(x)
        all_modl_act_ld.append(x_ld)

    for id, x in enumerate(var_explained):
        plt.plot(np.cumsum(x.get()[:600]),label=model_names[id])
    plt.legend(bbox_to_anchor=(.4, .4), frameon=True, fontsize=8)
    plt.show()
    [x.shape for x in all_modl_act_ld]
    del all_modl_act
    all_modl_act_ld=[torch.tensor(x).to(device) for x in all_modl_act_ld]
    all_modl_act_ld = torch.concat(all_modl_act_ld, dim=1)
    #
    knn_mdl = NearestNeighbors(n_neighbors=7, two_pass_precision=True, metric=d_metric)
    knn_mdl.fit(all_modl_act_ld)
    distances, indices = knn_mdl.kneighbors(all_modl_act_ld)
    sent_neighbor = []
    for _, ind in tqdm(enumerate(indices)):
        sent_neighbor.append('\t '.join([sentences[x] for x in list(ind.get())]))

    textfile = open(f'{RESULTS_DIR}/{Path(file_name).stem}_nearest_neighbors_combined_models_dist_metric_{d_metric}.txt',
                    "w")
    for idx in ev_selected_idx:
        textfile.write(sent_neighbor[idx] + "\n")
    textfile.close()
    del textfile

    optim_ids = 'coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=200-n_init=1-run_gpu=True'
    optimizer_obj = optim_pool[optim_ids]()
    optimizer_obj.load_extractor(extractor_obj)
    optimizer_obj.precompute_corr_rdm_on_gpu(low_dim=True, low_resolution=False, cpu_dump=False)

    ev_sent_Ds, ev_sent_ds_all = optimizer_obj.gpu_object_function_debug(ev_selected_idx)

    ev_sent_neighbors = list(indices.get()[ev_selected_idx, 1])
    neighbor_Ds, neighbor_ds_all = optimizer_obj.gpu_object_function_debug(ev_sent_neighbors)

    # random set
    ds_random = []
    for _, _ in tqdm(enumerate(range(200))):
        rand_set = list(np.random.choice(optimizer_obj.N_S, size=len(ev_selected_idx), replace=False))
        ds_rnd = optimizer_obj.gpu_object_function(rand_set)
        ds_random.append(ds_rnd)

    fig = plt.figure(figsize=(14, 7), dpi=100, frameon=False)
    ax = plt.axes((.1, .1, .1, .75))

    cmap = cm.get_cmap('viridis_r')
    tick_l = []
    tick = []
    idx = 0
    D_s_rand = ds_random
    ax.scatter(.2 * np.random.normal(size=(np.asarray(D_s_rand).shape)) + idx, np.asarray(D_s_rand),
               color=(.6, .6, .6),
               s=2, alpha=.2)
    ax.scatter(idx, np.asarray(D_s_rand).mean(), color=(.6, .6, .6), s=20,
               label=f'rand')
    ax.scatter(idx, ev_sent_Ds, color=(1, .7, 0), edgecolor=(.2, .2, .2), s=50,
               label=f'optim')
    ev_neighb_list = indices.get()
    neighb_col = cmap(np.divide(range(ev_neighb_list.shape[1]), ev_neighb_list.shape[1]))
    for idn in range(ev_neighb_list.shape[1] - 1):
        ev_sent_neighbors = list(ev_neighb_list[ev_selected_idx, idn + 1])
        neighbor_Ds = optimizer_obj.gpu_object_function(ev_sent_neighbors)
        ax.scatter(idx, neighbor_Ds, color=neighb_col[idn + 1, :], edgecolor=(.2, .2, .2), s=30,
                   label=f'neigh {idn + 1}')
    #
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.set_xlim((-.8, 2.8))
    ax.set_ylim((.7, 1.1))
    ax.set_xticks([])
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
    ax.legend(bbox_to_anchor=(2.1, .85), frameon=True, fontsize=8)
    ax.set_ylabel(r'$D_s$')
    ax.set_title(
        f'change in Ds when choosing neighbors in ANN_SET1 \n combined models, neighborhood metric: {d_metric}',
        horizontalalignment='left')

    ax = plt.axes((.35, .3, .25, .25))

    temp = np.asarray(ev_sent_ds_all.cpu()).astype(np.single)
    temp1 = np.asarray(neighbor_ds_all.cpu()).astype(np.single)
    vmax_val = np.max([temp.max(), temp1.max()])
    vmin_val = np.min([temp[temp > 0].min(), temp1[temp1 > 0].min(), 0.5])

    im = ax.imshow(temp, cmap='inferno', vmin=vmin_val, vmax=vmax_val)
    yticklabel = model_names
    xticklabel = model_names
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(xticklabel, rotation=0, fontsize=8)

    ax = plt.axes((.7, .3, .25, .25))
    im = ax.imshow(temp1, cmap='inferno', vmin=vmin_val, vmax=vmax_val)
    yticklabel = model_names
    xticklabel = model_names
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=8)

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(xticklabel, rotation=0, fontsize=8)
    fig.colorbar(im, orientation='vertical')
    plt.show()
    plt.savefig(os.path.join(ANALYZE_DIR,
                             f'U01_sent_neighbor_model_{model_name}_metric_{d_metric}.png'),
                dpi=None, facecolor='w',
                edgecolor='w',
                orientation='landscape',
                transparent=True, bbox_inches=None, pad_inches=0.1,
                frameon=False)

    temp1[0,1:]
    temp1[1,2:]
    temp1[:1, 1]
    flatten_corr=[]
    for id in range(temp1.shape[0]):
        #print(temp1[id, (id+1):])
        #print(temp1[:id, id])
        flatten_corr.append(np.concatenate([temp[id, :],temp[:, id]],axis=0))


