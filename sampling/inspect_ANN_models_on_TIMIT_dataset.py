import os
import numpy as np
import sys
from pathlib import Path
import getpass
if getpass.getuser() == 'eghbalhosseini':
    SAMPLING_PARENT = '/Users/eghbalhosseini/MyCodes/sent_sampling'
    SAMPLING_DATA = '/Users/eghbalhosseini/MyCodes//fmri_DNN/ds_parametric/'

elif getpass.getuser() == 'ehoseini':
    SAMPLING_PARENT = '/om/user/ehoseini/sent_sampling'
    SAMPLING_DATA = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'


deepjuice_path='/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project, optim
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
import scipy.io

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib
import scipy.io
from glob import glob
from scipy.stats import ks_2samp
from scipy.stats import shapiro, anderson, kstest, norm, probplot

matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 10,'font.weight':'bold'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False

import pickle
from glob import glob
if __name__ == '__main__':
    n_samples=80
    extract_id = 'group=best_performing_pereira_1-dataset=timit_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id = f'coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'

    ext_obj=extract_pool[extract_id]()
    #deepjuice_identifier=f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    #ext_obj.identifier=deepjuice_identifier
    selected_models=['torchvision_alexnet_imagenet1k_v1',
                    'torchvision_regnet_x_800mf_imagenet1k_v2',
                     'openclip_vit_b_32_laion2b_e16',
                     'timm_swinv2_cr_tiny_ns_224',
                     'torchvision_efficientnet_b1_imagenet1k_v2',
                     'clip_rn50',
                     'timm_convnext_large_in22k',
                     ]

    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
    ext_obj.load_dataset()
    ext_obj()

    optim_obj=optim_pool[optim_id]()
    optim_obj.N_S=len(np.unique(ext_obj.data_.sentence_number))
    optim_obj.load_extractor(ext_obj)
    optim_obj.early_stopping=False
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                                 save_results=False)
    # read the excel that contains the selected sentences
    # %%  Load ds min and ds max data from NSD_caption_optimization
    # create a random set of 80 sentences from the range  of optim_obj.N_S
    # over 1000 iterations sample d_s_rand from the range of optim_obj.N_S
    ds_rand_set=[]
    for i in range(1000):
        ds_rand_loc = np.random.choice(np.arange(0, optim_obj.N_S), optim_obj.N_s, replace=False)
        # compute
        d_s_rand, RDM_rand = optim_obj.gpu_object_function_debug(ds_rand_loc)
        ds_rand_set.append(d_s_rand)
    # compute a ds_max set
    #S_opt_d, DS_opt_d = optim_obj()
    #%%
    extract_id_ud = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    ext_obj_ud = extract_pool[extract_id_ud]()
    ext_obj_ud.load_dataset()
    ext_obj_ud()
    optim_obj_ud = optim_pool[optim_id]()
    optim_obj_ud.load_extractor(ext_obj_ud)
    optim_obj_ud.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                             save_results=False)
    # over 1000 iterations sample d_s_rand from the range of optim_obj.N_S
    ds_rand_set_ud = []
    for i in range(1000):
        ds_rand_loc_ud = np.random.choice(np.arange(0, optim_obj_ud.N_S), optim_obj_ud.N_s, replace=False)
        d_s_rand_ud, RDM_rand_ud = optim_obj_ud.gpu_object_function_debug(ds_rand_loc_ud)
        ds_rand_set_ud.append(d_s_rand_ud)

    #%%
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255)]
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio = 8 / 11
    ax = plt.axes((.2, .6, .6, .25 * pap_ratio))
    # plot the histogram of ds_rand_set
    ax.hist(2-np.asarray(ds_rand_set), bins=20, color=colors[0], alpha=.5, edgecolor='k', linewidth=1,label='TIMIT')
    ax.hist(2-np.asarray(ds_rand_set_ud), bins=20, color=colors[1], alpha=.5, edgecolor='k', linewidth=1,label='Univ. Dep.')
    # plot the average of the ds_rand_set
    ax.axvline(x=2-np.mean(ds_rand_set), color=colors[0], linestyle='-', linewidth=2,)
    ax.axvline(x=2-np.mean(ds_rand_set_ud), color=colors[1], linestyle='-', linewidth=2, )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlabel('model agreement')
    ax.set_ylabel('Frequency')
    ax.set_xlim([.9, 1.3])
    ax.legend()
    fig.show()


    save_path = Path(ANALYZE_DIR)
    (ext_sh,optim_sh)=make_shorthand(extract_id, optim_id)
    save_loc = Path(save_path.__str__(), f'ds_models_on_{ext_sh}_samples_{n_samples}_vs_UD.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_models_on_{ext_sh}_samples_{n_samples}_vs_UD.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)


    #%%
    ds_rand_loc = np.random.choice(np.arange(0, optim_obj.N_S), optim_obj.N_S, replace=False)
    # compute
    d_s_full, RDM_full = optim_obj.gpu_object_function_debug(ds_rand_loc)


    ds_full_set_ud = []
    for i in tqdm(range(1000)):
        ds_rand_loc_ud = np.random.choice(np.arange(0, optim_obj_ud.N_S), optim_obj.N_S, replace=False)
        d_s_rand_ud, RDM_rand_ud = optim_obj_ud.gpu_object_function_debug(ds_rand_loc_ud)
        ds_full_set_ud.append(d_s_rand_ud)

    # plot the
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255)]
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio = 8 / 11
    ax = plt.axes((.2, .6, .6, .25 * pap_ratio))
    # plot the histogram of ds_rand_set
    ax.hist(2 - np.asarray(ds_full_set_ud), bins=20, color=colors[1], alpha=.5, edgecolor='k', linewidth=1,
            label='Univ. Dep.')
    # plot the average of the ds_rand_set
    ax.axvline(x=2 - np.mean(d_s_full), color=colors[0], linestyle='-', linewidth=2,label='TIMIT' )
    ax.axvline(x=2 - np.mean(ds_full_set_ud), color='k', linestyle='-', linewidth=2, )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlabel('model agreement')
    ax.set_ylabel('Frequency')
    ax.set_xlim([.9, 1.3])
    ax.legend()
    fig.show()

    save_path = Path(ANALYZE_DIR)
    (ext_sh, optim_sh) = make_shorthand(extract_id, optim_id)
    save_loc = Path(save_path.__str__(), f'ds_models_on_{ext_sh}_Full_set_vs_UD.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_models_on_{ext_sh}_Full_set_vs_UD.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)