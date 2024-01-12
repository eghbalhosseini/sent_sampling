import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import torch
from scipy.spatial.distance import pdist, squareform
# check if gpu is available
import seaborn as sns
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == '__main__':
    extract_ids = [
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False',
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textPeriod-activation-bench=None-ave=False']
    optim_id_low = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
                 'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True'             ]
    optim_id_full = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
                    'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']

    low_resolution = 'False'
    optim_files = []
    optim_results_low = []
    for ext in extract_ids:
        for optim in optim_id_low:
            optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
            assert(optim_file.exists())
            optim_files.append(optim_file.__str__())
            optim_results_low.append(load_obj(optim_file.__str__()))
    # extrat optim results for full
    optim_results_full = []

    for ext in extract_ids:
        for optim in optim_id_full:
            optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
            assert(optim_file.exists())
            optim_results_full.append(load_obj(optim_file.__str__()))

    # load extract no period
    ext_obj_textNoPeriod=extract_pool[extract_ids[0]]()
    ext_obj_textNoPeriod.load_dataset()
    ext_obj_textNoPeriod()

    ext_obj_textPeriod=extract_pool[extract_ids[1]]()
    ext_obj_textPeriod.load_dataset()
    ext_obj_textPeriod()

   # compute distance on

    optim_low_dim_textPeriod = optim_pool[optim_id_low[0]]()
    optim_low_dim_textNoPeriod = optim_pool[optim_id_low[0]]()

    optim_full_dim_textPeriod=optim_pool[optim_id_full[0]]()
    optim_full_dim_textNoPeriod = optim_pool[optim_id_full[0]]()

    # first load text period extractor
    optim_full_dim_textPeriod.load_extractor(ext_obj_textPeriod)
    optim_full_dim_textPeriod.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=True,
                                             save_results=False)

    optim_low_dim_textPeriod.load_extractor(ext_obj_textPeriod)
    optim_low_dim_textPeriod.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=True,
                                                         save_results=False)

    optim_full_dim_textNoPeriod.load_extractor(ext_obj_textNoPeriod)
    optim_full_dim_textNoPeriod.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=True,
                                                         save_results=False)
    optim_low_dim_textNoPeriod.load_extractor(ext_obj_textNoPeriod)
    optim_low_dim_textNoPeriod.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=True,
                                                         save_results=False)
    # compute ds_rand for each of the optimizers
    DS_RAND=[]
    for optim_ in [optim_full_dim_textPeriod, optim_low_dim_textPeriod, optim_full_dim_textNoPeriod, optim_low_dim_textNoPeriod]:
        ds_rands = []
        for k in tqdm(enumerate(range(200))):
            sent_random = list(np.random.choice(optim_.N_S, optim_.N_s))
            d_s_r, RDM_r = optim_.gpu_object_function_debug(sent_random)
            ds_rands.append(d_s_r)
        DS_RAND.append(ds_rands)


    # compute Ds from optim_full_dim results
    DS_max_on_textPeriod_full_Optim_textNoPeriod, RDM_max = optim_full_dim_textPeriod.gpu_object_function_debug(optim_results_full[0]['optimized_S'])
    DS_min_on_textPeriod_full_Optim_textNoPeriod, RDM_min = optim_full_dim_textPeriod.gpu_object_function_debug(optim_results_full[1]['optimized_S'])

    DS_max_on_textPeriod_full_Optim_textPeriod, RDM_max = optim_full_dim_textPeriod.gpu_object_function_debug(optim_results_full[2]['optimized_S'])
    DS_min_on_textPeriod_full_Optim_textPeriod, RDM_min = optim_full_dim_textPeriod.gpu_object_function_debug(optim_results_full[3]['optimized_S'])

    # load optim_full with textNoPeriod
    DS_max_on_textNoPeriod_full_Optim_textNoPeriod, RDM_max = optim_full_dim_textNoPeriod.gpu_object_function_debug(optim_results_full[0]['optimized_S'])
    DS_min_on_textNoPeriod_full_Optim_textNoPeriod, RDM_min = optim_full_dim_textNoPeriod.gpu_object_function_debug(optim_results_full[1]['optimized_S'])

    DS_max_on_textNoPeriod_full_Optim_textPeriod, RDM_max = optim_full_dim_textNoPeriod.gpu_object_function_debug(optim_results_full[2]['optimized_S'])
    DS_min_on_textNoPeriod_full_Optim_textPeriod, RDM_min = optim_full_dim_textNoPeriod.gpu_object_function_debug(optim_results_full[3]['optimized_S'])


    # do the same for optim_low_dim results
    DS_max_on_textPeriod_low_Optim_textNoPeriod, RDM_max = optim_low_dim_textPeriod.gpu_object_function_debug(optim_results_low[0]['optimized_S'])
    DS_min_on_textPeriod_low_Optim_textNoPeriod, RDM_min = optim_low_dim_textPeriod.gpu_object_function_debug(optim_results_low[1]['optimized_S'])

    DS_max_on_textPeriod_low_Optim_textPeriod, RDM_max = optim_low_dim_textPeriod.gpu_object_function_debug(optim_results_low[2]['optimized_S'])
    DS_min_on_textPeriod_low_Optim_textPeriod, RDM_min = optim_low_dim_textPeriod.gpu_object_function_debug(optim_results_low[3]['optimized_S'])

    # load optim_low_dim with textNoPeriod
    DS_max_on_textNoPeriod_low_Optim_textNoPeriod, RDM_max = optim_low_dim_textNoPeriod.gpu_object_function_debug(optim_results_low[0]['optimized_S'])
    DS_min_on_textNoPeriod_low_Optim_textNoPeriod, RDM_min = optim_low_dim_textNoPeriod.gpu_object_function_debug(optim_results_low[1]['optimized_S'])

    DS_max_on_textNoPeriod_low_Optim_textPeriod, RDM_max = optim_low_dim_textNoPeriod.gpu_object_function_debug(optim_results_low[2]['optimized_S'])
    DS_min_on_textNoPeriod_low_Optim_textPeriod, RDM_min = optim_low_dim_textNoPeriod.gpu_object_function_debug(optim_results_low[3]['optimized_S'])

    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.2, .6, .18, .25))
    ax.scatter(.02 * np.random.normal(size=(np.asarray(len(DS_RAND[0])))) + 0, np.asarray(DS_RAND[0]),
               color=(.6, .6, .6), s=2, alpha=.3)
    rand_mean = np.asarray(DS_RAND[0]).mean()
    ax.scatter(0, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
               label=f'{rand_mean:.4f}', edgecolor='k')
    DS_min=DS_min_on_textPeriod_full_Optim_textPeriod
    DS_max=DS_max_on_textPeriod_full_Optim_textPeriod
    ax.scatter(0, DS_min, color=np.divide((188, 80, 144), 255), s=50, label=f'{DS_min:.4f}', edgecolor='k',linewidth=2)
    # include text in the plot to show the difference between the two

    ax.scatter(0, DS_max, color=np.divide((255, 128, 0), 255), s=50, label=f'{DS_max:.4f}', edgecolor='k',linewidth=2)


    DS_min = DS_min_on_textPeriod_full_Optim_textNoPeriod
    DS_max = DS_max_on_textPeriod_full_Optim_textNoPeriod
    ax.scatter(0, DS_min, color=np.divide((255, 255, 255), 255), s=50, label=f'{DS_min:.4f}', edgecolor=np.divide((188, 80, 144), 255),linewidth=2)

    ax.scatter(0, DS_max, color=np.divide((255, 255, 255), 255), s=50, label=f'{DS_max:.4f}', edgecolor=np.divide((255, 128, 0), 255),linewidth=2)

    # do the same for textNoPeriod
    ax.scatter(1+.02 * np.random.normal(size=(np.asarray(len(DS_RAND[2])))) + 0, np.asarray(DS_RAND[2]),
               color=(.6, .6, .6), s=2, alpha=.3)
    rand_mean = np.asarray(DS_RAND[2]).mean()
    ax.scatter(1, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
               label=f'{rand_mean:.4f}', edgecolor='k',marker='o')
    DS_min = DS_min_on_textNoPeriod_full_Optim_textNoPeriod
    DS_max = DS_max_on_textNoPeriod_full_Optim_textNoPeriod
    ax.scatter(1, DS_min, color=np.divide((188, 80, 144), 255), s=50, label=f'{DS_min:.4f}', edgecolor='k',marker='o',linewidth=2)
    ax.text(1.1, DS_min, f'Ds_min for sentences optimized WO period', fontsize=8, color=np.divide((188, 80, 144), 255),
            ha='left', va='center')
    ax.scatter(1, DS_max, color=np.divide((255, 128, 0), 255), s=50, label=f'{DS_max:.4f}', edgecolor='k',marker='o',linewidth=2)
    ax.text(1.1, DS_max, f'Ds_max for sentences optimized WO period', fontsize=8, color=np.divide((255, 128, 0), 255),
            ha='left', va='center')
    DS_min = DS_min_on_textNoPeriod_full_Optim_textPeriod
    DS_max = DS_max_on_textNoPeriod_full_Optim_textPeriod
    ax.scatter(1, DS_min, color='w', s=50, label=f'{DS_min:.4f}', edgecolor=np.divide((188, 80, 144), 255),marker='o',linewidth=2)
    ax.text(1.1, DS_min, f'Ds_min for sentences optimized W period', fontsize=8, color=np.divide((188, 80, 144), 255),
            ha='left', va='center')
    ax.scatter(1, DS_max, color='w', s=50, label=f'{DS_max:.4f}', edgecolor=np.divide((255, 128, 0), 255),marker='o',linewidth=2)
    ax.text(1.1, DS_max, f'Ds_max for sentences optimized W period', fontsize=8, color=np.divide((188, 80, 144), 255),
            ha='left', va='center')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.4, 1.4))
    ax.set_ylim((0.0, 1.2))
    ax.set_xticks([0,1])
    ax.set_xticklabels(['W Period','WO Period'],rotation=0)
    ax.legend(bbox_to_anchor=(3.6, .8), frameon=True)
    ax.set_ylabel(r'$D_s$')
    ax.set_title('when activation directly used for optimization')
    ax.tick_params(direction='out', length=2, width=1.5, colors='k',
                   grid_color='k', grid_alpha=0.5)


    ax = plt.axes((.2, .1, .18, .25))
    # low dim textPeriod
    ax.scatter(.02 * np.random.normal(size=(np.asarray(len(DS_RAND[1])))) + 0, np.asarray(DS_RAND[1]),
               color=(.6, .6, .6), s=2, alpha=.3)
    rand_mean = np.asarray(DS_RAND[1]).mean()
    ax.scatter(0, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
               label=f'{rand_mean:.4f}', edgecolor='k')
    DS_min = DS_min_on_textPeriod_low_Optim_textPeriod
    DS_max = DS_max_on_textPeriod_low_Optim_textPeriod
    ax.scatter(0, DS_min, color=np.divide((188, 80, 144), 255), s=50, label=f'{DS_min:.4f}', edgecolor='k',linewidth=2)
    ax.scatter(0, DS_max, color=np.divide((255, 128, 0), 255), s=50, label=f'{DS_max:.4f}', edgecolor='k',linewidth=2)

    DS_min = DS_min_on_textPeriod_low_Optim_textNoPeriod
    DS_max = DS_max_on_textPeriod_low_Optim_textNoPeriod
    ax.scatter(0, DS_min, color=np.divide((255, 255, 255), 255), s=50, label=f'{DS_min:.4f}', edgecolor=np.divide((188, 80, 144), 255),linewidth=2)
    ax.scatter(0, DS_max, color=np.divide((255, 255, 255), 255), s=50, label=f'{DS_max:.4f}', edgecolor=np.divide((255, 128, 0), 255),linewidth=2)


    # do the same for textNoPeriod

    ax.scatter(1+.02 * np.random.normal(size=(np.asarray(len(DS_RAND[3])))) + 0, np.asarray(DS_RAND[3]),
               color=(.6, .6, .6), s=2, alpha=.3)
    rand_mean = np.asarray(DS_RAND[3]).mean()
    ax.scatter(1, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
               label=f'{rand_mean:.4f}', edgecolor='k',marker='s')
    DS_min = DS_min_on_textNoPeriod_low_Optim_textNoPeriod
    DS_max = DS_max_on_textNoPeriod_low_Optim_textNoPeriod
    ax.scatter(1, DS_min, color=np.divide((188, 80, 144), 255), s=50, label=f'{DS_min:.4f}', edgecolor='k',marker='o',linewidth=2)
    ax.scatter(1, DS_max, color=np.divide((255, 128, 0), 255), s=50, label=f'{DS_max:.4f}', edgecolor='k',marker='o',linewidth=2)

    DS_min = DS_min_on_textNoPeriod_low_Optim_textPeriod
    DS_max = DS_max_on_textNoPeriod_low_Optim_textPeriod
    ax.scatter(1, DS_min, color='w', s=50, label=f'{DS_min:.4f}', edgecolor=np.divide((188, 80, 144), 255),marker='o',linewidth=2)
    ax.scatter(1, DS_max, color='w', s=50, label=f'{DS_max:.4f}', edgecolor=np.divide((255, 128, 0), 255),marker='o',linewidth=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.4, 1.4))
    ax.set_ylim((0.0, 1.2))
    ax.set_xticks([0,1])
    ax.set_xticklabels(['W Period','WO Period'],rotation=0)

    ax.legend(bbox_to_anchor=(1.1, .8), frameon=True)
    ax.set_title('when a low dimensional \nactivation (PCA) used for optimization')
    ax.set_ylabel(r'$D_s$')
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)


    fig.show()

    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'role_of_W_and_WO_in_optimization_results.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'role_of_W_and_WO_in_optimization_results.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)


    #%%
    # get sentences and check if there is an overlap between the sentences that are used for the optimization
    extract_ids = [
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False',
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textPeriod-activation-bench=None-ave=False']
    ext_obj_textPeriod
    ext_obj_textNoPeriod
    optim_id_low = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
                 'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True'             ]
    optim_id_full = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
                    'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']

    k=0
    ext_objs=[ext_obj_textNoPeriod,ext_obj_textPeriod]
    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_low[0]}.pkl").__str__())['optimized_S']
    select_activations=[]
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_max_textNoPeriod_lowDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id, columns=ext_objs[k].model_spec)

    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_low[1]}.pkl").__str__())['optimized_S']
    select_activations = []
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_min_textNoPeriod_lowDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id,
                                              columns=ext_objs[k].model_spec)

    k=1
    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_low[0]}.pkl").__str__())['optimized_S']
    select_activations = []
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_max_textPeriod_lowDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id,
                                              columns=ext_objs[k].model_spec)

    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_low[1]}.pkl").__str__())['optimized_S']
    select_activations = []
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_min_textPeriod_lowDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id,
                                              columns=ext_objs[k].model_spec)

    #%%

    k = 0
    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_full[0]}.pkl").__str__())['optimized_S']
    select_activations = []
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_max_textNoPeriod_fullDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id,
                                              columns=ext_objs[k].model_spec)

    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_full[1]}.pkl").__str__())['optimized_S']
    select_activations = []
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_min_textNoPeriod_fullDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id,
                                               columns=ext_objs[k].model_spec)

    k = 1
    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_full[0]}.pkl").__str__())['optimized_S']
    select_activations = []
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_max_textPeriod_fullDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id,
                                            columns=ext_objs[k].model_spec)

    S_id = load_obj(Path(RESULTS_DIR, f"results_{extract_ids[k]}_{optim_id_full[1]}.pkl").__str__())['optimized_S']
    select_activations = []
    for model_act in ext_objs[k].model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])
    df_min_textPeriod_fullDim = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id,
                                             columns=ext_objs[k].model_spec)

    # get the sentences from first column of df_max_textNoPeriod_lowDim
    # and check if they are in the df_max_textPeriod_lowDim
    S0 = df_max_textNoPeriod_fullDim['roberta-base'].values
    S1 = df_max_textPeriod_fullDim['roberta-base'].values
    # drop the '.' from elements in S1 if exist
    S1_nodot = [s[:-1] for s in S1 if s[-1] == '.']
    # check if there is overlap between S1_nodot and S0
    overlap_max_full_dim = [s for s in S0 if s in S1_nodot]

    S0 = df_min_textNoPeriod_fullDim['roberta-base'].values
    S1 = df_min_textPeriod_fullDim['roberta-base'].values
    # drop the '.' from elements in S1 if exist
    S1_nodot = [s[:-1] for s in S1 if s[-1] == '.']
    # check if there is overlap between S1_nodot and S0
    overlap_min_full_dim = [s for s in S0 if s in S1_nodot]

    S0 = df_max_textNoPeriod_lowDim['roberta-base'].values
    S1 = df_max_textPeriod_lowDim['roberta-base'].values
    # drop the '.' from elements in S1 if exist
    S1_nodot = [s[:-1] for s in S1 if s[-1] == '.']
    # check if there is overlap between S1_nodot and S0
    overlap_max_low_dim = [s for s in S0 if s in S1_nodot]

    S0 = df_min_textNoPeriod_lowDim['roberta-base'].values
    S1 = df_min_textPeriod_lowDim['roberta-base'].values
    # drop the '.' from elements in S1 if exist
    S1_nodot = [s[:-1] for s in S1 if s[-1] == '.']
    # check if there is overlap between S1_nodot and S0
    overlap_min_low_dim = [s for s in S0 if s in S1_nodot]

    S0 = df_max_textNoPeriod_fullDim['roberta-base'].values
    S1 = df_max_textNoPeriod_lowDim['roberta-base'].values
    # check if there is overlap between S1_nodot and S0
    overlap_max_textNoPeriod = [s for s in S0 if s in S1]

    S0 = df_min_textNoPeriod_fullDim['roberta-base'].values
    S1 = df_min_textNoPeriod_lowDim['roberta-base'].values
    # drop the '.' from elements in S1 if exist
    overlap_min_textNoPeriod = [s for s in S0 if s in S1]
    # check if there is overlap between S1_nodot and S0

    S0 = df_min_textPeriod_fullDim['roberta-base'].values
    S1 = df_min_textPeriod_lowDim['roberta-base'].values
    # drop the '.' from elements in S1 if exist
    overlap_min_textPeriod = [s for s in S0 if s in S1]
    # check if there is overlap between S1_nodot and S0

    S0 = df_max_textPeriod_fullDim['roberta-base'].values
    S1 = df_max_textPeriod_lowDim['roberta-base'].values
    # drop the '.' from elements in S1 if exist
    overlap_max_textPeriod = [s for s in S0 if s in S1]
    # check if there is overlap between S1_nodot and S0
