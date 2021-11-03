import warnings

import numpy as np
from utils import extract_pool
from utils.extract_utils import model_extractor
from utils.optim_utils import optim_pool
import argparse
import utils.optim_utils
from utils.extract_utils import model_extractor, model_extractor_parallel, SAVE_DIR
from utils.data_utils import SENTENCE_CONFIG

from utils.optim_utils import optim, optim_pool, pt_create_corr_rdm_short, optim_group
from utils.data_utils import RESULTS_DIR, save_obj, load_obj, ANALYZE_DIR
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import importlib
import matplotlib
import os
import warnings

#
if __name__ == '__main__':



    extract_name = 'gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layers-dataset=coca_spok_filter_punct_10K_sample_ev_editsOct16-activation-bench=None-ave=False'
    optim_id='coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=200-n_init=1-run_gpu=True'
    optim_fname = os.path.join(RESULTS_DIR,f'results_{extract_name}_{optim_id}_low_dim_gpu.pkl')
    if os.path.exists(optim_fname):
        optim_result=load_obj(optim_fname)
    else:
        warnings.warn('file doesnt exist')


    extract_id = optim_result['extractor_grp_name']
    optimizer_obj = optim_pool[optim_id]()

    optim_group_obj = optim_group(n_init=optimizer_obj.n_init, extract_group_name=extract_name,
                                  ext_group_ids=extract_id,
                                  n_iter=optimizer_obj.n_iter,
                                  N_s=200,
                                  objective_function=optimizer_obj.objective_function,
                                  optim_algorithm=optimizer_obj.optim_algorithm,
                                  run_gpu=optimizer_obj.run_gpu)


    select_sent = []
    values = []
    for name in extract_id:
        ext_obj = extract_pool[name]()
        ext_obj.load_dataset()
        [values.append([id, ext_obj.data_[id]['text']]) for id in np.sort(optim_result['optimized_S'])]
        with open(os.path.join(RESULTS_DIR, f"sentences_{name}_{optim_id}.txt"), 'w') as f:
            for item in values:
                f.write("%d, %s\n" % (item[0], item[1]))

    D_precompute_path=os.path.join(SAVE_DIR,f'{extract_name}_XY_corr_list.pkl')
    if os.path.exists(D_precompute_path):
        D_precompute = load_obj(D_precompute_path)
        optim_group_obj.grp_XY_corr_list = D_precompute['grp_XY_corr_list']
        optim_group_obj.N_S = D_precompute['N_S']
    else:
        optim_group_obj.load_extr_grp_and_corr_rdm_in_low_dim()
    # get final optimal values:
    sent_optim=optim_result['optimized_S']

    optim_score_oct16 = optim_group_obj.gpu_obj_function(sent_optim)
    # get the score for individual models:
    d_optim_ev_list = []
    S = sent_optim
    for XY_corr_list in optim_group_obj.grp_XY_corr_list:
        d_optim_ev_list.append(optim_group_obj.XY_corr_obj_func(S, XY_corr_list=XY_corr_list))

    # get the results for a random set wtithin selected sentencse:
    ds_rand_oct16 = []
    for k in tqdm(enumerate(range(250))):
        sent_random = list(np.random.choice(optim_group_obj.N_S, optim_group_obj.N_s))
        ds_rand_oct16.append(optim_group_obj.gpu_obj_function(sent_random))

    ds_rand_oct16_list = []
    for k in tqdm(enumerate(range(250))):
        ds_rand_ = []
        S = list(np.random.choice(optim_group_obj.N_S, optim_group_obj.N_s))
        for XY_corr_list in optim_group_obj.grp_XY_corr_list:
            ds_rand_.append(optim_group_obj.XY_corr_obj_func(S, XY_corr_list=XY_corr_list))
        ds_rand_oct16_list.append(ds_rand_)


    # get previous reults
    # get the random set
    optim_ids = ['coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=250-n_init=1-run_gpu=True']
    results_files = [
        'results_gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layers-dataset=coca_spok_filter_punct_10K_sample_1-activation-bench=None-ave=False_coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=250-n_init=1-run_gpu=True_low_dim_gpu.pkl', ]
    optim_files = []
    optim_results = []
    for result in results_files:
        optim_file = os.path.join(RESULTS_DIR, result)
        optim_files.append(optim_file)
        optim_results.append(load_obj(optim_file))

    res = optim_results[0]
    extract_grp = res['extractor_grp_name']
    extract_name = 'gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layers'
    optimizer_obj = optim_pool[res['optimizatin_name']]()
    optim_group_obj = optim_group(n_init=optimizer_obj.n_init, extract_group_name=extract_name,
                                  ext_group_ids=extract_grp,
                                  n_iter=optimizer_obj.n_iter,
                                  N_s=optimizer_obj.N_s,
                                  objective_function=optimizer_obj.objective_function,
                                  optim_algorithm=optimizer_obj.optim_algorithm,
                                  run_gpu=optimizer_obj.run_gpu)
    D_precompute = load_obj(os.path.join(SAVE_DIR,
                                         f"gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layers-dataset=coca_spok_filter_punct_10K_sample_1-activation-bench=None-ave=False_XY_corr_list.pkl"))


    optim_group_obj.grp_XY_corr_list = D_precompute['grp_XY_corr_list']
    optim_group_obj.N_S = D_precompute['N_S']

    sent_optim = optim_results[0]['optimized_S']
    optim_score = optim_group_obj.gpu_obj_function(sent_optim)

    ds_rand = []
    for k in tqdm(enumerate(range(250))):
        sent_random = list(np.random.choice(optim_group_obj.N_S, optim_group_obj.N_s))
        ds_rand.append(optim_group_obj.gpu_obj_function(sent_random))

    d_optim_list = []
    S = sent_optim
    for XY_corr_list in optim_group_obj.grp_XY_corr_list:
        d_optim_list.append(optim_group_obj.XY_corr_obj_func(S, XY_corr_list=XY_corr_list))


    ds_rand_list = []
    for k in tqdm(enumerate(range(250))):
        ds_rand_ = []
        S = list(np.random.choice(optim_group_obj.N_S, optim_group_obj.N_s))
        for XY_corr_list in optim_group_obj.grp_XY_corr_list:
            ds_rand_.append(optim_group_obj.XY_corr_obj_func(S, XY_corr_list=XY_corr_list))
        ds_rand_list.append(ds_rand_)


    # plot the results
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(26, 14), dpi=100, frameon=False)
    ax = plt.axes((.1, .3, .02, .45))

    cmap = cm.get_cmap('viridis_r')

    alph_col = cmap(np.divide(range(len(optim_results)), len(optim_results)))
    tick_l = []
    tick = []
    idx = 0
    D_s_rand = ds_rand
    ax.scatter(.2 * np.random.normal(size=(np.asarray(D_s_rand).shape)) + idx, np.asarray(D_s_rand), color=(.6, .6, .6),
               s=2, alpha=.2)
    ax.scatter(idx, np.asarray(D_s_rand).mean(), color=(.6, .6, .6), s=20, label=f'random, size={optim_group_obj.N_s}')
    ax.scatter(idx, res['optimized_d'], color=(1, .7, 0),edgecolor=(.2,.2,.2),  s=50, label=f'optimized, size={optim_group_obj.N_s}')
    #
    idx=2
    D_s_rand = ds_rand_oct16
    ax.scatter(.2 * np.random.normal(size=(np.asarray(D_s_rand).shape)) + idx, np.asarray(D_s_rand), color=(.6, .4, .4),
               s=2, alpha=.1)
    ax.scatter(idx, np.asarray(D_s_rand).mean(), color=(.4, 0, 0),edgecolor=(1,1,1) ,linewidth=.5,s=30, label=f'random set from ev selected, size=200')
    ax.scatter(idx, optim_score_oct16, color=(1, 0, 0),edgecolor=(0,0,0),  s=50, label=f'optimized from ev selected, size=200')


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.set_xlim((-.8, 2.8))
    ax.set_ylim((0, 1.1))
    ax.set_xticks([])
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    ax.legend(bbox_to_anchor=(5.1, .85), frameon=True)
    ax.set_ylabel(r'$D_s$')
    # full set
    model_names = [re.findall('.+_layers', x)[0][0:-7] for x in extract_grp]
    ax = plt.axes((.5, .3, .2, .45))
    cmap = cm.get_cmap('viridis_r')
    alph_col = cmap(np.divide(range(len(optim_results)), len(optim_results)))
    tick_l = []
    tick = []
    idx = 0
    for idx, _ in enumerate(d_optim_ev_list):
        # original set
        Ds_rand = np.asarray(ds_rand_list)[:, idx]
        ax.scatter(.1 * np.random.normal(size=(np.asarray(Ds_rand).shape)) + idx, np.asarray(Ds_rand),
                   color=(.6, .6, .6), s=2, alpha=.3,zorder=0)
        ax.scatter(idx, np.asarray(Ds_rand).mean(), color=(.4, .4, .4), s=20,zorder=2)
        ax.scatter(idx, d_optim_list[idx], color=(1, .7, 0), s=50,edgecolor=(.2,.2,.2), label=f'optimized',zorder=4)
        # ev selected set
        Ds_rand_ev = np.asarray(ds_rand_oct16_list)[:, idx]
        ax.scatter(.1 * np.random.normal(size=(np.asarray(Ds_rand_ev).shape)) + idx, np.asarray(Ds_rand_ev),
                   color=(.6, .4, .4),s=2, alpha=.1,zorder=0)
        ax.scatter(idx, np.asarray(Ds_rand_ev).mean(),color=(.4, 0, 0),edgecolor=(1,1,1) ,linewidth=.5,s=30,zorder=2)
        ax.scatter(idx, d_optim_ev_list[idx], color=(1, 0, 0),edgecolor=(0,0,0), s=50, label=f'ev filtered',zorder=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    # ax.set_xlim((0,6))
    # ax.set_ylim((0,1))
    ax.set_xticks(list(range(len(d_optim_ev_list))))
    ax.set_xticklabels(model_names, rotation=90)

    ax.set_title('Ds for each model')


    plt.savefig(os.path.join(ANALYZE_DIR, f'optimized_ev_oct16_{extract_name}_{optim_id}_low_dim_gpu.pkl'.replace('.pkl','.pdf')), dpi=None, facecolor='w',
                edgecolor='w',
                orientation='landscape',
                transparent=True, bbox_inches=None, pad_inches=0.1,
                frameon=False)