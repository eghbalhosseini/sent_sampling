import numpy as np
from utils import extract_pool
from utils.extract_utils import model_extractor
from utils.optim_utils import optim_pool, optim_group, optim
import argparse
from utils.extract_utils import model_extractor, model_extractor_parallel
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import RESULTS_DIR, save_obj, load_obj, SAVE_DIR, ANALYZE_DIR
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

#
if __name__ == '__main__':

    # file_name = 'sentence_group=gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_ev_editSept12.csv'
    # df_ev_selected = pd.read_csv(os.path.join(RESULTS_DIR, f"{file_name}"))
    # sent_str_filtered_from_ev=list(df_ev_selected[df_ev_selected['use?']==1]['sentence'])
    # set_num_filtered_from_ev = list(df_ev_selected[df_ev_selected['use?'] == 1]['set_num'])
    # # find sentences in COCA corpus :
    # uniq_set_num=np.unique(set_num_filtered_from_ev)
    # sentence_sets=[f'coca_spok_filter_punct_10K_sample_{x}' for x in uniq_set_num]
    # loc_in_sent_config=[[x['name'] for x in SENTENCE_CONFIG].index(y) for y in sentence_sets]
    # file_loc=[SENTENCE_CONFIG[x]['file_loc'] for x in loc_in_sent_config]
    # data_=[]
    # for file in file_loc:
    #     data_.append(load_obj(file))
    # len(data_)
    # data_=list(np.ravel(data_))
    # all_sentences=[x['text'] for x in data_]
    # corres=[all_sentences.index(x) for x in sent_str_filtered_from_ev]
    # data_ev=[data_[x] for x in corres]
    # construct an extractor object and get the sentences representations :
    sentence_grp='coca_spok_filter_punct_10K_sample_ev_editsSep12'
    # save the data
    # ev_data_file=f'{RESULTS_DIR}/{sentence_grp}'
    # with open(ev_data_file,'wb') as fout:
    #     pickle.dump(data_ev, fout)


    extract_name = f'gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layers-dataset={sentence_grp}'
    extract_id = [f'group=gpt2-xl_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=ctrl_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=bert-large-uncased-whole-word-masking_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=gpt2_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=openaigpt_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                 f'group=lm_1b_layers-dataset={sentence_grp}-activation-bench=None-ave=False']

    optim_ids = 'coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=200-n_init=1-run_gpu=False'
    optimizer_obj = optim_pool[optim_ids]()

    optim_group_obj = optim_group(n_init=optimizer_obj.n_init, extract_group_name=extract_name,
                                  ext_group_ids=extract_id,
                                  n_iter=optimizer_obj.n_iter,
                                  N_s=200,
                                  objective_function=optimizer_obj.objective_function,
                                  optim_algorithm=optimizer_obj.optim_algorithm,
                                  run_gpu=optimizer_obj.run_gpu)

    optim_group_obj.load_extr_grp_and_corr_rdm_in_low_dim()


    sent_random = list(np.random.choice(optim_group_obj.N_S, optim_group_obj.N_s))
    new_score=optim_group_obj.gpu_obj_function(sent_random)
    # individual models
    # ev selected
    d_optim_ev_list = []
    S = sent_random
    for XY_corr_list in optim_group_obj.grp_XY_corr_list:
        d_optim_ev_list.append(optim_group_obj.XY_corr_obj_func(S, XY_corr_list=XY_corr_list))

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
    ds_rand = []
    ds_rand_ev = []
    for k in tqdm(enumerate(range(250))):
        sent_random = list(np.random.choice(optim_group_obj.N_S, optim_group_obj.N_s))
        ds_rand.append(optim_group_obj.gpu_obj_function(sent_random))



    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(26, 14), dpi=100, frameon=False)
    ax = plt.axes((.1, .5, .02, .25))

    cmap = cm.get_cmap('viridis_r')

    alph_col = cmap(np.divide(range(len(optim_results)), len(optim_results)))
    tick_l = []
    tick = []
    idx = 0
    D_s_rand = ds_rand
    ax.scatter(.2 * np.random.normal(size=(np.asarray(D_s_rand).shape)) + idx, np.asarray(D_s_rand), color=(.6, .6, .6),
               s=2, alpha=.3)
    ax.scatter(idx, np.asarray(D_s_rand).mean(), color=(0, 0, 0), s=50, label=f'random, size={optim_group_obj.N_s}')



    ax.scatter(idx, res['optimized_d'], color=(1, 0, 0), s=50, label=f'optimized, size={optim_group_obj.N_s}')
    ax.scatter(idx, new_score, color=(0, 1, 0), s=50, label=f'ev filtered, size={200}')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.set_xlim((-.8, .8))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    ax.legend(bbox_to_anchor=(5.1, .85), frameon=True)
    ax.set_ylabel(r'$D_s$')

    plt.savefig(os.path.join(ANALYZE_DIR, sentence_grp[0].replace('.pkl', '.pdf')), dpi=None, facecolor='w',
                edgecolor='w',
                orientation='landscape',
                transparent=True, bbox_inches=None, pad_inches=0.1,
                frameon=False)

    # full set
    d_optim_list = []
    S = res['optimized_S']
    for XY_corr_list in optim_group_obj.grp_XY_corr_list:
        d_optim_list.append(optim_group_obj.XY_corr_obj_func(S, XY_corr_list=XY_corr_list))

    ds_rand_list = []
    for k in tqdm(enumerate(range(250))):
        ds_rand_ = []
        S = list(np.random.choice(optim_group_obj.N_S, optim_group_obj.N_s))
        for XY_corr_list in optim_group_obj.grp_XY_corr_list:
            ds_rand_.append(optim_group_obj.XY_corr_obj_func(S, XY_corr_list=XY_corr_list))
        ds_rand_list.append(ds_rand_)

    model_names = [re.findall('.+_layers', x)[0][0:-7] for x in extract_grp]
    model_names



    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(26, 14), dpi=100, frameon=False)

    ax = plt.axes((.1, .5, .02, .25))

    cmap = cm.get_cmap('viridis_r')

    alph_col = cmap(np.divide(range(len(optim_results)), len(optim_results)))
    tick_l = []
    tick = []
    idx = 0
    D_s_rand = ds_rand
    ax.scatter(.2 * np.random.normal(size=(np.asarray(D_s_rand).shape)) + idx, np.asarray(D_s_rand), color=(.6, .6, .6),
               s=2, alpha=.3)
    ax.scatter(idx, np.asarray(D_s_rand).mean(), color=(0, 0, 0), s=50, label=f'random, size={250}')




    ax.scatter(idx, res['optimized_d'], color=(1, 0, 0), s=50, label=f'optimized, size={250}')
    ax.scatter(idx, new_score, color=(0, 1, 0), s=50, label=f'ev filtered, size={200}')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.set_xlim((-.8, .8))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    ax.legend(bbox_to_anchor=(2.1, .85), frameon=True)
    ax.set_ylabel(r'$D_s$')

    ax.set_title('average Ds')

    ax = plt.axes((.3, .5, .2, .25))

    cmap = cm.get_cmap('viridis_r')

    alph_col = cmap(np.divide(range(len(optim_results)), len(optim_results)))
    tick_l = []
    tick = []
    idx = 0
    for idx, _ in enumerate(d_optim_ev_list):
        Ds_rand = np.asarray(ds_rand_list)[:, idx]
        ax.scatter(.1 * np.random.normal(size=(np.asarray(Ds_rand).shape)) + idx, np.asarray(Ds_rand),
                   color=(.6, .6, .6), s=2, alpha=.3)
        ax.scatter(idx, np.asarray(Ds_rand).mean(), color=(0, 0, 0), s=50)

        ax.scatter(idx, d_optim_list[idx], color=(1, 0, 0), s=50, label=f'optimized')
        ax.scatter(idx, d_optim_ev_list[idx], color=(0, 1, 0), s=50, label=f'ev filtered')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    # ax.set_xlim((0,6))
    # ax.set_ylim((0,1))
    ax.set_xticks(list(range(len(d_optim_ev_list))))
    ax.set_xticklabels(model_names, rotation=90)
    # ax.tick_params(direction='out', length=3, width=2, colors='k',
    #                grid_color='k', grid_alpha=0.5)

    ax.set_title('Ds for each model')

    # ax.legend(bbox_to_anchor=(5.1, .85), frameon=True)
    # ax.set_ylabel(r'$D_s$')

    plt.savefig(os.path.join(ANALYZE_DIR, sentence_grp+'.pdf'), dpi=None, facecolor='w',
                edgecolor='w',
                orientation='landscape',
                transparent=True, bbox_inches=None, pad_inches=0.1,
                frameon=False)