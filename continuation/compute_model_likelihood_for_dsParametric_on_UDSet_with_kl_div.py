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
import pickle
import scipy
import seaborn as sns
sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from tqdm import tqdm
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig,AutoModelForMaskedLM
# determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import minicons
from minicons import scorer
import matplotlib
import scipy.io
matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 7,'font.weight':'normal'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False


from sent_sampling.utils import make_shorthand
def add_significance_info(ax, x1, x2, y, height, p_value,added_text=None):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.5, c='black')
    star = ''
    if p_value < 0.05:
        star = '*'
    if p_value < 0.01:
        star = '**'
    if p_value < 0.001:
        star = '***'
    else:
        star = 'n.s.'
    ax.text((x1 + x2) * 0.5, y + height, added_text+'\n'+star, ha='center', va='bottom', fontsize=7)


# determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    #
    # read the excel that contains the selected sentences
    # %%  RUN SANITY CHECKS
    n_samples = 200
    n_selected = 196
    kl_muliplier = 5.0
    kl_threshold = 0.05
    bins = 200
    epsilon = 1e-10
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optimizer_id = f"coordinate_ascent_eh-obj=D_s_kl_div-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    (ext_sh, optim_sh) = make_shorthand(extract_id, optimizer_id)
    save_path = Path(ANALYZE_DIR)
    ax_title = f'sentences,{ext_sh}_Ns={n_samples}_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}'
    df_incl = pd.read_excel(save_path / f'{ax_title}_included.xlsx')

    # chose df_max as from columns ds_max_sent, ds_max_loc, ds_max_include
    df_max = df_incl[['ds_max_sent', 'ds_max_loc', 'ds_max_include']]
    # select rows with ds_max_include == 1
    df_max = df_max[df_max['ds_max_include'] == 1]
    ds_max_loc_incl = df_max['ds_max_loc'].tolist()
    ds_max_sent_incl = df_max['ds_max_sent'].tolist()
    # make it int
    ds_max_loc_incl = [int(x) for x in ds_max_loc_incl]
    # do the same for min
    df_min = df_incl[['ds_min_sent', 'ds_min_loc', 'ds_min_include']]
    df_min = df_min[df_min['ds_min_include'] == 1]
    ds_min_loc_incl = df_min['ds_min_loc'].tolist()
    ds_min_loc_incl = [int(x) for x in ds_min_loc_incl]
    ds_min_sent_incl = df_min['ds_min_sent'].tolist()
    # do the same for rand
    df_rand = df_incl[['ds_rand_sent', 'ds_rand_loc', 'ds_rand_include']]
    df_rand = df_rand[df_rand['ds_rand_include'] == 1]
    ds_rand_loc_incl = df_rand['ds_rand_loc'].tolist()
    ds_rand_loc_incl = [int(x) for x in ds_rand_loc_incl]
    ds_rand_sent_incl = df_rand['ds_rand_sent'].tolist()
    # assert len of ds_min_loc_incl and ds_max_loc_incl is 200
    assert len(ds_min_loc_incl) == n_selected
    assert len(ds_max_loc_incl) == n_selected
    assert len(ds_rand_loc_incl) == n_selected
    # %% MORE SANITY CHECKS FOR THE ACTIVATIONS
    # get the
    ds_min_sent = ds_max_loc_incl
    ds_max_sent = ds_min_sent_incl
    ds_rand_sent = ds_rand_sent_incl
    # laod the extractor
    ext_obj = extract_pool[extract_id]()

    #ext_obj()
    model_names = ext_obj.model_spec

    # %% create a model for causalLM
    ds_sentences=[ds_min_sent,ds_rand_sent,ds_max_sent]
    ds_sentence_cond=['ds_min','ds_rand','ds_max']

    all_models_scores = dict()
    bidirectional_ids = [0, 2, 5]
    unidirect_ids = [1, 3, 4, 6]
    # order 1
    #bidirectional_ids = [0, 2,3, 5]
    #unidirect_ids = [1, 4, 6]
    for bidir_id in bidirectional_ids:
        ds_scores_parametric = []
        #model_ = AutoModelForMaskedLM.from_pretrained(model_names[bidir_id], return_dict=True).to(device)
        #model_tokenizer = AutoTokenizer.from_pretrained(model_names[bidir_id], use_fast=False)
        mlm_model = scorer.MaskedLMScorer(model_names[bidir_id], device)
        # toke=AutoTokenizer.from_pretrained(model_names[bidir_id])
        # mlm_model.tokenizer=toke
        for id_ds, ds_sent in enumerate(ds_sentences):
            stimuli = ds_sent
            # ilm_model.compute_stats(ilm_model.prepare_text(stimuli))
            ds_scores = []
            for stim in tqdm(stimuli):
                #ds_score = mlm_model.sequence_score(stim, reduction= lambda x: -x.sum(0).item(),
               #                                     PLL_metric='within_word_l2r')
                ds_score = mlm_model.sequence_score(stim, reduction= lambda x: -x.mean(0).item(),
                                                    PLL_metric='within_word_l2r')
                ds_scores.append(ds_score[0])
            ds_scores_parametric.append(ds_scores)
            # plot the distirbution of the scores
        all_models_scores[model_names[bidir_id]] = ds_scores_parametric


    # order

    for unidir_id in unidirect_ids:
        ds_scores_parametric=[]
        ilm_model = scorer.IncrementalLMScorer(model_names[unidir_id], device)
        for id_ds,ds_sent in enumerate(ds_sentences):
            stimuli=ds_sent
            #ilm_model.compute_stats(ilm_model.prepare_text(stimuli))
            ds_scores=[]
            for stim in tqdm(stimuli):
                #ds_score=ilm_model.sequence_score(stim, reduction=lambda x: -x.sum(0).item())
                ds_score = ilm_model.sequence_score(stim, reduction=lambda x: -x.mean(0).item())

                ds_scores.append(ds_score[0])
            ds_scores_parametric.append(ds_scores)
            # plot the distirbution of the scores
        all_models_scores[model_names[unidir_id]]=ds_scores_parametric

    ## save model results as
    # save the ANNSet1_ds, ANNSet1_lex, ANNSet1_lex_rand, RDM_max_dict, RDM_rand_dict
    f'model_sum_likelihood,{ext_sh}_Ns={n_samples}_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_included.pkl'
    save_path=Path(ANALYZE_DIR,f'model_sum_likelihood,{ext_sh}_Ns={n_samples}_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_included.pkl')
    # make sure it exists
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(all_models_scores, f)

    # use scipy.io to save it as a mat file
    save_path=Path(ANALYZE_DIR,f'model_sum_likelihood,{ext_sh}_Ns={n_samples}_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_included.mat')
    scipy.io.savemat(save_path.__str__(), all_models_scores)

    model_names_new_order=[0,2,5,3,1,6,4]
    model_names_new = [model_names[i] for i in model_names_new_order]
    processed_dfs = []
    for key  in model_names_new:
        # For each DataFrame, create a new DataFrame with the required structure
        df = all_models_scores[key]
        for idx,col in enumerate(['min','rand','max']):
            temp_df = pd.DataFrame({
                'log-likelihood': df[idx],
                'model': key,
                'group': col
            })
            processed_dfs.append(temp_df)
    final_df = pd.concat(processed_dfs, ignore_index=True)

    colors = [np.divide((255, 153, 51), 255), np.divide((160, 160, 160), 256), np.divide((51, 153, 255), 255)]
    # make a dataset with
    # create a list of labels
    fig = plt.figure(figsize=(11, 8))
    pap_fac=8/11
    # fig_length = 0.055 * len(models_scores)
    #ax = plt.axes((.1, .4, .35, .35))
    ax= fig.add_axes([0.2, 0.3, 0.35, 0.2*pap_fac])
    x = np.arange(len(all_models_scores))
    model_ds_min = np.asarray([all_models_scores[x][0] for x in all_models_scores.keys()])
    # make a panda dataframe with 2 columns, model and distance
    model_sh = ['RoBERTa', 'BERT-L', 'ALBERT-XXL', 'XLNet-L', 'XLM', 'GPT2-XL', 'CTRL']
    # model_ds_min=pd.DataFrame(model_ds_min.T,columns=model_sh)

    #sns.despine(bottom=True, left=True, ax=ax)
    sns.set_theme(style="ticks", palette="pastel")
    #sns.despine(offset=10, trim=True)
    # set width of bar
    barWidth = 0.25
    ax = sns.boxplot(x="model", y="log-likelihood",
                     hue="group", palette=colors, flierprops={"marker": "o"},
                     data=final_df, ax=ax, showmeans=False, meanline=False, showfliers=True, widths=0.2)
    # Add in points to show each observation
    # sns.stripplot(x="distance", y="method", data=planets,
    #              size=4, color=".3", linewidth=0)

    # plot model_ds_min using boxplot with mean and std
    # ax.boxplot(model_ds_min.T, positions=x - .25, showmeans=True, meanline=True, showfliers=False, widths=0.2,
    #            boxprops=dict(color=colors[0]))
    # # plot ds_rand
    # model_ds_rand = np.asarray([all_models_scores[x][1] for x in all_models_scores.keys()])
    # ax.boxplot(model_ds_rand.T, positions=x, showmeans=True, meanline=True, showfliers=False, widths=0.2,
    #            boxprops=dict(color=colors[1]))
    # # plot ds_max
    # model_ds_max = np.asarray([all_models_scores[x][2] for x in all_models_scores.keys()])
    # ax.boxplot(model_ds_max.T, positions=x + .25, showmeans=True, meanline=True, showfliers=False, widths=0.2,
    #            boxprops=dict(color=colors[2]))
    # ax.set_xticks(x)
    ax.set_xticklabels(model_sh, rotation=45)
    ax.set_ylabel('Sequence Log-likehood')
    # make the ticks shortes
    ax.tick_params(axis='x', length=2)
    ax.tick_params(axis='y', length=2)
    # turn of the legend
    ax.get_legend().remove()
    # turn of the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.show()
    save_path = Path(ANALYZE_DIR)

    save_loc = Path(save_path.__str__(), f'ds_parametric_model_logLiklhood_sum.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_parametric_model_logLiklhood_sum.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)

    fig.show()

    for model_name in all_models_scores.keys():
        ds_min=all_models_scores[model_name][0]
        ds_rand=all_models_scores[model_name][1]
        ds_max=all_models_scores[model_name][2]
        a=scipy.stats.ttest_ind(ds_min,ds_rand)
        b=scipy.stats.ttest_ind(ds_min,ds_max)
        c=scipy.stats.ttest_ind(ds_rand,ds_max)
        # print a.stat with 3 decimal points
        print(model_name)
        print('ds_min vs ds_rand s=%.3f, p=%.3f' % (a.statistic, a.pvalue))
        print('ds_min vs ds_max s=%.3f, p=%.3f' % (b.statistic, b.pvalue))
        print('ds_rand vs ds_max s=%.3f, p=%.3f' % (c.statistic, c.pvalue))
