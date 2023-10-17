import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from utils import extract_pool
from utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
from matplotlib.pyplot import GridSpec
import pandas as pd
from pathlib import Path
import torch
from utils.curvature_utils import compute_model_activations, compute_model_curvature
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer
import scipy as sp
from minicons import scorer
if __name__ == '__main__':
    dataset = 'ud_sentencez_token_filter_v3_wordFORM'
    modelclass = 'gpt2'
    modelname = 'gpt2-xl'
    masked = False
    dataset = 'ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']

    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences = [x['text'] for x in ext_obj.data_]
    sentences_words = [x['word_FORM'] for x in ext_obj.data_]
    good_sent_id = np.where(np.asarray([len(x['word_FORM']) == len(x['surprisal_3']) for x in ext_obj.data_]))[0]
    sentences_ = [sentences[i] for i in good_sent_id]

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModel.from_pretrained(modelname)
    model.cuda()

    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

    print('getting activations for model: {}'.format(modelname))
    all_layers=compute_model_activations(model,indexed_tokens)
    # printe that we are getting curvature
    print('getting curvature for model: {}'.format(modelname))
    curvature_dict=compute_model_curvature(all_layers)

    mincons_model = scorer.IncrementalLMScorer(modelname, device='cuda')

    mincons_model.prepare_text(sentences_[1])
    model_cond_p=[]
    model_surp=[]
    for idx, sentence in tqdm(enumerate(sentences),total=len(sentences)):
        True
        cond_p = mincons_model.compute_stats(mincons_model.prepare_text(sentence))
        modl_surp = mincons_model.token_score(sentence, surprisal=True)
        model_cond_p.append(cond_p)
        model_surp.append(modl_surp)

    # compute the average seutnece surprisal for model

    model_surp_avg = np.asarray([np.mean([y[1] for y in x[0]][2:]) for x in model_surp])
    model_cond_p_avg = np.asarray([np.mean([y for y in x[0]][2:]) for x in model_cond_p])
    fig = plt.figure(figsize=(4, 3), frameon=False)
    curve = np.stack(curvature_dict['curve'])
    num_colors = len(curve) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .45, .35))
    for i, curv in tqdm(enumerate(curve)):
        # find if curv or tot_surprise_ave has nan values
        # if so, remove them from both
        nan_idx = np.logical_or(np.isnan(curv), np.isnan(model_surp_avg))
        curv = curv[~nan_idx]
        tot_surprise_ave_ = np.array(model_surp_avg)[~nan_idx]
        # if tot_surprise_ave contains nan drop it and adijst the curv

        r, p = sp.stats.pearsonr(tot_surprise_ave_, curv)
        # ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        #        transform=ax.transAxes,fontsize=6)
        if p < 1e-2:
            ax.scatter(i, r, s=10, color=line_cols[i, :], zorder=4, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        else:
            ax.scatter(i, r, s=10, color=(1, 1, 1), zorder=4, edgecolor=(0, 0, 0), linewidth=.5)

    #ax.set_ylim((-.06, .21))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelname} \n {dataset} \n correlation between curvature and model",fontsize=8)
    # set xtick to go with step of 2 from layer zero to 12
    ax.set_xticks(np.arange(0, len(curve)+1, 5))
    ax.set_xticklabels(np.arange(0, len(curve)+1, 5), fontsize=6)
    # set y tick
    ax.set_yticks(np.arange(-.11, .21, .1))
    # set y tick labels to go with step of .1 from -.1 to .2 with percision of 2
    ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-.1, .2, .1)], fontsize=6)
    fig.show()
    # save figure
    plt.savefig(os.path.join(ANALYZE_DIR, f"{modelname}_curvature_vs_model_surprisal_correlation_{dataset}.pdf"), bbox_inches='tight')
    # save as png
    plt.savefig(os.path.join(ANALYZE_DIR, f"{modelname}_curvature_vs_model_surprisal_correlation_{dataset}.png"), bbox_inches='tight')


    # ax = plt.axes((.1, .1, .45, .25))
    # for i, curv in tqdm(enumerate(curve)):
    #
    #     nan_idx = np.logical_or(np.isnan(curv), np.isnan(model_cond_p_avg))
    #     curv = curv[~nan_idx]
    #     tot_surprise_ave_ = np.array(model_cond_p_avg)[~nan_idx]
    #     # if tot_surprise_ave contains nan drop it and adijst the curv
    #
    #     r, p = sp.stats.pearsonr(tot_surprise_ave_, curv)
    #     # ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
    #     #        transform=ax.transAxes,fontsize=6)
    #     if p < 1e-2:
    #         ax.scatter(i, r, s=10, color=line_cols[i, :], zorder=4, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
    #     else:
    #         ax.scatter(i, r, s=10, color=(1, 1, 1), zorder=4, edgecolor=(0, 0, 0), linewidth=.5)
    #
    # #ax.set_ylim((-.06, .21))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)  #
    # ax.set_ylabel(r'$\rho$')
    # # set ytick fontsize to be 6
    #
    # ax.set_title(f" correlation between curvature and model conditional probability ",fontsize=8)
    # # set xtick to go with step of 2 from layer zero to 12
    # ax.set_xticklabels(np.arange(0, 13, 2), fontsize=6)
    # fig.show()
    #
