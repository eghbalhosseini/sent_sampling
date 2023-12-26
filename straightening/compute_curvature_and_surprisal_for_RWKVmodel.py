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
from matplotlib.pyplot import GridSpec
import pandas as pd
from pathlib import Path
import torch
from sent_sampling.utils.curvature_utils import compute_model_activations, compute_model_curvature
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer,GPTNeoXForCausalLM, AutoModelForSequenceClassification,AutoModelForCausalLM,AutoModelForTokenClassification
from transformers import AutoTokenizer, RwkvConfig, RwkvModel
import scipy as sp
import matplotlib
from minicons import scorer

if __name__ == '__main__':
    dataset = 'ud_sentencez_token_filter_v3_wordFORM'
    modelclass = 'gpt2'
    modelname= 'gpt2'
    newmodelname="RWKV/rwkv-4-169m-pile"
    masked = False
    dataset = 'ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # emcpty cuda cache
    torch.cuda.empty_cache()
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences_all = [x['text'] for x in ext_obj.data_]
    sentences_words = [x['word_FORM'] for x in ext_obj.data_]
    good_sent_id = np.where(np.asarray([len(x['word_FORM']) == len(x['surprisal_3']) for x in ext_obj.data_]))[0]
    sentences_ = [sentences_all[i] for i in good_sent_id]
    # sample 1000 sentences
    n_sent=2000
    sentences = np.random.choice(sentences_, n_sent, replace=False)


    new_tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
    new_model = RwkvModel.from_pretrained("RWKV/rwkv-4-169m-pile")
    # make an untrained verison of the model
    new_model_untrained_config=RwkvConfig.from_pretrained(newmodelname)
    new_model_untrained=RwkvModel(new_model_untrained_config)

    # define an ablated model

    # Ablate MLP layers in the first transformer layer


    new_model.cuda()
    tokenized_new = [new_tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens_new=[new_tokenizer.convert_tokens_to_ids(x) for x in tokenized_new]
    all_layers=compute_model_activations(new_model,indexed_tokens_new)
    curvature_dict_new = compute_model_curvature(all_layers)
    # compute the curvature for untrained model
    new_model_untrained.cuda()
    all_layers=compute_model_activations(new_model_untrained,indexed_tokens_new)
    curvature_dict_new_untrained = compute_model_curvature(all_layers)


    mincons_model = scorer.IncrementalLMScorer(newmodelname, device='cuda')

    model_cond_p=[]
    model_surp=[]
    for idx, sentence in tqdm(enumerate(sentences),total=len(sentences)):
        True
        cond_p = mincons_model.compute_stats(mincons_model.prepare_text(sentence))
        modl_surp = mincons_model.token_score(sentence, surprisal=True)
        model_cond_p.append(cond_p)
        model_surp.append(modl_surp)

    model_surp_avg = np.asarray([np.mean([y[1] for y in x[0]][2:]) for x in model_surp])

    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    curve_ = np.stack(curvature_dict_new_untrained['curve'], axis=0)
    curve_change = (curve_[1:, :] - curve_[1, :])

    curve_new = np.stack(curvature_dict_new['curve'], axis=0)
    curve_change_new = (curve_new[1:, :] - curve_new[1, :])

    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    # if bool(re.findall(r'-untrained', modelname)):
    line_cols_unt = line_cols * 0 + (.6)
    # if bool(re.findall(r'-untrained', modelname)):
    line_cols_unt = line_cols * 0 + (.6)
    ax = plt.axes((.05, .1, .45, .25 * pap_ratio))

    # plot a line for the average
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(.5, .5, .5), linewidth=1,
            zorder=2, marker='o', markersize=4, label=modelname)
    # add fill between
    ax.fill_between(np.arange(curve_.shape[0]),
                    np.nanmean(curve_, axis=1) * 180 / np.pi - np.nanstd(curve_, axis=1) * 180 / np.pi/np.sqrt(curve_.shape[1]),
                    np.nanmean(curve_, axis=1) * 180 / np.pi + np.nanstd(curve_, axis=1) * 180 / np.pi/np.sqrt(curve_.shape[1]), alpha=1,
                    color=(.5, .5, .5))
    ax.plot(np.arange(curve_new.shape[0]), np.nanmean(curve_new, axis=1) * 180 / np.pi, color=(1, 0, 0), linewidth=1,
            marker='o', markersize=4, zorder=2, label="RWKV/rwkv-4-169m-pile")
    # add fill between
    ax.fill_between(np.arange(curve_new.shape[0]),
                    np.nanmean(curve_new, axis=1) * 180 / np.pi - np.nanstd(curve_new, axis=1) * 180 / np.pi/np.sqrt(curve_.shape[1]),
                    np.nanmean(curve_new, axis=1) * 180 / np.pi + np.nanstd(curve_new, axis=1) * 180 / np.pi/np.sqrt(curve_.shape[1]), alpha=1,
                    color=(1, 0, 0))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    #    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature')
    ax.set_xlabel('layer')

    ax = plt.axes((.6, .1, .35, .25* pap_ratio))
    for i, curv in tqdm(enumerate(curve_new)):
        # find if curv or tot_surprise_ave has nan values
        # if so, remove them from both
        nan_idx = np.logical_or(np.isnan(curv), np.isnan(model_surp_avg))
        curv = curv[~nan_idx]
        curv_unt=curve_[i, :][~nan_idx]
        tot_surprise_ave_ = np.array(model_surp_avg)[~nan_idx]

        # if tot_surprise_ave contains nan drop it and adijst the curv

        r, p = sp.stats.pearsonr(tot_surprise_ave_, curv)
        r_unt, p_unt = sp.stats.pearsonr(tot_surprise_ave_, curv_unt)
        # ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        #        transform=ax.transAxes,fontsize=6)
        if p < 1e-2:
            ax.scatter(i, r, s=10, color='red', zorder=4, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        else:
            ax.scatter(i, r, s=10, color=(1, 1, 1), zorder=4, edgecolor=(0, 0, 0), linewidth=.5)

        if p_unt < 1e-2:
            ax.scatter(i, r_unt, s=10, color=(.5, .5, .5), zorder=4, edgecolor=(.5, .5, .5), linewidth=.5, alpha=1)
        else:
            ax.scatter(i, r_unt, s=10, color=(1, 1, 1), zorder=4, edgecolor=(.5, .5, .5), linewidth=.5)

    # ax.set_ylim((-.06, .21))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelname} \n correlation between curvature and model", fontsize=8)
    # set xtick to go with step of 2 from layer zero to 12
    ax.set_xticks(np.arange(0, len(curve_new) + 1, 2))
    ax.set_xticklabels(np.arange(0, len(curve_new) + 1, 2), fontsize=6)
    # set y tick
    ax.set_yticks(np.arange(-.11, .21, .05))
    # set y tick labels to go with step of .1 from -.1 to .2 with percision of 2
    ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-.1, .2, .05)], fontsize=6)


    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_and_surpsial_RWKA.pdf'), transparent=True)
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_and_surprisal_RWKA.png'), transparent=True)
