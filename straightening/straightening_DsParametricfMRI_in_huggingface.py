import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch
import itertools
import matplotlib
import re
import scipy as sp
import transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,AutoModelForCausalLM, AutoTokenizer,AutoModel,AutoModelForMaskedLM, AutoConfig
import xarray as xr
from minicons import scorer

from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer

from transformers import PreTrainedTokenizer
import pickle
import pandas as pd


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def compute_model_activations(model,indexed_tokens):
    # get activations
    all_layers = []
    for i in tqdm(range(len(indexed_tokens))):
        tokens_tensor = torch.tensor([indexed_tokens[i]]).to('cuda')
        with torch.no_grad():
            outputs = model(tokens_tensor, output_hidden_states=True, output_attentions=False)
            hidden_states = outputs['hidden_states']
            # squeeze the first dimension
            hidden_states = [x.squeeze(0).cpu() for x in hidden_states]
        all_layers.append(hidden_states)
    torch.cuda.empty_cache()
    return all_layers

def compute_model_curvature(all_layers):
    all_layer_curve = []
    all_layer_curve_all = []
    all_layer_curve_rnd = []
    all_layer_curve_rnd_all = []
    for idk, layer_act in tqdm(enumerate(all_layers)):
        sent_act = [torch.diff(x, axis=0).cpu() for x in layer_act]
        sent_act = [normalized(x) for x in sent_act]
        curvature = []
        for idy, vec in (enumerate(sent_act)):
            curve = [np.dot(vec[idx, :], vec[idx + 1, :]) for idx in range(vec.shape[0] - 1)]
            curvature.append(np.arccos(curve))
        all_layer_curve.append([np.mean(x) for x in curvature])
        all_layer_curve_all.append(curvature)

    curve_ = np.stack(all_layer_curve).transpose()
    curve_change = (curve_[0:, :] - curve_[0, :])
    # make a dictionary with fieldds 'curve','curve_change','all_layer_curve_all' and return the dictionary
    return dict(curve=curve_,curve_change=curve_change,all_layer_curve_all=all_layer_curve_all)

if __name__ == '__main__':
    #%%

    #modelnames='facebook/opt-125m'

    #modelclass='facebook/opt-125m'
    modelclass = 'gpt2-xl'
    basemodel=modelclass
    colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255)]
    ds_random_sentence = pd.read_csv(os.path.join(ANALYZE_DIR, 'ds_parametric',
                                                  'sent,G=best_performing_pereira_1-D=ud_sentencez_ds_random_100_edited_selected_textNoPeriod_final.csv'))
    ds_min_sentence = pd.read_csv(os.path.join(ANALYZE_DIR, 'ds_parametric',
                                               'sent,G=best_performing_pereira_1-D=ud_sentencez_ds_min_100_edited_selected_textNoPeriod_final.csv'))
    ds_max_sentence = pd.read_csv(os.path.join(ANALYZE_DIR, 'ds_parametric',
                                               'sent,G=best_performing_pereira_1-D=ud_sentencez_ds_max_100_edited_selected_textNoPeriod_final.csv'))

    model = AutoModel.from_pretrained(basemodel)
    model.cuda()
    model.eval()
    cuvatures_dict={}
    for ds in [ds_min_sentence,ds_random_sentence,ds_max_sentence]:
    # get sentences from ext_obj
        sentences=list(ds[ds.keys()[1]])
        tokenizer = AutoTokenizer.from_pretrained(basemodel)
    # tokenize sentences
        tokenized_text = [tokenizer.tokenize(x) for x in sentences]
        # get ids
        indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
        print('getting activations for model: {}'.format(basemodel))
        all_layers=compute_model_activations(model,indexed_tokens)
        print('getting curvature for model: {}'.format(basemodel))
        curvature_dict=compute_model_curvature(all_layers)
        torch.cuda.empty_cache()
        cuvatures_dict[ds.keys()[1]]=curvature_dict


    #%%
    fig = plt.figure(figsize=(5.5,9), dpi=200, frameon=False)
    pap_ratio=5.5/9
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = len(cuvatures_dict) + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .75, .45 * pap_ratio))
    kk=0

    for key,val in cuvatures_dict.items():
        curve_ = val['curve']
        curve_change = (curve_[1:, :] - curve_[1, :])
        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=5, color=colors[kk], zorder=2, edgecolor=(0, 0, 0),linewidth=.5, alpha=1)
        # plot plot errorbars as fillbetween
        ax.fill_between(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi - np.nanstd(curve_change) * 180 / np.pi,
                        np.nanmean(curve_change, axis=1) * 180 / np.pi + np.nanstd(curve_change) * 180 / np.pi, color=colors[kk], alpha=.2, zorder=1)

        #ax.errorbar(np.arange(curve_change.shape[0])+kk*.25, np.nanmean(curve_change, axis=1) * 180 / np.pi,
        #            yerr=np.nanstd(curve_change) * 180 / np.pi, color=colors[kk],linewidth=.5, zorder=1, alpha=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=colors[kk], linewidth=1,
            zorder=1,label=key)
        kk+=1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change')
    # set xlim to [-1,49]
    ax.set_xlim([-1,len(curve_change)])
    #ax.set_ylim([-20,5])
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)


    ax = plt.axes((.1, .55, .75, .45 * pap_ratio))
    kk = 0
    for key, val in cuvatures_dict.items():
        curve_ = val['curve']

        ax.scatter(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, s=5,
                   color=colors[kk], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        # plot plot errorbars as fillbetween
        ax.fill_between(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi - np.nanstd(curve_) * 180 / np.pi,
                        np.nanmean(curve_, axis=1) * 180 / np.pi + np.nanstd(curve_) * 180 / np.pi, color=colors[kk],
                        alpha=.2, zorder=1)
        ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=colors[kk],
                linewidth=1,
                zorder=1, label=key)
        kk += 1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature')
    # set xlim to [-1,49]
    ax.set_xlim([-1, len(curve_)-1])
    #ax.set_ylim([-12, 2])


    # ax = plt.axes((.1, .55, .55, .25 * pap_ratio))
    # kk = 0
    # for key, val in cuvatures_dict.items():
    #     curve_ = val['curve']
    #     curva_change = (curve_[1:, :] - curve_[1, :])
    #     # plot individual curves
    #     for i in range(curve_.shape[1]):
    #         ax.plot(np.arange(curva_change.shape[0]), curva_change[:, i] * 180 / np.pi, color=colors[kk], linewidth=.5,
    #                 zorder=1, alpha=.2)
    #     ax.scatter(np.arange(curva_change.shape[0]), np.nanmean(curva_change, axis=1) * 180 / np.pi, s=5,
    #                color=colors[kk], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
    #     ax.plot(np.arange(curva_change.shape[0]), np.nanmean(curva_change, axis=1) * 180 / np.pi, color=colors[kk],
    #             linewidth=1,
    #             zorder=1, label=key)
    #     kk += 1
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)  #
    # ax.set_ylabel(f'curvature change$')
    # # set xlim to [-1,49]
    # ax.set_xlim([-1, 49])
    # #ax.set_ylim([-12, 2])


    fig.show()
    # plot the best layer
    #fig.show()
    basemodel_ = basemodel.replace('/', '_')
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_change_{basemodel_}_models_dsParametric.eps'),dpi=200,format='eps',bbox_inches='tight',transparent=True)
    # save as png
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_change_{basemodel_}_models_dsParametric.png'),dpi=200,format='png',bbox_inches='tight',transparent=True)
