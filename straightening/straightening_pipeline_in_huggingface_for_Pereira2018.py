import os
import numpy as np
import sys
from pathlib import Path

import pandas as pd

sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
import torch
import itertools
import matplotlib
import re
import scipy as sp
from scipy import stats
import transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,AutoModelForCausalLM, AutoTokenizer,AutoModel,AutoModelForMaskedLM, AutoConfig
import xarray as xr
from minicons import scorer

from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer
from transformers import PreTrainedTokenizer
import pickle
from transformers import AutoModel
import pandas as pd


if __name__ == '__main__':
    #%%
    #modelnames='facebook/opt-125m'
    Pereira_dat=xr.load_dataarray('/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Pereira2018.nc')
    Pereira_stim=pd.read_csv('/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Pereira2018-stimulus_set.csv')

    modelclass='gpt2'
    modelname='gpt2'
    masked=False

    # get sentences from ext_obj
    sentences_=Pereira_stim.sentence.values
    sentence_id=Pereira_stim.stimulus_id.values
    sentence_passage=Pereira_stim.passage_index.values
    sentence_experiment=Pereira_stim.experiment.values
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    model = AutoModel.from_pretrained(modelname)
    model.cuda()

    # tokenize sentences
    tokenized_text = [tokenizer.tokenize(x) for x in sentences_]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
                # get activations
                # print that we are getting activations
    print('getting activations for model: {}'.format(modelname))
    all_layers=compute_model_activations(model,indexed_tokens)
    # printe that we are getting curvature
    print('getting curvature for model: {}'.format(modelname))
    curvature_dict=compute_model_curvature(all_layers)
                # empty cuda cache
    torch.cuda.empty_cache()
                # delete model
    del model
    #%%


    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    all_layer_curve=curvature_dict['curve']
    curve_ = np.stack(all_layer_curve)
    curve_change = (curve_[1:, :] - curve_[1, :])
    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', modelname)):
        line_cols = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .35*pap_ratio))
    for i,curv in enumerate(curve_):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
#    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature$')
    model_name=modelname.replace('/',')')

    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    for i, curv in enumerate(curve_change):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1,
            zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature change$')

    fig.show()

    # select atlas == language in perire dat
    (Pereira_dat.atlas=='language').values
    Pereira_dat_lang=Pereira_dat.sel(neuroid=(Pereira_dat.atlas=='language').values)
    # sort Pereira_dat_lang by values in sentence_id
    alinments=np.asarray([np.argwhere(Pereira_dat_lang.stimulus_id.values==x) for x in sentence_id]).squeeze()
    Pereira_dat_lang=Pereira_dat_lang.isel(presentation=alinments)

    # drop nans from Pereira_dat_lang
    Pereira_dat_lang=Pereira_dat_lang.dropna('neuroid')


    # find correlation between curvature value and neuroid value for each voxel
    curve_vox_corrs=[]
    for curv in curve_change:
        # compute the correaltion between curv and Pereira_dat_lang
        layer_corr=[]
        for x in tqdm(Pereira_dat_lang.values.T):
            #layer_corr.append(np.corrcoef(curv,x)[0,1])
            # do pearson correlation between curv and x
            [r,p]=stats.pearsonr(curv,x)
            #if p<.05:
            layer_corr.append(r)
            #else:
            #    layer_corr.append(np.nan)
        curve_vox_corrs.append(np.asarray(layer_corr))

    curve_vox_corrs=np.stack(curve_vox_corrs)
    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    # create 12 subplots
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    for i, curv in enumerate(curve_vox_corrs):

        ax.scatter(i, np.nanmean(curv) , s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) , yerr=np.nanstd(curv), linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_vox_corrs.shape[0]), np.nanmean(curve_vox_corrs, axis=1), color=(0, 0, 0), linewidth=1,
            zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylim((-.1, .1))
    ax.set_ylabel(f'curvature change$')

    fig.show()

    curve_vox_corrs_roi = []

    for curv in curve_change:
        # compute the correaltion between curv and Pereira_dat_lang
        layer_corr_dict = dict()
        for g, grp in Pereira_dat_lang.groupby('roi'):
            layer_corr = []
            for x in tqdm(grp.values.T):
                # layer_corr.append(np.corrcoef(curv,x)[0,1])
                #do pearson correlation between curv and x
                [r, p] = stats.pearsonr(curv, x)
                # if p<.05:
                layer_corr.append(r)
            layer_corr_dict[g]= np.asarray(layer_corr)
            # else:
            #    layer_corr.append(np.nan)
        curve_vox_corrs_roi.append(layer_corr_dict)

    #
    AA=[[x[key] for x in curve_vox_corrs_roi] for key in curve_vox_corrs_roi[0].keys()]

    # create figure wiht 12 subplots, and plot the elements in AA
    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    for i,curv in enumerate(AA):
        curv=np.stack(curv)
        ax = fig.add_subplot(4, 3, i+1)
        for j,curv_ in enumerate(curv):
            ax.scatter(j, np.nanmean(curv_) , s=25, color=line_cols[j, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
            ax.errorbar(j, np.nanmean(curv_) , yerr=np.nanstd(curv_), linewidth=0, elinewidth=1,
                    color=line_cols[j, :], zorder=0, alpha=1)
        # plot a line for the average
        ax.plot(np.arange(len(curv)), np.nanmean(curv, axis=1), color=(0, 0, 0), linewidth=1,
            zorder=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax title
        ax.set_title(f'{list(curve_vox_corrs_roi[0].keys())[i]}')
    fig.show()

    #%%
    # group sentences in each layer by their curvature change and find the the level of activation is for those sentances
    # in perereia dat lang
    for curv in curve_change:
        True
