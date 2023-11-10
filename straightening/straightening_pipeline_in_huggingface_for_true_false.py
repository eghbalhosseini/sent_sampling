import os
import numpy as np
import sys
from pathlib import Path

import pandas as pd

import sent_sampling.utils.data_utils as data_utils
#sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj, ANALYZE_DIR
from sent_sampling.utils.extract_utils import extractor
from sent_sampling.utils.optim_utils import optim
from sent_sampling.utils.curvature_utils import compute_model_activations, compute_model_curvature
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
#from minicons import scorer

from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer



from transformers import PreTrainedTokenizer
import pickle
from transformers import AutoModel
if __name__ == '__main__':
    #%%
    #modelnames='facebook/opt-125m'
    data_animals=pd.read_csv(os.path.join(SAVE_DIR,'straightening','true_false_dataset','animals_true_false.csv'))
    data_cities=pd.read_csv(os.path.join(SAVE_DIR,'straightening','true_false_dataset','cities_true_false.csv'))
    data_companies=pd.read_csv(os.path.join(SAVE_DIR,'straightening','true_false_dataset','companies_true_false.csv'))
    data_elements=pd.read_csv(os.path.join(SAVE_DIR,'straightening','true_false_dataset','elements_true_false.csv'))
    data_facts=pd.read_csv(os.path.join(SAVE_DIR,'straightening','true_false_dataset','facts_true_false.csv'))
    data_generated=pd.read_csv(os.path.join(SAVE_DIR,'straightening','true_false_dataset','generated_true_false.csv'))
    data_invensions=pd.read_csv(os.path.join(SAVE_DIR,'straightening','true_false_dataset','inventions_true_false.csv'))
    # combine them into one data
    data=pd.concat([data_animals,data_cities,data_companies,data_elements,data_facts,data_generated,data_invensions])

    modelclass='gpt2-xl'
    modelname='gpt2-xl'
    masked=False
    # get statements with label 0
    false_statements=data[data['label']==0]['statement'].values
    # get statements with label 1
    true_statements=data[data['label']==1]['statement'].values
    # drop the period at the end of each statement
    false_statements=[x[:-1] for x in false_statements]
    true_statements=[x[:-1] for x in true_statements]
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    model = AutoModel.from_pretrained(modelname)
    model.cuda()

    # tokenize sentences
    tokenized_false = [tokenizer.tokenize(x) for x in false_statements]
    tokenized_true = [tokenizer.tokenize(x) for x in true_statements]
    # get ids
    indexed_tokens_false = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_false]
    indexed_tokens_true = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_true]

    print('getting activations for model: {}'.format(modelname))
    all_layers=compute_model_activations(model,indexed_tokens_false)
    curvature_dict_false = compute_model_curvature(all_layers)
    # do for true
    print('getting activations for model: {}'.format(modelname))
    all_layers=compute_model_activations(model,indexed_tokens_true)
    curvature_dict_true = compute_model_curvature(all_layers)
    torch.cuda.empty_cache()
                # delete model
    del model

    #%%
    fig = plt.figure(figsize=(5.5,9), dpi=200, frameon=False)
    pap_ratio=5.5/9
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .55, .25 * pap_ratio))
    kk=0
    curve_ = curvature_dict_false['curve']
    curve_change = (curve_[1:, :] - curve_[1, :])

    ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=5, color=line_cols[kk,:], zorder=2, edgecolor=(0, 0, 0),linewidth=.5, alpha=1)
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=line_cols[kk,:], linewidth=1,zorder=1,label='false')
    ax.fill_between(np.arange(curve_change.shape[0]),np.nanmean(curve_change, axis=1)* 180 / np.pi - ((np.nanstd(curve_change, axis=1))*180 / np.pi)/np.sqrt(curve_change.shape[1]),
                        np.nanmean(curve_change, axis=1)* 180 / np.pi + ((np.nanstd(curve_change, axis=1))*180 / np.pi)/np.sqrt(curve_change.shape[1]),color=line_cols[kk, :], alpha=.2, zorder=1)
    # do the same thing for true
    kk=1
    curve_ = curvature_dict_true['curve']
    curve_change = (curve_[1:, :] - curve_[1, :])
    ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=5,
               color=line_cols[kk, :], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=line_cols[kk, :],
            linewidth=1, zorder=1, label='true')
    ax.fill_between(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi - (
                (np.nanstd(curve_change, axis=1)) * 180 / np.pi) / np.sqrt(curve_change.shape[1]),
                    np.nanmean(curve_change, axis=1) * 180 / np.pi + (
                                (np.nanstd(curve_change, axis=1)) * 180 / np.pi) / np.sqrt(curve_change.shape[1]),
                    color=line_cols[kk, :], alpha=.2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change$')


    ax = plt.axes((.1, .45, .45, .3 * pap_ratio))
    kk = 0
    curve_ = curvature_dict_false['curve']


    ax.scatter(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, s=5,
               color=line_cols[kk, :], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=line_cols[kk, :],
            linewidth=1, zorder=1, label='false')
    ax.fill_between(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi - (
                (np.nanstd(curve_, axis=1)) * 180 / np.pi) / np.sqrt(curve_.shape[1]),
                    np.nanmean(curve_, axis=1) * 180 / np.pi + (
                                (np.nanstd(curve_, axis=1)) * 180 / np.pi) / np.sqrt(curve_.shape[1]),
                    color=line_cols[kk, :], alpha=.2, zorder=1)
    # do the same thing for true
    kk = 1
    curve_ = curvature_dict_true['curve']
    curve_change = (curve_[1:, :] - curve_[1, :])
    ax.scatter(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, s=5,
               color=line_cols[kk, :], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=line_cols[kk, :],
            linewidth=1, zorder=1, label='true')
    ax.fill_between(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi - (
            (np.nanstd(curve_, axis=1)) * 180 / np.pi) / np.sqrt(curve_.shape[1]),
                    np.nanmean(curve_, axis=1) * 180 / np.pi + (
                            (np.nanstd(curve_, axis=1)) * 180 / np.pi) / np.sqrt(curve_.shape[1]),
                    color=line_cols[kk, :], alpha=.2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature$')
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    fig.show()