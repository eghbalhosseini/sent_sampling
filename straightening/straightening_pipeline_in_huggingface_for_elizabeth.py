import os
import numpy as np
import sys
from pathlib import Path
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
from transformers import PreTrainedTokenizer
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import Dataset, DatasetDict
from scipy import stats
from sent_sampling.utils.curvature_utils import compute_model_activations,compute_model_curvature,normalized


if __name__ == '__main__':
    #%%
    #modelnames = 'xlnet-base-cased'
    #modelnames='bigscience/bloom-7b1'
    #modelnames='microsoft/DialoGPT-medium'
    #modelnames='funnel-transformer/small'
    #modelnames='facebook/opt-125m'
    modelname = 'gpt2'
    masked=False
    dataset='ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences = [x['text'] for x in ext_obj.data_]
    del ext_obj
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModel.from_pretrained(modelname)
    model.cuda()
    # tokenize sentences
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    model=AutoModelForCausalLM.from_pretrained(modelname)
    model.to('cuda')
    # if model continuation dict doesnt exsist create it
    all_layers=compute_model_activations(model,indexed_tokens)
    curvature_dict_true=compute_model_curvature(all_layers)
    #%%
    curve_ = curvature_dict_true['curve']
    curve_change = (curve_[1:, :] - curve_[1, :])
    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = curve_change.shape[0] + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .55, .55, .25 * pap_ratio))
    kk = 0


    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    for i, curv in enumerate(curve_):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=5, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=(np.nanstd(curv) * 180 / np.pi)/np.sqrt(curv.shape[0]), linewidth=0, elinewidth=1,
                color=line_cols[i, :], zorder=0, alpha=1)
           # plot a line for the average

    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(0, 0, 0),
            linewidth=2,
            zorder=1)
    # plot sem around the average as fill_between
    fig.show()

    ax.fill_between(np.arange(curve_.shape[0]),
                    (np.nanmean(curve_, axis=1) - np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    (np.nanmean(curve_, axis=1) + np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    color=(0, 0, 0), alpha=.2, zorder=1)

    fig.savefig(os.path.join(ANALYZE_DIR, f'{modelname}_curvature_change_vs_voxel_activation_Pereira.pdf'), transparent=True)