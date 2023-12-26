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
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer,GPTNeoXForCausalLM
import scipy as sp
import matplotlib
from datasets import load_dataset, load_from_disk
import datasets
import string
from minicons import scorer

import re
if __name__ == '__main__':
    modelclass = 'gpt2'

    modelname= 'gpt2-xl'
    masked = False
    max_upper_coun_th = 5
    min_sent_len=10
    max_sent_len=50
    # mini_dataset = load_from_disk(
    #     os.path.join('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext/',
    #                  'train'))
    sample_dataset = load_from_disk(
            os.path.join('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext_sample/','train'))

    tokenizer = AutoTokenizer.from_pretrained(modelclass)
    # sample 1000 sentences
    sentences_split=[]
    for id, sent in tqdm(enumerate(sample_dataset['text'])):
        sentence_split = re.split(r' *[\.\?!][\'"\)\]]* *', sent.replace('\n', ' '))
        sentences_split.extend(sentence_split)
    sentences_length=[len(x) for x in sentences_split]
    selected = np.argwhere(np.logical_and(np.array(sentences_length) > 20, np.array(sentences_length) < 50))
    sentences_split_ow=[sentences_split[int(i)] for i in np.random.permutation(selected)[:4000]]

    tokenized_text_ow = [tokenizer.tokenize(x) for x in sentences_split_ow]
    indexed_tokens_ow = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text_ow]


    mincons_model = scorer.IncrementalLMScorer(modelname, device='cuda')

    model_cond_p=[]
    model_surp=[]
    for idx, sentence in tqdm(enumerate(sentences_split_ow),total=len(sentences_split_ow)):
        True
        cond_p = mincons_model.compute_stats(mincons_model.prepare_text(sentence))
        modl_surp = mincons_model.token_score(sentence, surprisal=True)
        model_cond_p.append(cond_p)
        model_surp.append(modl_surp)


    model = AutoModel.from_pretrained(modelname)
    # drop (ln_f): LayerNorm((1600,), eps=1e-05, elementwise_affine=True) of the model
    #model.ln_f = torch.nn.Identity()
    model.cuda()
    # move index_tokens to cuda

    all_layers=compute_model_activations(model,indexed_tokens_ow)
    curvature_dict_ow=compute_model_curvature(all_layers)


    # compute the model activation for the untrained model

    model_surp_avg = np.asarray([np.mean([y[1] for y in x[0]][2:]) for x in model_surp])
    model_cond_p_avg = np.asarray([np.mean([y for y in x[0]][2:]) for x in model_cond_p])
    fig = plt.figure(figsize=(4, 3), frameon=False)
    curve = np.stack(curvature_dict_ow['curve'])
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

    # ax.set_ylim((-.06, .21))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelname} \n correlation between curvature and model", fontsize=8)
    # set xtick to go with step of 2 from layer zero to 12
    ax.set_xticks(np.arange(0, len(curve) + 1, 5))
    ax.set_xticklabels(np.arange(0, len(curve) + 1, 5), fontsize=6)
    # set y tick
    ax.set_yticks(np.arange(-.11, .21, .1))
    # set y tick labels to go with step of .1 from -.1 to .2 with percision of 2
    ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-.1, .2, .1)], fontsize=6)
    fig.show()