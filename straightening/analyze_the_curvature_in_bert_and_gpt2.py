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
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer,AutoModelForCausalLM, AutoTokenizer,AutoModel,AutoModelForMaskedLM, AutoConfig
import scipy as sp
import matplotlib

if __name__ == '__main__':
    dataset = 'ud_sentencez_token_filter_v3_wordFORM'
    modelclass = 'gpt2'
    modelname= 'gpt2-medium'
    masked = False
    dataset = 'ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # emcpty cuda cache
    torch.cuda.empty_cache()
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences = [x['text'] for x in ext_obj.data_]
    sentences_words = [x['word_FORM'] for x in ext_obj.data_]
    good_sent_id = np.where(np.asarray([len(x['word_FORM']) == len(x['surprisal_3']) for x in ext_obj.data_]))[0]
    sentences_ = [sentences[i] for i in good_sent_id]
    # random select 1000 sentences
    np.random.seed(0)
    n_count=8408
    sentences = np.random.choice(sentences_, n_count, replace=False)


    model = AutoModelForCausalLM.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelclass)
    bert_tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model=AutoModelForCausalLM.from_pretrained('bert-large-uncased')
    #bert_tokenizer=AutoTokenizer.from_pretrained("roberta-large")
    #bert_model=AutoModelForCausalLM.from_pretrained("roberta-large")

    # define an ablated model

    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    tokenized_text_bert = [bert_tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    indexed_tokens_bert = [bert_tokenizer.convert_tokens_to_ids(x) for x in tokenized_text_bert]

    # move model to cuda
    model.to('cuda')
    all_layers = compute_model_activations(model, indexed_tokens)
    curvature_dict_gpt2 = compute_model_curvature(all_layers)
    # compute curvature for bert
    bert_model.to('cuda')
    all_layers = compute_model_activations(bert_model, indexed_tokens_bert)
    curvature_dict_bert = compute_model_curvature(all_layers)






    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    curve_ = np.stack(curvature_dict_gpt2['curve'], axis=0)
    curve_change = (curve_[1:, :] - curve_[1, :])

    curve_bert= np.stack(curvature_dict_bert['curve'], axis=0)
    curve_change_bert = (curve_bert[1:, :] - curve_bert[1, :])

    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    ax = plt.axes((.1, .55, .55, .25 * pap_ratio))
    kk = 0
    # plot the average for each of the curves
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color='black', linewidth=1,
            label='true', marker='o', markersize=1)
    ax.fill_between(np.arange(curve_.shape[0]),np.nanmean(curve_, axis=1) * 180 / np.pi - np.nanstd(curve_, axis=1) * 180 /np.sqrt(n_count),
                    np.nanmean(curve_, axis=1) * 180 / np.pi + np.nanstd(curve_, axis=1) * 180/np.sqrt(n_count) ,alpha=.2, color='black')
    # plot bert
    ax.plot(np.arange(curve_bert.shape[0]), np.nanmean(curve_bert, axis=1) * 180 / np.pi, color='red', linewidth=1,
            label='greedy', marker='o', markersize=1)
    ax.fill_between(np.arange(curve_bert.shape[0]), np.nanmean(curve_bert, axis=1) * 180 / np.pi - np.nanstd(curve_bert,axis=1) * 180 / np.pi/np.sqrt(n_count),np.nanmean(curve_bert, axis=1) * 180 / np.pi + np.nanstd(curve_bert,axis=1) * 180 / np.pi/np.sqrt(n_count),alpha=.2, color='red')

    # set the axis labels
    ax.set_xlabel('layer')
    ax.set_ylabel('angle (degrees)')
    # set xlim to -1 to length of curve
    ax.set_xlim(-1, curve_.shape[0])
    # turn of spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = plt.axes((.1, .15, .55, .25 * pap_ratio))
    kk = 0
    # plot the average for each of the curves
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color='black', linewidth=1,
            label='true', marker='o', markersize=1)
    ax.fill_between(np.arange(curve_change.shape[0]),
                    np.nanmean(curve_change, axis=1) * 180 / np.pi - np.nanstd(curve_change, axis=1) * 180/np.sqrt(n_count),
                    np.nanmean(curve_change, axis=1) * 180 / np.pi + np.nanstd(curve_change, axis=1) * 180/np.sqrt(n_count), alpha=.2, color='black')
    # plot bert
    ax.plot(np.arange(curve_change_bert.shape[0]), np.nanmean(curve_change_bert, axis=1) * 180 / np.pi, color='red', linewidth=1,
            label='greedy', marker='o', markersize=1)
    ax.fill_between(np.arange(curve_change_bert.shape[0]),
                    np.nanmean(curve_change_bert, axis=1) * 180 / np.pi - np.nanstd(curve_change_bert, axis=1) * 180 / np.pi/np.sqrt(n_count),
                    np.nanmean(curve_change_bert, axis=1) * 180 / np.pi + np.nanstd(curve_change_bert, axis=1) * 180 / np.pi/np.sqrt(n_count),
                    alpha=.2, color='red')

    # set the axis labels
    ax.set_xlabel('layer')
    ax.set_ylabel('angle change')
    # set xlim to -1 to length of curve
    ax.set_xlim(-1, curve_.shape[0])
    # turn of spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_{dataset}_bert_gpt2_medium.pdf'), transparent=True)
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_{dataset}_Attn_gpt2_medium.png'), transparent=True)

