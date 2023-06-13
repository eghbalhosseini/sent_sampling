import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
desired_path = '/om/weka/evlab/ehoseini/JointMDS'
sys.path.extend([desired_path,desired_path])
sys.path.append(desired_path)
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from utils import extract_pool
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
    curve_change = (curve_[1:, :] - curve_[1, :])
    # make a dictionary with fieldds 'curve','curve_change','all_layer_curve_all' and return the dictionary
    return dict(curve=curve_,curve_change=curve_change,all_layer_curve_all=all_layer_curve_all)

if __name__ == '__main__':
    #%%
    #modelnames = 'xlnet-base-cased'
    #modelnames='bigscience/bloom-7b1'
    #modelnames='microsoft/DialoGPT-medium'
    #modelnames='funnel-transformer/small'
    #modelnames='facebook/opt-125m'
    basemodel = 'gpt2-xl'
    masked=False
    dataset='ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences=[x['text'] for x in ext_obj.data_]
    sentences_words=[x['word_FORM'] for x in ext_obj.data_]
    good_sent_id=np.where(np.asarray([len(x['word_FORM'])==len(x['surprisal_3']) for x in ext_obj.data_]))[0]
    sentences_=[sentences[i] for i in good_sent_id]
    # go through the sentences and find sentences that are more than 6 words long and only take the frist 4 words from them and put them in sentence_piece list
    sentence_piece=[]
    sentence_full=[]
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    sentence_tokens=[]
    sent_token_all=[]
    for sent in sentences_:
        sent_tok_id=tokenizer(sent,return_tensors='pt')['input_ids'][0]
        sent_token_all.append(tokenizer(sent,return_tensors='pt'))
        sentence_tokens.append(sent_tok_id.tolist())
    continuation_k=7
    context_k=3
    # find sentence tokens that are longer than context_k+continuation_k
    long_sent_id=np.where(np.asarray([len(x)>context_k+continuation_k for x in sentence_tokens]))[0]
    # filter sentence on long_sent_id
    sentences_=[sentences_[i] for i in long_sent_id]
    sent_token_=[sent_token_all[i] for i in long_sent_id]

    # make a true continuation based on continuation_k+context_k
    true_continuation=[x[:context_k+continuation_k] for x in sentence_tokens]
    true_continuation=[true_continuation[i] for i in long_sent_id]
    # compute greed continuation

    model=AutoModelForCausalLM.from_pretrained(basemodel)
    model.to('cuda')
    # if model continuation dict doesnt exsist create it
    path_continuation_dict = Path(ANALYZE_DIR, f'{basemodel}_continuation_dict_cntx_{context_k}_cont_{continuation_k}.pkl')
    if not path_continuation_dict.exists():
    # go to evey element in sent_token_ and and for each key get the first context_k tokens and add the to greedy_inputs
        greedy_inputs=[dict(input_ids=x['input_ids'][:1,:context_k],attention_mask=x['attention_mask'][:1,:context_k]) for x in sent_token_]
        greedy_tok_id=[]
        beam_tok_id=[]
        sample_tok_id=[]
        for i in tqdm(range(len(greedy_inputs))):
            tokens_tensor = greedy_inputs[i]
            # move values of torch tensor to cuda
            for key in tokens_tensor.keys():
                tokens_tensor[key]=tokens_tensor[key].to('cuda')
            with torch.no_grad():
                output_greedy = model.generate(**tokens_tensor, max_new_tokens=continuation_k, return_dict_in_generate=False, output_scores=False)
                #output_beam = model.generate(**tokens_tensor,num_beams=4, max_new_tokens=continuation_k,
                #                               return_dict_in_generate=False, output_scores=False)
                #output_sample = model.generate(**tokens_tensor, max_new_tokens=continuation_k, return_dict_in_generate=False,do_sample=True, output_scores=False)
            #greedy_continuation.append(tokenizer.decode(outputs['sequences'][0]))
            greedy_tok_id.append(output_greedy[0].tolist())
            #beam_tok_id.append(output_beam[0].tolist())
            #sample_tok_id.append(output_sample[0].tolist())

        # convert greedy_tok_id to list

        # make a dictionary for both greedy and true continuation
        continuations_dict=dict(greedy=greedy_tok_id,true=true_continuation,beam=beam_tok_id,sample=sample_tok_id)
        # define a path for continuation_dict and basemodel and save it

        with open(path_continuation_dict.__str__(),'wb') as f:
            pickle.dump(continuations_dict,f)
    else:
        with open(path_continuation_dict.__str__(),'rb') as f:
            continuations_dict=pickle.load(f)

    all_layers_true=compute_model_activations(model,continuations_dict['true'])
    curvature_dict_true=compute_model_curvature(all_layers_true)

    # create a dicitonary of greed_continuation activations and true continuation

    tokenizer.convert_ids_to_tokens(continuations_dict['greedy'][3358])
    tokenizer.convert_ids_to_tokens(continuations_dict['true'][3358])

    all_layers_greedy = compute_model_activations(model, continuations_dict['greedy'])
    curvature_dict_greedy = compute_model_curvature(all_layers_greedy)

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


    ax.fill_between(np.arange(curve_.shape[0]),
                    (np.nanmean(curve_, axis=1) - np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    (np.nanmean(curve_, axis=1) + np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    color=(0, 0, 0), alpha=.2, zorder=1)

    curve_ = curvature_dict_greedy['curve']
    for i, curv in enumerate(curve_):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=5, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi/np.sqrt(curv.shape[0]), linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(1, .2, .2),
            linewidth=2,
            zorder=1)
    ax.fill_between(np.arange(curve_.shape[0]),
                    (np.nanmean(curve_, axis=1) - np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    (np.nanmean(curve_, axis=1) + np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    color=(1, .2, .2), alpha=.2, zorder=1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    #    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature ')
    ax.set_xlim((-1, 50))
    ax = plt.axes((.1, .1, .55, .25 * pap_ratio))
    curve_ = curvature_dict_true['curve']
    curve_change = (curve_[1:, :] - curve_[1, :])

    num_colors = curve_change.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    for i, curv in enumerate(curve_change):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=5, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
               linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi/np.sqrt(curv.shape[0]), linewidth=0, elinewidth=1,
                color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average

    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=2,
        zorder=1)
    ax.fill_between(np.arange(curve_change.shape[0]),
                    (np.nanmean(curve_change, axis=1) - np.nanstd(curve_change, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    (np.nanmean(curve_change, axis=1) + np.nanstd(curve_change, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    color=(0,0,1), alpha=.2, zorder=1)

    curve_ = curvature_dict_greedy['curve']
    curve_change = (curve_[1:, :] - curve_[1, :])
    for i, curv in enumerate(curve_change):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=5, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=(np.nanstd(curv) * 180 / np.pi) / np.sqrt(curv.shape[0]),
                    linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(1, .2, .2),
            linewidth=2,
            zorder=1)
    ax.fill_between(np.arange(curve_change.shape[0]),
                    (np.nanmean(curve_change, axis=1) - np.nanstd(curve_change, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    (np.nanmean(curve_change, axis=1) + np.nanstd(curve_change, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    color=(1, .2, .2), alpha=.2, zorder=1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_xlim((-1,49 ))
    #    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature change$')
    fig.show()

    fig.savefig(os.path.join(ANALYZE_DIR,
                             f'curvature_{basemodel}_{dataset}_true_vs_greedy_cntx_{context_k}_cont_{continuation_k}_sem.pdf'),
                transparent=True)

    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9

    curve_greedy=curvature_dict_greedy['curve']
    curve_change_greedy = (curve_greedy[1:, :] - curve_greedy[1, :])
    curve_true=curvature_dict_true['curve']
    curve_change_true = (curve_true[1:, :] - curve_true[1, :])

    # plot the difference in curve
    curvature_difference=curve_greedy-curve_true
    curvature_change_difference=curve_change_greedy-curve_change_true
    ax = plt.axes((.65, .1, .3, .3 * pap_ratio))
    ax.plot(np.arange(curvature_change_difference.shape[0]), np.nanmean(curvature_change_difference, axis=1) * 180 / np.pi, color=(1, .2, .2),
            linewidth=2,
            zorder=1)
    ax.fill_between(np.arange(curvature_change_difference.shape[0]),
                    (np.nanmean(curvature_change_difference, axis=1) - np.nanstd(curvature_change_difference, axis=1)) * 180 / np.pi,
                    (np.nanmean(curvature_change_difference, axis=1) + np.nanstd(curvature_change_difference, axis=1)) * 180 / np.pi,
                    color=(1, .2, .2), alpha=.2, zorder=1)

    ax = plt.axes((.65, .55, .3, .3 * pap_ratio))
    ax.plot(np.arange(curvature_difference.shape[0]),
            np.nanmean(curvature_difference, axis=1) * 180 / np.pi, color=(1, .2, .2),
            linewidth=2,
            zorder=1)
    ax.fill_between(np.arange(curvature_difference.shape[0]),
                    (np.nanmean(curvature_difference, axis=1) - np.nanstd(curvature_difference,
                                                                                 axis=1)) * 180 / np.pi,
                    (np.nanmean(curvature_difference, axis=1) + np.nanstd(curvature_difference,
                                                                                 axis=1)) * 180 / np.pi,
                    color=(1, .2, .2), alpha=.2, zorder=1)
    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_difference_{basemodel}_{dataset}_true_vs_greedy_cntx_{context_k}_cont_{continuation_k}.pdf'), transparent=True)
    # do a ttest to see if each column of curvature_difference is significantly different from 0
    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    ttest_results = np.zeros((curvature_difference.shape[0]))
    for i in range(curvature_difference.shape[0]):
        ttest_results[i] = sp.stats.ttest_1samp(curvature_difference[i, :], 0)[1]
    # plot the pvalues
    ax = plt.axes((.65, .55, .3, .3 * pap_ratio))
    ax.plot(np.arange(curvature_difference.shape[0]), -np.log10(ttest_results), color=(1, .2, .2),
            linewidth=2,
            zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_xlim((-1, 49))
    #    ax.set_ylim((-15, 5))
    fig.show()
    #%% mistral models
    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    ttest_results = np.zeros((curvature_change_difference.shape[0]))
    ttest_stat=np.zeros((curvature_change_difference.shape[0]))
    for i in range(curvature_change_difference.shape[0]):
        ttest_results[i] = sp.stats.ttest_1samp(curvature_change_difference[i, :], 0)[1]
        ttest_stat[i] = sp.stats.ttest_1samp(curvature_change_difference[i, :], 0)[0]
    # plot the pvalues
    ax = plt.axes((.65, .55, .3, .3 * pap_ratio))
    ax.plot(np.arange(curvature_change_difference.shape[0]), -np.log10(ttest_results), color=(1, .2, .2),
            linewidth=2,
            zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_xlim((-1, 49))
    #    ax.set_ylim((-15, 5))
    fig.show()

    #%%
    # check if curvature_change_greedy is significantly less than curvature_change_true for each column with signifiance level 0.01
    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    ttest_results = np.zeros((curvature_change_difference.shape[0]))
    for i in range(curvature_change_difference.shape[0]):
        ttest_results[i] = sp.stats.ttest_ind(curvature_change_greedy[i, :], curvature_change_true[i, :])[1]
    # plot the pvalues
    ax = plt.axes((.65, .55, .3, .3 * pap_ratio))
    ax.plot(np.arange(curvature_change_difference.shape[0]), -np.log10(ttest_results), color=(1, .2, .2),
            linewidth=2,
            zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_xlim((-1, 49))
    #    ax.set_ylim((-15, 5))
    fig.show()


