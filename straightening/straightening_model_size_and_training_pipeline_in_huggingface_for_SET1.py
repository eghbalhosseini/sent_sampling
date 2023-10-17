import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
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
from neural_nlp.models.gpt_neox_model import GPTNeoXPosLearnedModel,GPTNeoXPosLearnedConfig, initialize_gpt_neox_weights
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer
#AutoConfig.register('gpt-neox',GPTNeoXConfig)
AutoConfig.register('gpt-neox-pos-learned',GPTNeoXPosLearnedConfig)
#AutoModel.register(GPTNeoXConfig, GPTNeoXModel)
AutoModel.register(GPTNeoXPosLearnedConfig, GPTNeoXPosLearnedModel)

from transformers import PreTrainedTokenizer
import pickle
def align_tokens_debug(tokenized_sentences, sentences, max_num_words, additional_tokens, use_special_tokens,special_tokens):
    # sliding window approach (see https://github.com/google-research/bert/issues/66)
    # however, since this is a brain model candidate, we don't let it see future words (just like the brain
    # doesn't receive future word input). Instead, we maximize the past context of each word
    sentence_index = 0
    sentences_chain = ' '.join(sentences).split()
    previous_indices = []
    all_context=[]
    for token_index in tqdm(range(len(tokenized_sentences)), desc='token features',position=2,leave=False,disable=True):
    #    if tokenized_sentences[token_index] in additional_tokens:
    #        continue  # ignore altogether
        # combine e.g. "'hunts', '##man'" or "'jennie', '##s'"
        tokens = [
            # tokens are sometimes padded by prefixes, clear those here
            word.lstrip('##').lstrip('▁').rstrip('@@')
            for word in tokenized_sentences[previous_indices + [token_index]]]
        token_word = ''.join(tokens).lower()

        for special_token in special_tokens:
            token_word = token_word.replace(special_token, '')
        #print(token_word)
        if sentences_chain[sentence_index].lower() != token_word:
            previous_indices.append(token_index)
            continue
        previous_indices = []
        sentence_index += 1
        #print(token_index)
        context_start = max(0, token_index - max_num_words + 1)
        #print(context_start)
        context = tokenized_sentences[context_start:token_index + 1]
        if use_special_tokens and context_start > 0:  # `cls_token` has been discarded
            # insert `cls_token` again following
            # https://huggingface.co/pytorch-transformers/model_doc/roberta.html#pytorch_transformers.RobertaModel
            context = np.insert(context, 0, tokenized_sentences[0])
        #context_ids = self.tokenizer.convert_tokens_to_ids(context)
        all_context.append(context)
        #yield context
    return all_context
def get_word_locations(sentence: str, tokenizer: PreTrainedTokenizer):
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    # Get the word locations
    word_locations = {}
    current_word_tokens = []
    current_word_token_ids=[]
    i_word=0
    output_dict=dict()
    for i, token in enumerate(tokens):
        # Check if the current token is the start of a new word
        current_word_tokens.append(token)
        current_word_token_ids.append(i)
        current_candidate=''.join(tokenizer.convert_tokens_to_string(current_word_tokens))
        if  sentence.split()[:i_word] in current_candidate:
            output_dict[sentence.split()[:i_word]]=current_word_token_ids
            current_word_tokens=[]
            current_word_token_ids=[]
            i_word+=1
        else:
            continue

        # make sure length of the sentence is the same as the length of output_dict
    assert len(output_dict)==len(sentence.split())
    return output_dict
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
from transformers import AutoModel
if __name__ == '__main__':
    #%%
    #modelnames='facebook/opt-125m'

    modelclass='gpt2'
    basemodel='gpt2'
    modelnames=['distilgpt2','gpt2','gpt2-medium','gpt2-large','gpt2-xl']
    modelsizes=[82,117,345,774,1558]
    #modelclass = 'opt'
    # basemodel = 'facebook/opt-125m'
    # modelnames = ["cerebras/Cerebras-GPT-111M",
    # "cerebras/Cerebras-GPT-256M",
    # "cerebras/Cerebras-GPT-590M",
    # "cerebras/Cerebras-GPT-1.3B",
    # "cerebras/Cerebras-GPT-2.7B",
    # "cerebras/Cerebras-GPT-6.7B",
    #"cerebras/Cerebras-GPT-13B"
    #]
    #modelsizes=[125,330,1300,2700,6700]#,13000,30000,66000]
    #modelsizes=[111,256,590,1300,2700,6700]

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
    del ext_obj
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    # tokenize sentences
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

    model_curvature_dict=dict()
    model_curvature_dict_file=Path(os.path.join(ANALYZE_DIR,f'model_curvature_dict_{modelclass}.pkl'))
    if model_curvature_dict_file.exists():
        model_curvature_dict=pickle.load(open(model_curvature_dict_file,'rb'))
    else:
        for modelname in tqdm(modelnames):
            modelname_ = modelname.replace('/', '_')
            model_file=Path(os.path.join(ANALYZE_DIR,f'model_curvature_dict_{modelname_}.pkl'))
            # if modele_file  already exist then skip
            if not model_file.exists():
                if masked==True:
                    model = AutoModelForMaskedLM.from_pretrained(modelname)
                else:
                    model = AutoModel.from_pretrained(modelname)
            # send model to gpu
                model.cuda()

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
                # add curvature_dict to model_curvature_dict
             #   model_curvature_dict[modelname]=curvature_dict
                # save curvature dict
                # replace / with _

                with open(os.path.join(ANALYZE_DIR,f'model_curvature_dict_{modelname_}.pkl'),'wb') as f:
                    pickle.dump(curvature_dict,f)
            else:
                with open(os.path.join(ANALYZE_DIR,f'model_curvature_dict_{modelname_}.pkl'),'rb') as f:
                    model_curvature_dict[modelname]=pickle.load(f)


    # save model_curvature_dict

    #with open(os.path.join(ANALYZE_DIR,f'model_curvature_dict_{modelclass}.pkl'),'wb') as f:
    #    pickle.dump(model_curvature_dict,f)



    #%%
    fig = plt.figure(figsize=(5.5,9), dpi=200, frameon=False)
    pap_ratio=5.5/9
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = len(modelnames) + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .55, .25 * pap_ratio))
    kk=0
    for key,val in model_curvature_dict.items():

        curve_ = val['curve']
        curve_change = (curve_[1:, :] - curve_[1, :])

        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=5, color=line_cols[kk,:], zorder=2, edgecolor=(0, 0, 0),linewidth=.5, alpha=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=line_cols[kk,:], linewidth=1,
            zorder=1,label=key)
        ax.fill_between(np.arange(curve_change.shape[0]),
                        np.nanmean(curve_change, axis=1)* 180 / np.pi - ((np.nanstd(curve_change, axis=1))*180 / np.pi)/np.sqrt(curve_change.shape[1]),
                        np.nanmean(curve_change, axis=1)* 180 / np.pi + ((np.nanstd(curve_change, axis=1))*180 / np.pi)/np.sqrt(curve_change.shape[1]),
                        color=line_cols[kk, :], alpha=.2, zorder=1)
        kk+=1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change$')
    # set xlim to [-1,49]
    ax.set_xlim([-1,49])
    ax.set_ylim([-12,2])


# show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")

    # plot the best layer
    best_curves=[]
    for key, val in model_curvature_dict.items():
        curve_ = val['curve']
        curve_change = (curve_[1:, :] - curve_[1, :])
        lowest_layer=np.argmin(np.nanmean(curve_change, axis=1), axis=0)
        best_curve=curve_change[lowest_layer,:]
        best_curves.append(best_curve)

    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax = plt.axes((.75, .1, .2, .25 * pap_ratio))
    for i, curv in enumerate(best_curves):
        ax.scatter(modelsizes[i], np.nanmean(curv) * 180 / np.pi, s=15, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
               linewidth=.5, alpha=1)
        ax.errorbar(modelsizes[i], np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                color=line_cols[i, :], zorder=0, alpha=1)
    # draw a line connecting them
    ax.plot(modelsizes, np.nanmean(best_curves, axis=1) * 180 / np.pi, color=[0, 0, 0], linewidth=2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change$')
    ax.set_xlabel(f'model size (millions of parameters)')
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    # make the x axis log
    ax.set_xscale('log')
    ax.set_ylim([-14, -2])
    #fig.show()
    basemodel_ = basemodel.replace('/', '_')
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_change_{basemodel_}_models_{dataset}_masked_{masked}_sem.eps'),dpi=200,format='eps',bbox_inches='tight',transparent=True)

    #%%  do the same for curvature
    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8 / 11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = len(modelnames) + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .45, .3 * pap_ratio))
    kk = 0

    for key, val in model_curvature_dict.items():
        curve_change = val['curve']
        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=10,
                   color=[0, 0, 0], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi,
                color=line_cols[kk, :], linewidth=1,
                zorder=1, label=key)
        kk += 1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature$')
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")

    # plot the best layer
    best_curves = []
    for key, val in model_curvature_dict.items():
        curve_change = val['curve']
        lowest_layer = np.argmin(np.nanmean(curve_change, axis=1), axis=0)
        best_curve = curve_change[lowest_layer, :]
        best_curves.append(best_curve)

    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax = plt.axes((.65, .1, .3, .3 * pap_ratio))
    for i, curv in enumerate(best_curves):
        ax.scatter(modelsizes[i], np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2,
                   edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(modelsizes[i], np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0,
                    elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # draw a line connecting them
    ax.plot(modelsizes, np.nanmean(best_curves, axis=1) * 180 / np.pi, color=[0, 0, 0], linewidth=2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature')
    ax.set_xlabel(f'model size (millions of parameters)')
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    # make the x axis log
    ax.set_xscale('log')

    fig.show()
    basemodel_ = basemodel.replace('/', '_')
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{basemodel_}_models_{dataset}_masked_{masked}.eps'),
                dpi=200, format='eps', bbox_inches='tight', transparent=True)

#%% mistral models
    basemodel = 'gpt2'
    mistral_root_ = '/om/weka/evlab/ehoseini/MyData/mistral/caprica-gpt2-small-x81'
    chkpoints = ['0', '40', '400', '4000', '40000', '400000']
    masked = False
    modelclass = 'gpt2'
    basemodel = 'gpt2'
    modelnames = ['0', '40', '400', '4000', '40000','400000']
    modelsizes = [0, 40, 400, 4000, 40000,400000]

    # get sentences from ext_obj
    # tokenizer = AutoTokenizer.from_pretrained(basemodel)
    # # tokenize sentences
    # tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # # get ids
    # indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    # # build tokenizer
    #
    # # get attention mask
    # # feed individual sentences to model
    # # model = AutoModelForCausalLM.from_pretrained(modelnames)
    # model_curvature_dict = dict()
    # for chkpoint in tqdm(chkpoints):
    #     # send model to gpu
    #     modelConfig = AutoConfig.from_pretrained(Path(mistral_root_, f'ckpt_{chkpoint}', 'config.json').__str__())
    #     model = AutoModelForCausalLM.from_pretrained(Path(mistral_root_, f'ckpt_{chkpoint}').__str__(),config=modelConfig, state_dict=None)
    #     model.cuda()
    #     # get activations
    #     # print that we are getting activations
    #     print('getting activations for checkpoint: {}'.format(chkpoint))
    #     all_layers = compute_model_activations(model, indexed_tokens)
    #     # printe that we are getting curvature
    #     print('getting curvature for checkpoint: {}'.format(chkpoint))
    #     curvature_dict = compute_model_curvature(all_layers)
    #     # empty cuda cache
    #     torch.cuda.empty_cache()
    #     # delete model
    #     del model
    #     # add curvature_dict to model_curvature_dict
    #     model_curvature_dict[f'mistral_{chkpoint}'] = curvature_dict
    # # save model_curvature_dict
    #
    # with open(os.path.join(ANALYZE_DIR, 'model_curvature_dict_training.pkl'), 'wb') as f:
    #     pickle.dump(model_curvature_dict, f)
    # #%%
    # load data
    with open(os.path.join(ANALYZE_DIR, 'model_curvature_dict_training.pkl'), 'rb') as f:
        model_curvature_dict = pickle.load(f)
#%%
    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8 / 11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = len(modelnames) + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .45, .3 * pap_ratio))
    kk = 0

    for key, val in model_curvature_dict.items():
        curve_change = val['curve_change']
        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=10,
                   color=[0, 0, 0], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi,
                color=line_cols[kk, :], linewidth=1,
                zorder=1, label=key)
        kk += 1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change$')
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")

    # plot the best layer
    best_curves = []
    for key, val in model_curvature_dict.items():
        curve_change = val['curve_change']
        lowest_layer = np.argmin(np.nanmean(curve_change, axis=1), axis=0)
        best_curve = curve_change[lowest_layer, :]
        best_curves.append(best_curve)

    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax = plt.axes((.65, .1, .3, .3 * pap_ratio))
    for i, curv in enumerate(best_curves):
        ax.scatter(modelsizes[i], np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2,
                   edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(modelsizes[i], np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0,
                    elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # draw a line connecting them
    ax.plot(modelsizes, np.nanmean(best_curves, axis=1) * 180 / np.pi, color=[0, 0, 0], linewidth=2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change$')
    ax.set_xlabel(f'model size (millions of parameters)')
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    # make the x axis log
    ax.set_xscale('log')

    fig.show()
    basemodel_ = basemodel.replace('/', '_')
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_change_training_{basemodel_}_models_{dataset}_masked_{masked}.eps'),
                dpi=200, format='eps', bbox_inches='tight', transparent=True)

    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8 / 11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = len(modelnames) + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .45, .3 * pap_ratio))
    kk = 0

    for key, val in model_curvature_dict.items():
        curve_change = val['curve']
        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=10,
                   color=[0, 0, 0], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi,
                color=line_cols[kk, :], linewidth=1,
                zorder=1, label=key)
        kk += 1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature$')
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")

    # plot the best layer
    best_curves = []
    for key, val in model_curvature_dict.items():
        curve_change = val['curve']
        lowest_layer = np.argmin(np.nanmean(curve_change, axis=1), axis=0)
        best_curve = curve_change[lowest_layer, :]
        best_curves.append(best_curve)

    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax = plt.axes((.65, .1, .3, .3 * pap_ratio))
    for i, curv in enumerate(best_curves):
        ax.scatter(modelsizes[i], np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2,
                   edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(modelsizes[i], np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0,
                    elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # draw a line connecting them
    ax.plot(modelsizes, np.nanmean(best_curves, axis=1) * 180 / np.pi, color=[0, 0, 0], linewidth=2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature')
    ax.set_xlabel(f'model size (millions of parameters)')
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    # make the x axis log
    ax.set_xscale('log')

    fig.show()
    basemodel_ = basemodel.replace('/', '_')
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{basemodel_}_training_models_{dataset}_masked_{masked}.eps'),
                dpi=200, format='eps', bbox_inches='tight', transparent=True)

    #%% gpt_neox models
    checkpoint_='/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_1b_v2/gpt2/checkpoints_4/'
    # basemodel = 'gpt2'
    modelsizes = [1,10, 100,1000]
    modelclass='gpt_neox'
    modelnames=['gpt2-1m','gpt2-10m','gpt2-100m','gpt2-1b']
    ckpnts = '310000'
    weight_file = f'{checkpoint_}/global_step{ckpnts}/pytorch_model.bin',
    config_file = f'{checkpoint_}/global_step{ckpnts}/config.json'
    weight_file='/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_1b_v2/gpt2/checkpoints_4/global_step310000/pytorch_model.bin'
    modelConfig = GPTNeoXPosLearnedConfig.from_pretrained(config_file)
    model = GPTNeoXPosLearnedModel.from_pretrained(weight_file,config=modelConfig, state_dict=None)
    # chkpoints = ['0', '40', '400', '4000', '40000', '400000']
    basemodel = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    # # tokenize sentences
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    model.cuda()
    model_curvature_dict=dict()
    model_curvature_dict_file = Path(os.path.join(ANALYZE_DIR, f'model_curvature_dict_{modelclass}.pkl'))
    if model_curvature_dict_file.exists():
        model_curvature_dict = pickle.load(open(model_curvature_dict_file, 'rb'))
    else:
    #get activations
    # print that we are getting activations
        print('getting activations for checkpoint: {}'.format(ckpnts))
        all_layers = compute_model_activations(model, indexed_tokens)
        # printe that we are getting curvature
        print('getting curvature for checkpoint: {}'.format(ckpnts))
        curvature_dict = compute_model_curvature(all_layers)
        # empty cuda cache
        torch.cuda.empty_cache()
        # delete model
        del model
        model_curvature_dict[f'gpt_neox_1b'] = curvature_dict
        #     # add curvature_dict to model_curvature_dict
        weight_file = '/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_5/global_step14250/pytorch_model.bin'
        config_file = f'/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_5/global_step14250/config.json'
        modelConfig = GPTNeoXPosLearnedConfig.from_pretrained(config_file)
        model = GPTNeoXPosLearnedModel.from_pretrained(weight_file, config=modelConfig, state_dict=None)
        model.cuda()
        all_layers = compute_model_activations(model, indexed_tokens)
        curvature_dict = compute_model_curvature(all_layers)
        torch.cuda.empty_cache()
        del model
        model_curvature_dict[f'gpt_neox_100m'] = curvature_dict
        #

        weight_file = '/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_10m_v2/gpt2/checkpoints_6/global_step2000/pytorch_model.bin'
        config_file = f'/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_10m_v2/gpt2/checkpoints_6/global_step2000/config.json'
        modelConfig = GPTNeoXPosLearnedConfig.from_pretrained(config_file)
        model = GPTNeoXPosLearnedModel.from_pretrained(weight_file, config=modelConfig, state_dict=None)
        model.cuda()
        all_layers = compute_model_activations(model, indexed_tokens)
        curvature_dict = compute_model_curvature(all_layers)
        torch.cuda.empty_cache()
        del model
        model_curvature_dict[f'gpt_neox_10m'] = curvature_dict

        weight_file = '/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_1m_v2/gpt2/checkpoints_7/global_step1000/pytorch_model.bin'
        config_file = f'/om/weka/evlab/ehoseini/MyData/miniBERTa_training/miniBERTa_1m_v2/gpt2/checkpoints_7/global_step1000/config.json'
        modelConfig = GPTNeoXPosLearnedConfig.from_pretrained(config_file)
        model = GPTNeoXPosLearnedModel.from_pretrained(weight_file, config=modelConfig, state_dict=None)
        model.cuda()
        all_layers = compute_model_activations(model, indexed_tokens)
        curvature_dict = compute_model_curvature(all_layers)
        torch.cuda.empty_cache()
        del model
        model_curvature_dict[f'gpt_neox_1m'] = curvature_dict
        modelclass='gpt_neox'
        with open(os.path.join(ANALYZE_DIR, f'model_curvature_dict_{modelclass}.pkl'), 'wb') as f:
            pickle.dump(model_curvature_dict,f)

    keys=['gpt_neox_1m', 'gpt_neox_10m','gpt_neox_100m','gpt_neox_1b']

    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = len(modelnames) +1
    color_fact = num_colors + 2
    h0 = cm.get_cmap('viridis', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .55, .25 * pap_ratio))
    kk = 0

    for key in keys:
        val = model_curvature_dict[key]
        curve_ = val['curve']
        curve_change = (curve_[1:, :] - curve_[1, :])
        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=5,
                   color=line_cols[kk, :], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi,
                color=line_cols[kk, :], linewidth=1,
                zorder=1, label=key)
        # plot standard devation as fill-between
        ax.fill_between(np.arange(curve_change.shape[0]),
                        np.nanmean(curve_change, axis=1) * 180 / np.pi - (
                                    (np.nanstd(curve_change, axis=1)) * 180 / np.pi) / np.sqrt(curve_change.shape[1]),
                        np.nanmean(curve_change, axis=1) * 180 / np.pi + (
                                    (np.nanstd(curve_change, axis=1)) * 180 / np.pi) / np.sqrt(curve_change.shape[1]),
                        color=line_cols[kk, :], alpha=.2, zorder=1)

        kk += 1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change$')
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # plot the best layer
    best_curves = []
    for key in keys:
        val = model_curvature_dict[key]
        curve_ = val['curve']
        curve_change = (curve_[1:, :] - curve_[1, :])
        lowest_layer = np.argmin(np.nanmean(curve_change, axis=1), axis=0)
        best_curve = curve_change[lowest_layer, :]
        best_curves.append(best_curve)

    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax = plt.axes((.75, .1, .2, .25 * pap_ratio))
    for i, curv in enumerate(best_curves):
        ax.scatter(modelsizes[i], np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2,
                   edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(modelsizes[i], np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0,
                    elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # draw a line connecting them
    ax.plot(modelsizes, np.nanmean(best_curves, axis=1) * 180 / np.pi, color=[0, 0, 0], linewidth=2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature change$')
    ax.set_xlabel(f'model size (millions of parameters)')
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    # make the x axis log
    ax.set_xscale('log')

    #fig.show()
    basemodel_ = basemodel.replace('/', '_')
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_change_{basemodel_}_models_{dataset}_masked_{masked}_with_sem.eps'),
                dpi=200, format='eps', bbox_inches='tight', transparent=True)
    #%%
    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8 / 11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = len(modelnames) + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .45, .3 * pap_ratio))
    kk = 0

    for key in keys:
        val = model_curvature_dict[key]
        curve_change = val['curve']
        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=10,
                   color=[0, 0, 0], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi,
                color=line_cols[kk, :], linewidth=1,
                zorder=1, label=key)
        kk += 1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature$')
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")

    # plot the best layer
    best_curves = []
    for key in keys:
        val = model_curvature_dict[key]
        curve_change = val['curve']
        lowest_layer = np.argmin(np.nanmean(curve_change, axis=1), axis=0)
        best_curve = curve_change[lowest_layer, :]
        best_curves.append(best_curve)

    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax = plt.axes((.65, .1, .3, .3 * pap_ratio))
    for i, curv in enumerate(best_curves):
        ax.scatter(modelsizes[i], np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2,
                   edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(modelsizes[i], np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0,
                    elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # draw a line connecting them
    ax.plot(modelsizes, np.nanmean(best_curves, axis=1) * 180 / np.pi, color=[0, 0, 0], linewidth=2, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(f'curvature$')
    ax.set_xlabel(f'model size (millions of parameters)')
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # show the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    # make the x axis log
    ax.set_xscale('log')

    fig.show()
    basemodel_ = basemodel.replace('/', '_')
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature{basemodel_}_models_{dataset}_masked_{masked}.eps'),
                dpi=200, format='eps', bbox_inches='tight', transparent=True)

    #%%

    asemodel = 'gpt2'
    mistral_root_ = '/om/weka/evlab/ehoseini/MyData/mistral/caprica-gpt2-small-x81'
    chkpoints = ['0', '40', '400', '4000', '40000', '400000']
    masked = False
    modelclass = 'gpt2'
    basemodel = 'gpt2'
    modelnames = ['0', '40', '400', '4000', '40000', '400000']
    modelsizes = np.arange(0, 4200, 200)

    # get sentences from ext_obj
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    # # tokenize sentences
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    # build tokenizer

    # get attention mask
    # feed individual sentences to model
    # model = AutoModelForCausalLM.from_pretrained(modelnames)
    model_curvature_dict = dict()
    for chkpoint in tqdm(chkpoints):
        # send model to gpu
        modelConfig = AutoConfig.from_pretrained(Path(mistral_root_, f'ckpt_{chkpoint}', 'config.json').__str__())
        model = AutoModelForCausalLM.from_pretrained(Path(mistral_root_, f'ckpt_{chkpoint}').__str__(),config=modelConfig, state_dict=None)
        model.cuda()
        # get activations
        # print that we are getting activations
        print('getting activations for checkpoint: {}'.format(chkpoint))
        all_layers = compute_model_activations(model, indexed_tokens)
        # printe that we are getting curvature
        print('getting curvature for checkpoint: {}'.format(chkpoint))
        curvature_dict = compute_model_curvature(all_layers)
        # empty cuda cache
        torch.cuda.empty_cache()
        # delete model
        del model
        # add curvature_dict to model_curvature_dict
        model_curvature_dict[f'mistral_{chkpoint}'] = curvature_dict
    # save model_curvature_dict
    with open(os.path.join(ANALYZE_DIR, f'curvature_{basemodel}_mistral_models_long_{dataset}_masked_{masked}.pkl'), 'wb') as f:
        pickle.dump(model_curvature_dict, f)

