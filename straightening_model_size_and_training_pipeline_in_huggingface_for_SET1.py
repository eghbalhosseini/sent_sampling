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
    curve_change = (curve_[1:-1, :] - curve_[1, :])
    # make a dictionary with fieldds 'curve','curve_change','all_layer_curve_all' and return the dictionary
    return dict(curve=curve_,curve_change=curve_change,all_layer_curve_all=all_layer_curve_all)

if __name__ == '__main__':
    #%%

    #modelnames='facebook/opt-125m'
    modelclass='opt'
    basemodel = 'facebook/opt-125m'
    #modelnames=['distilgpt2','gpt2','gpt2-medium','gpt2-large','gpt2-xl']
    modelnames = ["facebook/opt-125m", "facebook/opt-350m","facebook/opt-1.3b","facebook/opt-2.7b","facebook/opt-6.7b","facebook/opt-13b","facebook/opt-30b","facebook/opt-66b"]
    modelsizes=[125,330,1300,2700,6700,13000,30000,66000]
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

    # save model_curvature_dict

    #with open(os.path.join(ANALYZE_DIR,f'model_curvature_dict_{modelclass}.pkl'),'wb') as f:
    #    pickle.dump(model_curvature_dict,f)



    #%%
    # fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    # pap_ratio=8/11
    # matplotlib.rcParams['font.size'] = 6
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    #
    # ax = plt.axes((.1, .1, .65, .35 * pap_ratio))
    # for key,val in model_curvature_dict.items():
    #     True
    #     curve_change=val['curve_change']
    #     num_colors = curve_change.shape[0] + 2
    #     color_fact = num_colors + 10
    #     h0 = cm.get_cmap('inferno', color_fact)
    #     line_cols = (h0(np.arange(color_fact) / color_fact))
    #     line_cols = line_cols[2:, :]
    #     if bool(re.findall(r'-untrained', modelname)):
    #         line_cols = line_cols * 0 + (.6)
    #
    #
    #     for i, curv in enumerate(curve_change):
    #         ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
    #                linewidth=.5, alpha=1)
    #         #ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
    #         #        color=line_cols[i, :], zorder=0, alpha=1)
    #     # plot a line for the average
    #     ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1,
    #         zorder=1)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)  #
    # #    ax.set_ylim((-15, 5))
    # ax.set_ylabel(f'curvature change$')
    #
    #
    # ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    #
    # ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    # fig.show()
    # fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_gpt2_models_{dataset}_masked_{masked}.pdf'), transparent=True)
    #
    # #
    # # plot the best layer
    # best_curves=[]
    # for key, val in model_curvature_dict.items():
    #     curve_change = val['curve_change']
    #     lowest_layer=np.argmin(np.nanmean(curve_change, axis=1), axis=0)
    #     best_curve=curve_change[lowest_layer,:]
    #     best_curves.append(best_curve)
    #
    # fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    # pap_ratio=8/11
    # matplotlib.rcParams['font.size'] = 6
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    # ax = plt.axes((.1, .1, .65, .35 * pap_ratio))
    # for i, curv in enumerate(best_curves):
    #     ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
    #            linewidth=.5, alpha=1)
    #     ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
    #             color=line_cols[i, :], zorder=0, alpha=1)
    # fig.show()


    #%% mistral models
    # basemodel = 'gpt2'
    # mistral_root_ = '/om/weka/evlab/ehoseini/MyData/mistral/caprica-gpt2-small-x81'
    #
    # chkpoints = ['0', '40', '400', '4000', '40000', '400000']
    # masked = False
    # # get sentences from ext_obj
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
    # torch.cuda.empty_cache()
    # fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    # pap_ratio = 8 / 11
    # matplotlib.rcParams['font.size'] = 6
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    #
    # ax = plt.axes((.1, .55, .65, .35 * pap_ratio))
    # num_colors = len(model_curvature_dict) + 2
    # h0 = cm.get_cmap('inferno', num_colors)
    # line_cols = (h0(np.arange(num_colors) / num_colors))
    # # line_cols = line_cols[2:, :]
    # kk = 0
    # for key, val in model_curvature_dict.items():
    #     curve_change = val['curve']
    #     for i, curv in enumerate(curve_change):
    #         ax.scatter(i + kk * .05, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[kk, :], zorder=2,
    #                    edgecolor=(0, 0, 0),
    #                    linewidth=.5, alpha=1)
    #         ax.errorbar(i + kk * .05, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0,
    #                     elinewidth=1,
    #                     color=line_cols[kk, :], zorder=0, alpha=1)
    #     # # plot a line for the average
    #     ax.plot(np.arange(curve_change.shape[0]) + kk * .05, np.nanmean(curve_change, axis=1) * 180 / np.pi,
    #             color=line_cols[kk, :], linewidth=1, zorder=1)
    #     kk += 1
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)  #
    # #    ax.set_ylim((-15, 5))
    # ax.set_ylabel(f'curvature')
    # ax.set_title(f" training effect {dataset}\n Huggingface")
    #
    #
    #
    # ax = plt.axes((.1, .1, .65, .35 * pap_ratio))
    # num_colors = len(model_curvature_dict)+2
    # h0 = cm.get_cmap('inferno', num_colors)
    # line_cols = (h0(np.arange(num_colors) / num_colors))
    # #line_cols = line_cols[2:, :]
    # kk=0
    # for key, val in model_curvature_dict.items():
    #     curve_change = val['curve_change']
    #     for i, curv in enumerate(curve_change):
    #         ax.scatter(i+kk*.05, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[kk, :], zorder=2, edgecolor=(0, 0, 0),
    #                     linewidth=.5, alpha=1)
    #         ax.errorbar(i+kk*.05, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
    #                 color=line_cols[kk, :], zorder=0, alpha=1)
    #     # # plot a line for the average
    #     ax.plot(np.arange(curve_change.shape[0])+kk*.05, np.nanmean(curve_change, axis=1) * 180 / np.pi, color=line_cols[kk, :],linewidth=1,zorder=1)
    #     kk+=1
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)  #
    # #    ax.set_ylim((-15, 5))
    # ax.set_ylabel(f'curvature change$')
    # ax.set_title(f" training effect {dataset}\n Huggingface")
    # fig.show()
    # fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_mistral_models_training_{dataset}_masked_{masked}.pdf'), transparent=True)
