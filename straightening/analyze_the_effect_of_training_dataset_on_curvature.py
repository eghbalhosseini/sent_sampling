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
def uppercount(str_in):
    count=0
    for i in str_in:
        if(i.isupper()):
            count=count+1
    return count
# works on panda series
def fix_period_space(x,field):
    x_1=x[field]
    if x_1[x_1.index[-1]] == '.':
        x_1=x_1.drop(x_1.index[-1])
        x=x.drop(x.index[-1])
        x.at[x.index[-1],field]=x_1[x_1.index[-1]]+'.'
    # drop last index
    return x

def fix_punct_space(x_in,field):
    f_tag = lambda x: True if re.match(r'\,', x) else False
    while any(x_in[field].apply(lambda x: f_tag(x))):
        x1 = x_in[field].apply(lambda x: f_tag(x))
        loc=np.argwhere(x1.values==True)[0]
        x_1=x_1.drop(x_1.index[loc])


        pass

VALID_punctuations=''.join([chr(x) for x in [33, 34, 39, 44, 45, 46, 58, 59, 63]])
VALID_spaces=''.join([' ', '\t', '\n', '\r', '\x0b', '\x0c'])
VALID_CHARACTERS=string.ascii_letters + string.digits+VALID_spaces+VALID_punctuations
f_str = lambda x: type(x) == str
f_cont = lambda x: np.diff(x) == 1
f_char = lambda x: not bool(set(''.join(x)) - set(VALID_CHARACTERS))
f_upper = lambda x: uppercount(x) > max_upper_coun_th
f_odd_quoate = lambda x: True if re.match(r'\"', x) else False
f_weird_start = lambda x: True if re.match(r'\'', x) or re.match(r':', x) else False
f_quota_tag = lambda x: True if re.match(r'--', x) else False
f_ngram = lambda x: not bool(set(x.apply(lambda y: y.lower())) - ngram)
f_length = lambda x: all([len(x) > min_sent_len, len(x) <= max_sent_len])
f_dot_det = lambda x: x[x.index[-1]] == '.' and x[x.index[-2]] != '.'
f_dot_rep = lambda x, flag: x


import re
if __name__ == '__main__':
    modelclass = 'gpt2'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    modelname= 'gpt2-xl'
    masked = False
    max_upper_coun_th = 5
    min_sent_len=10
    max_sent_len=50
    mini_dataset = load_from_disk(
        os.path.join('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext/',
                     'train'))
    sample_dataset = load_from_disk(
            os.path.join('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext_sample/','train'))

    ext_obj = extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences = [x['text'] for x in ext_obj.data_]
    sentences_words = [x['word_FORM'] for x in ext_obj.data_]
    good_sent_id = np.where(np.asarray([len(x['word_FORM']) == len(x['surprisal_3']) for x in ext_obj.data_]))[0]
    sentences_ = [sentences[i] for i in good_sent_id]
    ud_sentences=sentences_

        # subsample 1000 sentences
    #sample_dataset = mini_dataset.select(range(10000))
    # save sample dataset
    #sample_dataset.save_to_disk(os.path.join('/om2/user/ehoseini/MyData/openwebtext_sample/','train'))
    #del mini_dataset
    tokenizer = AutoTokenizer.from_pretrained(modelclass)

    tokenized_text_ud = [tokenizer.tokenize(x) for x in sentences]
    ud_lengths=[len(x) for x in tokenized_text_ud]
    indexed_tokens_ud = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text_ud]
    # randomly permute ud_lengths
    ud_lengths=np.random.permutation(ud_lengths)

    tokenized_text_ow = [tokenizer.tokenize(x) for x in sample_dataset['text']]
    indexed_tokens_ow = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text_ow]





    sentences_split=[]
    for id, sent in tqdm(enumerate(sample_dataset['text'])):
        sentence_split = re.split(r' *[\.\?!][\'"\)\]]* *', sent.replace('\n', ' '))
        sentences_split.extend(sentence_split)
    sentences_length=[len(x) for x in sentences_split]
    # appla function f_str to each element word in each element of  of sentences_split
    selected=np.argwhere(np.logical_and(np.array(sentences_length)>30,np.array(sentences_length)<100))
    # random select len(ud_lengths) sentences from selected
    sentences_split_ow=[sentences_split[int(i)] for i in np.random.permutation(selected)[:len(indexed_tokens_ud)]]

    tokenized_text_ow = [tokenizer.tokenize(x) for x in sentences_split_ow]
    indexed_tokens_ow = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text_ow]

    # cut them based on ud_lengt
    # get ids

    # make shorter sequences of length 128 from inde
    #indexed_tokens = [x[:128] for x in indexed_tokens]
    # get sentences from ext_obj
    model = AutoModel.from_pretrained(modelname)
    # drop (ln_f): LayerNorm((1600,), eps=1e-05, elementwise_affine=True) of the model
    #model.ln_f = torch.nn.Identity()
    model.cuda()
    # move index_tokens to cuda


    #print('getting activations for model: {}'.format(modelname))
    all_layers=compute_model_activations(model,indexed_tokens_ud)
    curvature_dict_ud=compute_model_curvature(all_layers)

    all_layers=compute_model_activations(model,indexed_tokens_ow)
    curvature_dict_ow=compute_model_curvature(all_layers)

    # compute the model activation for the untrained model

    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    curve_ud = np.stack(curvature_dict_ud['curve'], axis=0)
    curve_change_ud = (curve_ud[1:, :] - curve_ud[1, :])
    # get curvature for ow
    curve_ow = np.stack(curvature_dict_ow['curve'], axis=0)
    curve_change_ow = (curve_ow[1:, :] - curve_ow[1, :])
    num_colors = curve_ow.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    #if bool(re.findall(r'-untrained', modelname)):
    line_cols_unt = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .35*pap_ratio))


    # plot a line for the average
    ax.plot(np.arange(curve_ud.shape[0]), np.nanmean(curve_ud, axis=1) * 180 / np.pi, color=(.5, .5, .5), linewidth=1, zorder=2,marker='o',markersize=4,label='Universal Dependencies')
    # add fill between
    ax.fill_between(np.arange(curve_ud.shape[0]), np.nanmean(curve_ud, axis=1) * 180 / np.pi - np.nanstd(curve_ud, axis=1) * 180 / np.pi,
                    np.nanmean(curve_ud, axis=1) * 180 / np.pi + np.nanstd(curve_ud, axis=1) * 180 / np.pi, alpha=.2, color=(.5, .5, .5))
    ax.plot(np.arange(curve_ud.shape[0]), np.nanmean(curve_ow, axis=1) * 180 / np.pi, color=(1, 0, 0), linewidth=1,marker='o',markersize=4, zorder=2,label='OpenWebText')
    # add fill between
    ax.fill_between(np.arange(curve_ud.shape[0]), np.nanmean(curve_ow, axis=1) * 180 / np.pi - np.nanstd(curve_ow, axis=1) * 180 / np.pi,
                    np.nanmean(curve_ow, axis=1) * 180 / np.pi + np.nanstd(curve_ow, axis=1) * 180 / np.pi, alpha=.2, color=(1, 0, 0))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
#    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature')
    ax.set_xlabel('layer')

    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    # plot a line for the average
    ax.plot(np.arange(curve_change_ud.shape[0]), np.nanmean(curve_change_ud, axis=1) * 180 / np.pi, color=(.5, .5, .5), linewidth=1,marker='o',markersize=4,zorder=2,label='Universal  Dependencies')
    # add fill between
    ax.fill_between(np.arange(curve_change_ud.shape[0]), np.nanmean(curve_change_ud, axis=1) * 180 / np.pi - np.nanstd(curve_change_ud, axis=1) * 180 / np.pi,
                    np.nanmean(curve_change_ud, axis=1) * 180 / np.pi + np.nanstd(curve_change_ud, axis=1) * 180 / np.pi,color=(.5, .5, .5), alpha=.2)

    ax.plot(np.arange(curve_change_ow.shape[0]), np.nanmean(curve_change_ow, axis=1) * 180 / np.pi, color=(1, 0, 0), linewidth=1,marker='o',zorder=5,markersize=4,label='OpenWebText')
    # add fill between
    ax.fill_between(np.arange(curve_change_ow.shape[0]), np.nanmean(curve_change_ow, axis=1) * 180 / np.pi - np.nanstd(curve_change_ow, axis=1) * 180 / np.pi,
                    np.nanmean(curve_change_ow, axis=1) * 180 / np.pi + np.nanstd(curve_change_ow, axis=1) * 180 / np.pi, color=(1, 0, 0), alpha=.2)



    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature change')
    # show legend
    ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.2), ncol=1, frameon=False)
    # put the title
    ax.set_title(f'{modelname} on openwebtext, curvature', fontsize=8)

    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_openwebtext.pdf'), transparent=True)
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_openwebtext.png'), transparent=True)

