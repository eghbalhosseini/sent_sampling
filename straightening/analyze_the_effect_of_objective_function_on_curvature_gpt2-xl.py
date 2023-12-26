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
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer,GPTNeoXForCausalLM, AutoModelForSequenceClassification,AutoModelForCausalLM,AutoModelForTokenClassification
import scipy as sp
import matplotlib

if __name__ == '__main__':
    dataset = 'ud_sentencez_token_filter_v3_wordFORM'
    modelclass = 'gpt2'
    modelname= 'gpt2'
    masked = False
    dataset = 'ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # emcpty cuda cache
    torch.cuda.empty_cache()
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences_all = [x['text'] for x in ext_obj.data_]
    sentences_words = [x['word_FORM'] for x in ext_obj.data_]
    good_sent_id = np.where(np.asarray([len(x['word_FORM']) == len(x['surprisal_3']) for x in ext_obj.data_]))[0]
    sentences_ = [sentences_all[i] for i in good_sent_id]
    # sample 1000 sentences
    sentences = np.random.choice(sentences_, 1000, replace=False)


    model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelclass)
    # define an ablated model
    config = model.config
    # Ablate MLP layers in the first transformer layer

    base_model=AutoModel.from_pretrained(modelname)
    class_model =  AutoModelForTokenClassification.from_pretrained("brad1141/gpt2-finetuned-comp2")
    tokenizer_class = AutoTokenizer.from_pretrained("George-Ogden/gpt2-finetuned-mnli")
    class_model = AutoModelForSequenceClassification.from_pretrained("George-Ogden/gpt2-finetuned-mnli")

    base_model.cuda()
    class_model.cuda()

    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    tokenized_class = [tokenizer_class.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    indexed_tokens_class=[tokenizer_class.convert_tokens_to_ids(x) for x in tokenized_class]

    all_layers=compute_model_activations(base_model,indexed_tokens)
    curvature_dict_base = compute_model_curvature(all_layers)

    all_layers=compute_model_activations(class_model,indexed_tokens_class)
    curvature_dict_class = compute_model_curvature(all_layers)


    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    curve_ = np.stack(curvature_dict_base['curve'], axis=0)
    curve_change = (curve_[1:, :] - curve_[1, :])

    curve_class = np.stack(curvature_dict_class['curve'], axis=0)
    curve_change_class = (curve_class[1:, :] - curve_class[1, :])

    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    # if bool(re.findall(r'-untrained', modelname)):
    line_cols_unt = line_cols * 0 + (.6)
    # if bool(re.findall(r'-untrained', modelname)):
    line_cols_unt = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .35 * pap_ratio))

    # plot a line for the average
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(.5, .5, .5), linewidth=1,
            zorder=2, marker='o', markersize=4, label='Universal Dependencies')
    # add fill between
    ax.fill_between(np.arange(curve_.shape[0]),
                    np.nanmean(curve_, axis=1) * 180 / np.pi - np.nanstd(curve_, axis=1) * 180 / np.pi,
                    np.nanmean(curve_, axis=1) * 180 / np.pi + np.nanstd(curve_, axis=1) * 180 / np.pi, alpha=.2,
                    color=(.5, .5, .5))
    ax.plot(np.arange(curve_class.shape[0]), np.nanmean(curve_class, axis=1) * 180 / np.pi, color=(1, 0, 0), linewidth=1,
            marker='o', markersize=4, zorder=2, label='OpenWebText')
    # add fill between
    ax.fill_between(np.arange(curve_class.shape[0]),
                    np.nanmean(curve_class, axis=1) * 180 / np.pi - np.nanstd(curve_class, axis=1) * 180 / np.pi,
                    np.nanmean(curve_class, axis=1) * 180 / np.pi + np.nanstd(curve_class, axis=1) * 180 / np.pi, alpha=.2,
                    color=(1, 0, 0))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    #    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature')
    ax.set_xlabel('layer')

    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    # plot a line for the average
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(.5, .5, .5),
            linewidth=1, marker='o', markersize=4, zorder=2, label='Universal  Dependencies')
    # add fill between
    ax.fill_between(np.arange(curve_change.shape[0]),
                    np.nanmean(curve_change, axis=1) * 180 / np.pi - np.nanstd(curve_change, axis=1) * 180 / np.pi,
                    np.nanmean(curve_change, axis=1) * 180 / np.pi + np.nanstd(curve_change, axis=1) * 180 / np.pi,
                    color=(.5, .5, .5), alpha=.2)

    ax.plot(np.arange(curve_change_class.shape[0]), np.nanmean(curve_change_class, axis=1) * 180 / np.pi, color=(1, 0, 0),
            linewidth=1, marker='o', zorder=5, markersize=4, label='OpenWebText')
    # add fill between
    ax.fill_between(np.arange(curve_change_class.shape[0]),
                    np.nanmean(curve_change_class, axis=1) * 180 / np.pi - np.nanstd(curve_change_class, axis=1) * 180 / np.pi,
                    np.nanmean(curve_change_class, axis=1) * 180 / np.pi + np.nanstd(curve_change_class, axis=1) * 180 / np.pi,
                    color=(1, 0, 0), alpha=.2)

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
