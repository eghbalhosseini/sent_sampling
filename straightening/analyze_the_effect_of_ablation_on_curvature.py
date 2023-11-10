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
from sent_sampling.utils.curvature_utils import compute_model_activations, compute_model_curvature
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer,GPTNeoXForCausalLM
import scipy as sp
import matplotlib

if __name__ == '__main__':
    dataset = 'ud_sentencez_token_filter_v3_wordFORM'
    modelclass = 'gpt2'
    modelname= 'EleutherAI_pythia-1.4b'
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
    sentences = np.random.choice(sentences_, 1000, replace=False)


    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-1.4b",
        revision="step143000")

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-1.4b",
        revision="step143000")
    # define an ablated model
    config = model.config
    # Ablate MLP layers in the first transformer layer
    config.architectures = ["GPTNeoXForCausalLM-MLP_ablated"]
    ablated_model = GPTNeoXForCausalLM(config)
    model.cuda()
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    all_layers=compute_model_activations(model,indexed_tokens)
    # printe that we are getting curvature
    curvature_dict=compute_model_curvature(all_layers)

    layer_to_modify = 20  # Index of the layer to modify (0-indexed)
    ablated_state_dict=model.state_dict()
    for name, param_src in ablated_state_dict.items():
        if f"gpt_neox.layers.{layer_to_modify}.mlp" in name:
            # print the name of the layer

            print(f'{name} is ablated')
            # Modify parameters for the specified layer
            param_dst = ablated_model.state_dict()[name]
            param_dst.copy_(param_src.data)
            # create an random matrix with shape of param_dst
            #random_matrix = torch.randn_like(param_dst)
            if 'weight' in name:
                eye_matrix=torch.eye(param_dst.shape[0],param_dst.shape[1])
            else:
                #create an identity matrix with shape of param_dst
                eye_matrix=torch.ones_like(param_dst)
            # repalce the param_dst with random matrix
            param_dst.copy_(eye_matrix)
            # save the ablated state dict
            ablated_state_dict[name] = param_dst
        else:
            ablated_state_dict[name] = param_src.data

    ablated_model.load_state_dict(ablated_state_dict, strict=False)
    ablated_model.cuda()
    # compute the model activation for the ablated model
    all_layers_unt=compute_model_activations(ablated_model,indexed_tokens)
    curvature_dict_ablated=compute_model_curvature(all_layers_unt)


    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    curve_ = np.stack(curvature_dict['curve'], axis=0)
    curve_ablated_ = np.stack(curvature_dict_ablated['curve'], axis=0)
    curve_change = (curve_[1:, :] - curve_[1, :])
    curve_change_ablated = (curve_ablated_[1:, :] - curve_ablated_[1, :])
    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    #if bool(re.findall(r'-ablated', modelname)):
    line_cols_unt = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .35*pap_ratio))
    for i,curv in enumerate(curve_):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
        # plot ablated model
        ax.scatter(i, np.nanmean(curve_ablated_[i]) * 180 / np.pi, s=25, color=line_cols_unt[i, :], zorder=3, edgecolor=(0, 0, 0),
                     linewidth=.5, alpha=.5,)
        ax.errorbar(i, np.nanmean(curve_ablated_[i]) * 180 / np.pi, yerr=np.nanstd(curve_ablated_[i]) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols_unt[i, :], zorder=0, alpha=.5)

    # plot a line for the average
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1, zorder=1,label='trained(checkpoint=143k)')
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_ablated_, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1, zorder=1, alpha=.5,label='ablated(checkpoint=0)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
#    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature')
    ax.set_xlabel('layer')

    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    for i, curv in enumerate(curve_change):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
        # plot ablated model
        ax.scatter(i, np.nanmean(curve_change_ablated[i]) * 180 / np.pi, s=25, color=line_cols_unt[i, :], zorder=3, edgecolor=(0, 0, 0),
                     linewidth=.5, alpha=.5)
        ax.errorbar(i, np.nanmean(curve_change_ablated[i]) * 180 / np.pi, yerr=np.nanstd(curve_change_ablated[i]) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols_unt[i, :], zorder=0, alpha=.5)

    # plot a line for the average
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1,
            zorder=1,label='trained(checkpoint=143k)')
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change_ablated, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1,
            zorder=1, alpha=.5,label='ablated,checkpoint=0')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature change')
    # show legend
    ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.2), ncol=1, frameon=False)
    # put the title
    ax.set_title(f'{modelname} on {dataset}, effect of training on curvature', fontsize=8)

    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_{dataset}_trainig_vs_ablated.pdf'), transparent=True)
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_{dataset}_trainig_vs_ablated.png'), transparent=True)

