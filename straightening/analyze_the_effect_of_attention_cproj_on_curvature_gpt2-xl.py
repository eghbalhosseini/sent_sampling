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
    modelname= 'gpt2-xl'
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
    n_sent=500
    sentences = np.random.choice(sentences_, n_sent, replace=False)


    model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelclass)
    # define an ablated model
    config = model.config
    # Ablate MLP layers in the first transformer layer
    config.architectures = ["GPT2LMHeadModel-MLP_ablated"]
    ablated_model = AutoModel.from_config(config)
    nonablated_model = AutoModel.from_config(config)
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

    layers_to_ablate=[5, 15,25,35,45]
    ablated_curvatures=[]
    for layer_to_modify in layers_to_ablate:
        ablated_state_dict=model.state_dict()
        nonablated_state_dict=ablated_model.state_dict()
        for name, param_src in ablated_state_dict.items():
            if f"h.{layer_to_modify}.attn.c_proj" in name:
                # print the name of the layer
                print(f'{name} is ablated')
                # Modify parameters for the specified layer
                param_dst = ablated_model.state_dict()[name]
                param_dst.copy_(param_src.data)
                # create an random matrix with shape of param_dst
                #random_matrix = torch.randn_like(param_dst)
                if 'weight' in name:
                    dim_2=config.n_embd/config.n_head
                    eye_matrix=torch.eye(param_dst.shape[0],param_dst.shape[1],device=param_dst.device)
                    # repeat eye_matrix 3 times
                    # replace the beging part of param_dst with eye_matrix
                    new_weight=eye_matrix
                    # make sure new_weight has the same shape as param_dst
                    assert new_weight.shape==param_dst.shape
                else:
                    #create an identity matrix with shape of param_dst
                    new_weight=torch.ones_like(param_dst,device=param_dst.device)
                # repalce the param_dst with random matrix
                param_dst.copy_(new_weight)
                # save the ablated state dict
                ablated_state_dict[name] = param_dst
            else:
                ablated_state_dict[name] = param_src.data
            # add nonablated state dict to the ablated state dict
            nonablated_state_dict[name] = param_src.data
        # print that you're doing ablation on the layer
        print(f'ablation on layer {layer_to_modify}')
        ablated_model.load_state_dict(ablated_state_dict, strict=False)
        ablated_model.cuda()
        all_layers_unt = compute_model_activations(ablated_model, indexed_tokens)
        curvature_dict_ablated = compute_model_curvature(all_layers_unt)
        ablated_curvatures.append(curvature_dict_ablated)
        if layer_to_modify == 5:
            nonablated_model.load_state_dict(nonablated_state_dict, strict=False)
            nonablated_model.cuda()
            all_layers_nonablated=compute_model_activations(nonablated_model,indexed_tokens)
            curvature_dict_nonablated=compute_model_curvature(all_layers_nonablated)

    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8 / 11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    curve_ = np.stack(curvature_dict_nonablated['curve'], axis=0)
    curve_change = (curve_[1:, :] - curve_[1, :])
    curve_ablated_ = [np.stack(curvature_dict_ablated['curve'], axis=0) for curvature_dict_ablated in
                      ablated_curvatures]
    curve_change_ablated = [(x[1:, :] - x[1, :]) for x in curve_ablated_]
    num_colors = curve_.shape[0] + 10
    color_fact = num_colors
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    # if bool(re.findall(r'-ablated', modelname)):
    line_cols_unt = line_cols * 0 + (.6)
    # make 5 subplot y location for each ablation each with height of .15, since the page goes from 0 to 1, need to adjust it.
    y_loc = np.linspace(.05, .9, len(layers_to_ablate) + 1)[:-1]
    y_loc = y_loc[::-1]
    for idx, y_ in enumerate(y_loc):
        ax = plt.axes((.05, y_, .4, .2 * pap_ratio))
        ax.scatter(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, s=10, color=(0, 0, 0),
                   zorder=2, edgecolor=(.5, .5, .5), linewidth=.5, alpha=1)
        # do fill between
        ax.fill_between(np.arange(curve_.shape[0]),
                        np.nanmean(curve_, axis=1) * 180 / np.pi - np.nanstd(curve_, axis=1) * 180 / np.pi / np.sqrt(
                            n_sent),
                        np.nanmean(curve_, axis=1) * 180 / np.pi + np.nanstd(curve_, axis=1) * 180 / np.pi / np.sqrt(
                            n_sent), color=(0, 0, 0), alpha=.1, zorder=1)
        ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, linewidth=.5, color=(0, 0, 0),
                zorder=0, alpha=1)
        curve_abl = curve_ablated_[idx]  # for each ablated model
        # plot curve_abl only from layer_to_ablate[idx] to the end
        curve_abl = curve_abl[layers_to_ablate[idx]:, :] * 180 / np.pi
        ax.scatter(np.arange(curve_abl.shape[0]) + layers_to_ablate[idx], np.nanmean(curve_abl, axis=1), s=10,
                   color=line_cols[layers_to_ablate[idx], :], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        # do fill between
        ax.fill_between(np.arange(curve_abl.shape[0]) + layers_to_ablate[idx],
                        np.nanmean(curve_abl, axis=1) - np.nanstd(curve_abl, axis=1) / np.sqrt(n_sent),
                        np.nanmean(curve_abl, axis=1) + np.nanstd(curve_abl, axis=1) / np.sqrt(n_sent),
                        color=line_cols[layers_to_ablate[idx], :], alpha=.1, zorder=1)
        ax.plot(np.arange(curve_abl.shape[0]) + layers_to_ablate[idx], np.nanmean(curve_abl, axis=1), linewidth=.5,
                color=line_cols[layers_to_ablate[idx], :], zorder=0, alpha=1)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  #
        #    ax.set_ylim((-15, 5))
        ax.set_ylabel(f'curvature')
        ax.set_xlabel('layer')

    # do the same thing for curvature change
    y_loc = np.linspace(.05, .9, len(layers_to_ablate) + 1)[:-1]
    # flip y_loc
    y_loc = y_loc[::-1]
    for idx, y_ in enumerate(y_loc):
        ax = plt.axes((.55, y_, .4, .2 * pap_ratio))
        ax.scatter(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, s=10,
                   color=(0, 0, 0), zorder=2, edgecolor=(.5, .5, .5), linewidth=.5, alpha=1)
        # do fill between
        ax.fill_between(np.arange(curve_change.shape[0]),
                        np.nanmean(curve_change, axis=1) * 180 / np.pi - np.nanstd(curve_change,
                                                                                   axis=1) * 180 / np.pi / np.sqrt(
                            n_sent), np.nanmean(curve_change, axis=1) * 180 / np.pi + np.nanstd(curve_change,
                                                                                                axis=1) * 180 / np.pi / np.sqrt(
                n_sent), color=(0, 0, 0), alpha=.1, zorder=1)
        ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, linewidth=.5,
                color=(0, 0, 0), zorder=0, alpha=1)
        curve_abl = curve_change_ablated[idx]  # for each ablated model
        # plot curve_abl only from layer_to_ablate[idx] to the end
        layer_forward = layers_to_ablate[idx] - 1
        curve_abl = curve_abl[layer_forward:, :] * 180 / np.pi
        ax.scatter(np.arange(curve_abl.shape[0]) + layer_forward, np.nanmean(curve_abl, axis=1), s=10,
                   color=line_cols[layers_to_ablate[idx], :], zorder=2, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        # do fill between
        ax.fill_between(np.arange(curve_abl.shape[0]) + layer_forward,
                        np.nanmean(curve_abl, axis=1) - np.nanstd(curve_abl, axis=1) / np.sqrt(n_sent),
                        np.nanmean(curve_abl, axis=1) + np.nanstd(curve_abl, axis=1) / np.sqrt(n_sent),
                        color=line_cols[layers_to_ablate[idx], :], alpha=.1, zorder=1)
        ax.plot(np.arange(curve_abl.shape[0]) + layer_forward, np.nanmean(curve_abl, axis=1), linewidth=.5,
                color=line_cols[layers_to_ablate[idx], :], zorder=0, alpha=1)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  #
        ax.set_ylim((-15, 5))
        ax.set_ylabel(f'curvature change')
    # show legend

    # put the title
    ax.set_title(f'{modelname} on {dataset}, effect of training on curvature', fontsize=8)

    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_{dataset}_Attn_proj_ablated.pdf'), transparent=True)
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{modelname}_{dataset}_Attn_proj_ablated.png'), transparent=True)

