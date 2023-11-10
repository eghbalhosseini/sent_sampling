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
from utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib
import re
import scipy as sp
import transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,AutoModelForCausalLM, AutoTokenizer,AutoModel,AutoModelForMaskedLM

from minicons import scorer
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
if __name__ == '__main__':
    #modelnames = 'xlnet-base-cased'
    #modelnames='bigscience/bloom-7b1'
    #modelnames='microsoft/DialoGPT-medium'
    #modelnames='funnel-transformer/small'
    modelnames='facebook/opt-125m'
    masked=False
    dataset='ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences=[x['text'] for x in ext_obj.data_]
    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(modelnames)
    # tokenize sentences
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    # get attention mask
    # feed individual sentences to model
    #model = AutoModelForCausalLM.from_pretrained(modelnames)
    if masked==True:
        model = AutoModelForMaskedLM.from_pretrained(modelnames)
    else:
        model = AutoModelForCausalLM.from_pretrained(modelnames,return_dict_in_generate=True)
    #model = AutoModel.from_pretrained(modelnames)
    # send model to gpu
    model.cuda()

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
    #optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
    #             'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True']
    # empty cuda cache
    torch.cuda.empty_cache()
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


    #%%
    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    #curve_change = (np.stack(all_layer_curve)[1:-1, :] - np.stack(all_layer_curve)[1, :]).transpose()
    curve_change = np.stack(all_layer_curve).transpose()
    num_colors = curve_change.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', modelnames)):
        line_cols = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .75, .45))
    for i in range(len(curve_change)):
        curv = curve_change[i]

        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=15, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=.7)
    # m, b = np.polyfit(tot_surprise_ave, curv, 1)
    # X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
    # plt.plot(X_plot, m*X_plot + b, 'k-',zorder=4)
    # ax.tick_params(
    #                     axis='x',          # changes apply to the x-axis
    #                     which='both',      # both major and minor ticks are affected
    #                     bottom=False,      # ticks along the bottom edge are off
    #                     top=False,         # ticks along the top edge are off
    #                     labelbottom=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
#    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature$')
    model_name=modelnames.replace('/',')')
    ax.set_title(f" {modelnames} \n {dataset}\n Huggingface")
    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{model_name}_{dataset}_masked_{masked}.pdf'), transparent=True)


    #%%
    tot_surprise = []
    tot_surprise_ave = []
    tot_sent_with_period = []


    # remove period in the end if exist from sent_dat_text

    # for each element in sent_text if exist in sent_dat_text, find its locaiton
    # and add the corresponding surprisal value to tot_surprise
    for i in tqdm(range(len(sentences))):
        tot_surprise.append(ext_obj.data_[i]['surprisal_3'])

    # compute the mean of each element in tot_surprise
    for i in tqdm(range(len(tot_surprise))):
        tot_surprise_ave.append(np.nanmean(tot_surprise[i]))

    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    curve= np.stack(all_layer_curve).transpose()
    num_colors = len(curve) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', modelnames)):
        line_cols = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .45))
    for i,curv in tqdm(enumerate(curve)):
        True

        # find if curv or tot_surprise_ave has nan values
        # if so, remove them from both
        nan_idx = np.logical_or(np.isnan(curv), np.isnan(tot_surprise_ave))
        curv = curv[~nan_idx]
        tot_surprise_ave_ = np.array(tot_surprise_ave)[~nan_idx]
        # if tot_surprise_ave contains nan drop it and adijst the curv

        r, p = sp.stats.pearsonr(tot_surprise_ave_, curv)
        # ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        #        transform=ax.transAxes,fontsize=6)
        if p < 1e-2:
            ax.scatter(i, r, s=10, color=line_cols[i, :], zorder=4, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        else:
            ax.scatter(i, r, s=10, color=(1, 1, 1), zorder=4, edgecolor=(0, 0, 0), linewidth=.5)

    # m, b = np.polyfit(tot_surprise_ave, curv, 1)
    # X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
    # plt.plot(X_plot, m*X_plot + b, 'k-',zorder=4)
    # ax.tick_params(
    #                     axis='x',          # changes apply to the x-axis
    #                     which='both',      # both major and minor ticks are affected
    #                     bottom=False,      # ticks along the bottom edge are off
    #                     top=False,         # ticks along the top edge are off
    #                     labelbottom=False)

    ax.set_ylim((-.2, .31))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelnames} \n {dataset} \n correlation between curvature and surprisal")
    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'correlation_curvature_vs_surprisal_{model_name}_huggingface_{dataset}_masked_{masked}.eps'),transparent=True)

    # clear cuda memory
    torch.cuda.empty_cache()

    #%%
    all_layers = []
    nll= []
    # send model to cuda
    model.cuda()
    for i in tqdm(range(len(indexed_tokens))):
        tokens_tensor = torch.tensor([indexed_tokens[i]]).to('cuda')
        target_ids= tokens_tensor.clone().to('cuda')
        with torch.no_grad():
            outputs = model(tokens_tensor,labels=target_ids, output_hidden_states=True)
            hidden_states = outputs['hidden_states']
            neg_log_likelihood = outputs['loss']
            # squeeze the first dimension
            hidden_states = [x.squeeze(0).cpu() for x in hidden_states]

        all_layers.append(hidden_states)
        nll.append(neg_log_likelihood.cpu().numpy())

    torch.cuda.empty_cache()
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


    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    curve = np.stack(all_layer_curve).transpose()
    num_colors = len(curve) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', modelnames)):
        line_cols = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .45))
    for i, curv in tqdm(enumerate(curve)):
        True

        # find if curv or tot_surprise_ave has nan values
        # if so, remove them from both
        nan_idx = np.logical_or(np.isnan(curv), np.isnan(nll))
        curv = curv[~nan_idx]
        tot_nll_ave_ = np.array(nll)[~nan_idx]
        # if tot_surprise_ave contains nan drop it and adijst the curv

        r, p = sp.stats.pearsonr(tot_nll_ave_, curv)
        # ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        #        transform=ax.transAxes,fontsize=6)
        if p < 1e-2:
            ax.scatter(i, r, s=10, color=line_cols[i, :], zorder=4, edgecolor=(0, 0, 0), linewidth=.5, alpha=1)
        else:
            ax.scatter(i, r, s=10, color=(1, 1, 1), zorder=4, edgecolor=(0, 0, 0), linewidth=.5)

    # m, b = np.polyfit(tot_surprise_ave, curv, 1)
    # X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
    # plt.plot(X_plot, m*X_plot + b, 'k-',zorder=4)
    # ax.tick_params(
    #                     axis='x',          # changes apply to the x-axis
    #                     which='both',      # both major and minor ticks are affected
    #                     bottom=False,      # ticks along the bottom edge are off
    #                     top=False,         # ticks along the top edge are off
    #                     labelbottom=False)

    ax.set_ylim((-.2, .31))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelnames} \n {dataset} \n correlation between curvature and NLL")
    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR,
                             f'correlation_curvature_vs_surprisal_{model_name}_huggingface_{dataset}_masked_{masked}.eps'),
                transparent=True)
