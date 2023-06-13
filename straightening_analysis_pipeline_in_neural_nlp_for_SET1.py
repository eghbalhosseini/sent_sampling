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
import xarray as xr
import itertools
import seaborn as sns
from minicons import scorer
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
    #modelname = 'xlnet-base-cased'
    #modelname='bigscience/bloom-7b1'
    #modelname='microsoft/DialoGPT-medium'
    #modelname='funnel-transformer/small'
    #modelname='facebook/opt-125m'

    dataset='ud_sentencez_token_filter_v3_wordFORM'
    # get data
    modelname = 'gpt2-xl'
    group = f'{modelname}_layers'
    # dataset='coca_spok_filter_punct_10K_sample_1'
    #dataset = 'ud_sentencez_token_filter_v3_wordFORM'
    activatiion_type = 'activation'
    average = 'None'
    extractor_id = f'group={group}-dataset={dataset}-{activatiion_type}-bench=None-ave={average}'

    ext_obj=extract_pool[extractor_id]()
    ext_obj.load_dataset()
    ext_obj()

    all_layer_curve = []
    all_layer_curve_all = []
    all_layer_curve_rnd = []
    all_layer_curve_rnd_all = []
    for idk, layer_act in tqdm(enumerate(ext_obj.model_group_act)):
        sent_act_list = layer_act['activations']
        # sent_act=[torch.tensor(x[0], dtype=float, device=optim_obj.device, requires_grad=False) for x in sent_act_list]
        sent_act = [x[0] for x in sent_act_list]
        sent_act = [np.diff(x, axis=0) for x in sent_act]

        sent_act = [normalized(x) for x in sent_act]
        curvature = []
        for idy, vec in (enumerate(sent_act)):
            curve = [np.dot(vec[idx, :], vec[idx + 1, :]) for idx in range(vec.shape[0] - 1)]
            curvature.append(np.arccos(curve))
        all_layer_curve.append([np.mean(x) for x in curvature])
        all_layer_curve_all.append(curvature)

    #%%
    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    curve_ = np.stack(all_layer_curve)
    curve_change = (curve_[1:, :] - curve_[1, :])
    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', modelname)):
        line_cols = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .35*pap_ratio))
    for i,curv in enumerate(curve_):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
#    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature$')
    model_name=modelname.replace('/',')')

    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    for i, curv in enumerate(curve_change):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=np.nanstd(curv) * 180 / np.pi, linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1,
            zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature change$')


    model_name = modelname.replace('/', ')')
    ax.set_title(f" {modelname} \n {dataset}\n neural_nlp")


    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{model_name}_{dataset}_.pdf'), transparent=True)



    #%%
    tot_surprise = []
    tot_surprise_ave = []
    tot_sent_with_period = []

    sent_text = [x[1] for x in ext_obj.model_group_act[0]['activations']]
    if 'wordFORM' in dataset:
        sent_dat_text = [' '.join(x['word_FORM']) for x in ext_obj.data_]
    else:
        sent_dat_text = [x['text'] for x in ext_obj.data_]
        sent_dat_text = [x[:-1] if x[-1] == '.' else x for x in sent_dat_text]
    # remove period in the end if exist from sent_dat_text

    # for each element in sent_text if exist in sent_dat_text, find its locaiton
    # and add the corresponding surprisal value to tot_surprise
    num_nan=0
    for i in tqdm(range(len(sent_text))):
        if sent_text[i] in sent_dat_text:
            tot_surprise.append(ext_obj.data_[sent_dat_text.index(sent_text[i])]['surprisal_3'])
            tot_sent_with_period.append(sent_text[i])

        else:
            tot_surprise.append(np.nan * np.ones(2))
            tot_sent_with_period.append(sent_text[i])
            num_nan+=1

    # compute the mean of each element in tot_surprise
    word_start=0
    for i in tqdm(range(len(tot_surprise))):
        tot_surprise_ave.append(np.nanmean(tot_surprise[i][word_start:]))

    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    curve= np.stack(all_layer_curve)
    num_colors = len(curve) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', modelname)):
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


    ax.set_ylim((-.06, .21))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelname} \n {dataset} \n correlation between curvature and surprisal")
    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'correlation_curvature_vs_surprisal_word_start_{word_start}_{modelname}_neural_nlp_{dataset}.eps'),transparent=True)

    # clear cuda memory
    torch.cuda.empty_cache()

    #%%

    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    for i in range(len(all_layer_curve)):
        curv = all_layer_curve[i]
        # drop nan values from both suprise and curvature
        nan_idx = np.logical_or(np.isnan(curv), np.isnan(tot_surprise_ave))
        # select only non nan values
        curv = np.array(curv)[~nan_idx]
        tot_surprise_ave_ = np.array(tot_surprise_ave)[~nan_idx]
        ax = plt.subplot(10, 5, i + 1)
        # ax.scatter(tot_surprise_ave,curv,s=5,color=(0,0,1),zorder=4,edgecolor=(1,1,1),linewidth=.5,alpha=.5)
        curv_deg=curv * 180 / np.pi
        ax = sns.regplot(x=tot_surprise_ave_, y=curv_deg,
                         scatter_kws={"s": 3, "alpha": .2, "edgecolor": (1, 1, 1), "linewidth": .5},
                         line_kws={"lw": .5, 'color': 'k'})
        r, p = sp.stats.pearsonr(tot_surprise_ave_, curv)
        ax = plt.gca()
        ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes, fontsize=6)
        # m, b = np.polyfit(tot_surprise_ave, curv, 1)
        # X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
        # plt.plot(X_plot, m*X_plot + b, 'k-',zorder=4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  #
        # ax.set_yticks([])
        ax.set_title(f"layer {i + 1}")
        # ax.tick_params(
        #                     axis='x',          # changes apply to the x-axis
        #                     which='both',      # both major and minor ticks are affected
        #                     bottom=False,      # ticks along the bottom edge are off
        #                     top=False,         # ticks along the top edge are off
        #                     labelbottom=False)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        if i == len(curve_change) - 1:
            ax.set_xlabel('surprisal')
            ax.set_ylabel('curvature')

    plt.tight_layout()
    fig.show()
#%%

    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    axes_set=[(.1, .1, .2, .15 * pap_ratio),(.1, .3, .2, .15 * pap_ratio)]
    for ax_id,i in enumerate([0,20]):
            #range(len(all_layer_curve)):
        curv = all_layer_curve[i]
        ax = plt.axes(axes_set[ax_id])
        nan_idx = np.logical_or(np.isnan(curv), np.isnan(tot_surprise_ave))
        # select only non nan values
        curv = np.array(curv)[~nan_idx]
        tot_surprise_ave_ = np.array(tot_surprise_ave)[~nan_idx]
        # ax.scatter(tot_surprise_ave,curv,s=5,color=(0,0,1),zorder=4,edgecolor=(1,1,1),linewidth=.5,alpha=.5)
        curv_deg = curv * 180 / np.pi
        ax = sns.regplot(x=tot_surprise_ave_, y=curv_deg,
                         scatter_kws={"s": 3, "alpha": .2, "edgecolor": (1, 1, 1), "linewidth": .25},
                         line_kws={"lw": .5, 'color': 'k'})
        x_data=tot_surprise_ave_
        y_data=curv_deg
        #ax=sns.scatterplot(x=x_data, y=y_data,s=3,alpha=.2,edgecolor=(1,1,1),linewidth=.5)
        m, b = np.polyfit(x_data, y_data, 1)
        X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        #ax.plot(X_plot, m * X_plot + b, '-')

        #slope, intercept, r, p, sterr = sp.stats.linregress(x=ax.get_lines()[0].get_xdata(),y=ax.get_lines()[0].get_ydata())
        r, p = sp.stats.pearsonr(tot_surprise_ave_, curv)
        ax = plt.gca()
        ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes, fontsize=6)
        #m, b = np.polyfit(tot_surprise_ave, curv, 1)
        #X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
        #ax.plot(X_plot, m*X_plot + b, 'k-',zorder=4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  #
        #  limit xlim and ylims to 99 percentile
        ax.set_xlim([np.percentile(tot_surprise_ave_, 0.05), np.percentile(tot_surprise_ave_, 99.95)])
        ax.set_ylim([np.percentile(curv_deg, 0.05), np.percentile(curv_deg, 99.95)])
        ax.set_title(f"layer {i + 1}")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  #
        if i == len(curve_change) - 1:
            ax.set_xlabel('surprisal')
            ax.set_ylabel('curvature')

    plt.tight_layout()
    #fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_vs_surprisal_word_start{word_start}_select_layers_{modelname}_neural_nlp_{dataset}.eps'),
                transparent=True)
    #%% calculate relationship between curvature values and modellikelihiods
    mincons_model=scorer.IncrementalLMScorer(modelname,device='cuda')
    tokenizer=AutoTokenizer.from_pretrained(modelname)
    mincons_model.prepare_text(sent_text[1])

    from transformers import AutoTokenizer




    # compute the conditional probability of the next word given the previous words
    sent_model_cond_p=[]
    sent_model_surp=[]
    for idx,sentence in tqdm(enumerate(sent_text),total=len(sent_text)):
        True
        words=sentence.split(' ')
        tokenized_sentences=[tokenizer.tokenize(sentence)]
        tokenized_sentences = list(itertools.chain.from_iterable(tokenized_sentences))
        tokenized_sentences = np.array(tokenized_sentences)
        aligned_bug=align_tokens_debug(tokenized_sentences=tokenized_sentences,additional_tokens=[],sentences=[sentence],max_num_words=512,use_special_tokens=False,special_tokens=('ġ',))
        cond_p=mincons_model.compute_stats(mincons_model.prepare_text(sentence))
        modl_surp=mincons_model.token_score(sentence,surprisal=True)
        #assert len(cond_p[0])==len(modl_surp[0])
        sent_model_cond_p.append(cond_p)
        sent_model_surp.append(modl_surp)


    # each element in all_layer_curve_all is a list of curvature values for each layer * words in the sentence
    # for each element in layer curve compute the correlation of each column with the sentence_cond_p
    curv_cond_p=[]
    curve_surp_p=[]
    for i_sent,layer_curve in tqdm(enumerate(all_layer_curve_all)):
        sent_p=sent_model_cond_p[i_sent][0][2:]
        sent_surp= [x[1] for x in sent_model_surp[i_sent][0][3:]]
        rs=[]
        rs_surp=[]
        for idy,curv in enumerate(layer_curve):
            r,p=sp.stats.pearsonr(sent_p,curv)
            if p < 1:
                rs.append(r)
            else:
                rs.append(np.nan)
            r,p=sp.stats.pearsonr(sent_surp,curv)
            if p < 1:
                rs_surp.append(r)
            else:
                rs_surp.append(np.nan)

        curv_cond_p.append(rs)
        curve_surp_p.append(rs_surp)

    curv_cond_p=np.stack(curv_cond_p).transpose()
    curve_surp_p = np.stack(curve_surp_p).transpose()

    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)

    num_colors = len(curv_cond_p) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .65, .45))
    for i, curv in tqdm(enumerate(curv_cond_p)):
        ax.scatter(i, np.nanmean(curv) , s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) , yerr=np.nanstd(curv) , linewidth=0, elinewidth=1,color=line_cols[i, :], zorder=0, alpha=1)
    #ax.set_ylim((-.2, .31))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelname} \n {dataset} \n correlation between curvature and NLL")
    fig.show()

    #%%
    torch.cuda.empty_cache()
    all_layer_diff_mag = []
    all_sentence_activations=all_layers
    for idk, sentence_act in tqdm(enumerate(all_sentence_activations)):
        # each element in sentence act is sentence representation for each layer, and each element is size of (seq_len,hidden_size)
        v_ = [torch.diff(x, axis=0).cpu() for x in sentence_act]
        v_direction = [normalized(x) for x in v_]
        v_length= [torch.norm(x, dim=1).cpu() for x in v_]
        # make sure the lentgh of v_direction vectors is close to 1
        assert np.allclose([torch.norm(x, dim=1).mean().numpy() for x in v_direction],1,atol=1e-2)
        # create an empty list of estimated v_
        v_diff_magnitueds= []
        for idl, v_dir in enumerate(v_direction):
            # multiply each element of v_length[idl][1:] with the direction of v_dir[:-1]
            v_estimate=(v_dir[1:] * v_length[idl][:-1].unsqueeze(1))
            # compute the cosine distance between the estimated v_ and the actual v_
            v_diff=v_estimate-v_[idl][1:]
            # compute the magnitude of the difference
            v_diff_mag=torch.norm(v_diff,dim=1)
            v_diff_magnitueds.append(v_diff_mag)

        all_layer_diff_mag.append(v_diff_magnitueds)


    diff_cond_p=[]
    diff_surp_p=[]
    for i_sent,layer_diff in tqdm(enumerate(all_layer_diff_mag)):
        sent_p=sent_model_cond_p[i_sent][0][2:]
        sent_surp= [x[1] for x in sent_model_surp[i_sent][0][3:]]
        rs=[]
        rs_surp=[]
        for idy,l_diff in enumerate(layer_diff):
            r,p=sp.stats.pearsonr(sent_p,l_diff)
            if p < 1:
                rs.append(r)
            else:
                rs.append(np.nan)
            r,p=sp.stats.pearsonr(sent_surp,l_diff)
            if p < 1:
                rs_surp.append(r)
            else:
                rs_surp.append(np.nan)

        diff_cond_p.append(rs)
        diff_surp_p.append(rs_surp)

    diff_cond_p=np.stack(diff_cond_p).transpose()
    diff_surp_p = np.stack(diff_surp_p).transpose()
    # plot the relationships between the difference in curvature and the surprisal of the word
    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)

    num_colors = len(diff_cond_p) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .1, .65, .45))
    for i, curv in tqdm(enumerate(diff_cond_p)):
        ax.scatter(i, np.nanmean(curv) , s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) , yerr=np.nanstd(curv) , linewidth=0, elinewidth=1,color=line_cols[i, :], zorder=0, alpha=1)
    #ax.set_ylim((-.2, .31))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{modelname} \n {dataset} \n correlation between diff and surprisal")
    fig.show()

    # show diff_cond_p as a heatmap
    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    ax = plt.axes((.1, .1, .65, .45))
    ax.imshow(diff_cond_p, cmap='inferno', aspect='auto')
    fig.show()

    #%% inspect them after running the neural_nlp_2022 pipeline






