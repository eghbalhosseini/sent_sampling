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
from sent_sampling.utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib
import re
import scipy as sp
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
if __name__ == '__main__':
    modelnames = ['roberta-base',  'xlnet-large-cased',  'bert-large-uncased','xlm-mlm-en-2048', 'gpt2-xl', 'albert-xxlarge-v2','ctrl']
    extract_id = [
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    #optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
    #             'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True']
    mdl_name = 'gpt2'
    group = f'{mdl_name}_layers'
    # dataset='coca_spok_filter_punct_10K_sample_1'
    dataset = 'ud_sentencez_token_filter_v3_textNoPeriod'
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
    matplotlib.rcParams['font.size'] = 5
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    num_colors = len(all_layer_curve) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', mdl_name)):
        line_cols = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .75, .45))
    for i in tqdm(range(len(all_layer_curve))):
        curv = all_layer_curve[i]
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
    ax.set_ylim((105, 125))
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel('$average curvature$')
    ax.set_title(f"{group} \n {dataset} \n full sentence")
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_{group}_{dataset}.pdf'), transparent=True)
    fig.show()


    #%%
    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    curve_change = (np.stack(all_layer_curve)[1:-1, :] - np.stack(all_layer_curve)[1, :])
    num_colors = len(all_layer_curve) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', mdl_name)):
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
    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature change$')
    ax.set_title(f"{group} \n {dataset} \n full sentence")
    fig.show()
    fig.savefig(os.path.join(ANALYZE_DIR, f'curvature_change_{group}_{dataset}.pdf'), transparent=True)


    #%%
    tot_surprise = []
    tot_surprise_ave = []
    tot_sent_with_period = []

    sent_text = [x[1] for x in ext_obj.model_group_act[0]['activations']]
    sent_dat_text=[x['text'] for x in ext_obj.data_]
    # remove period in the end if exist from sent_dat_text
    sent_dat_text = [x[:-1] if x[-1] == '.' else x for x in sent_dat_text]
    # for each element in sent_text if exist in sent_dat_text, find its locaiton
    # and add the corresponding surprisal value to tot_surprise
    for i in tqdm(range(len(sent_text))):
        if sent_text[i] in sent_dat_text:
            tot_surprise.append(ext_obj.data_[sent_dat_text.index(sent_text[i])]['surprisal_3'])
            tot_sent_with_period.append(sent_text[i])
        else:
            tot_surprise.append(np.nan*np.ones(2))
            tot_sent_with_period.append(sent_text[i])

    # compute the mean of each element in tot_surprise
    for i in tqdm(range(len(tot_surprise))):
        tot_surprise_ave.append(np.nanmean(tot_surprise[i]))

    fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False)
    num_colors = len(all_layer_curve) + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    if bool(re.findall(r'-untrained', mdl_name)):
        line_cols = line_cols * 0 + (.6)
    ax = plt.axes((.1, .1, .65, .45))
    for i in tqdm(range(len(all_layer_curve))):
        curv = np.asarray(all_layer_curve[i])
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

    ax.set_ylim((-.06, .21))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f"{group} \n {dataset} \n correlation between curvature and surprisal")
    fig.savefig(os.path.join(ANALYZE_DIR, f'correlation_curvature_vs_surprisal_{group}_{dataset}.eps'),transparent=True)

    # clear cuda memory
    torch.cuda.empty_cache()