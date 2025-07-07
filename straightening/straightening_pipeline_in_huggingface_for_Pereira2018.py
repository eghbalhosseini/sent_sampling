import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sent_sampling.utils.data_utils import ANALYZE_DIR
from tqdm import tqdm
import torch
import itertools
import matplotlib
import re
from scipy import stats
import xarray as xr
from sent_sampling.utils.curvature_utils import compute_model_activations,compute_model_curvature,compute_one_sided_statistics

from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer
from transformers import PreTrainedTokenizer
import pickle
from transformers import AutoModel
import pandas as pd

matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
pap_ratio = 8 / 11
if __name__ == '__main__':
    #%%
    #modelnames='facebook/opt-125m'
    Pereira_dat=xr.load_dataarray('/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Pereira2018.nc')
    Pereira_stim=pd.read_csv('/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Pereira2018-stimulus_set.csv')

    modelclass='gpt2'
    modelname='gpt2-xl'
    masked=False

    # get sentences from ext_obj
    sentences_=Pereira_stim.sentence.values
    sentence_id=Pereira_stim.stimulus_id.values
    sentence_passage=Pereira_stim.passage_index.values
    sentence_experiment=Pereira_stim.experiment.values
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    # combine each element of the sentence passage id with sentence_experiment
    sentence_passage_experiment=[str(x)+'_'+str(y) for x,y in zip(sentence_passage,sentence_experiment)]
    model = AutoModel.from_pretrained(modelname)
    model.cuda()
    tokenized_text = [tokenizer.tokenize(x) for x in sentences_]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    print('getting activations for model: {}'.format(modelname))
    all_layers=compute_model_activations(model,indexed_tokens,device='cuda')
    # printe that we are getting curvature
    print('getting curvature for model: {}'.format(modelname))
    curvature_dict=compute_model_curvature(all_layers)
    torch.cuda.empty_cache()
    curvature_dict_sent=curvature_dict
    #%%
    all_layers = []
    # cumulative tokens
    # group sentence by which passage they belong to
    _, idx = np.unique(sentence_passage_experiment, return_index=True)
    unique_sent_pass = [sentence_passage_experiment[i] for i in sorted(idx)]
    all_sentence_commulative = []
    for i in tqdm(unique_sent_pass):
        # find index of setnecne_passage_experiment that are equal to i
        idx = [j for j, x in enumerate(sentence_passage_experiment) if x == i]
        # make sure its sorted
        idx = sorted(idx)
        # find the sentences that are in idx
        sentences = [sentences_[j] for j in idx]
        # now incremeantlly add each sentence to the previous one so there is a list of sentences
        sentences_cumulative = [sentences[0]]
        for j in range(1, len(sentences)):
            sentences_cumulative.append(sentences_cumulative[j - 1] + ' ' + sentences[j])
        all_sentence_commulative.append(sentences_cumulative)
    # make all_sentence_commulative flat
    all_sentence_commulative = list(itertools.chain(*all_sentence_commulative))
    tokenized_text_cumulative = [tokenizer.tokenize(x) for x in all_sentence_commulative]
    # get ids
    indexed_tokens_cumulative = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text_cumulative]
    all_layers_cm=compute_model_activations(model,indexed_tokens_cumulative,device='cuda')
    curvature_dict = compute_model_curvature(all_layers_cm)

    curvature_dict_para=curvature_dict


    #%%
    curvature_dict=curvature_dict_sent
    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    all_layer_curve=curvature_dict['curve']
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
        curv_m,curv_l_std,curv_r_std=compute_one_sided_statistics(curv)
        # multiply curv_m,curv_l_std,and curv_r_std by 180/pi to convert to degrees
        curv_m,curv_l_std,curv_r_std=curv_m * 180 / np.pi,curv_l_std * 180 / np.pi,curv_r_std * 180 / np.pi

        ax.scatter(i, curv_m, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar([i,i], [curv_m,curv_m] , yerr=[curv_l_std,curv_r_std], linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1)*180 / np.pi, color=(0, 0, 0), linewidth=1, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
#    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature$')
    model_name=modelname.replace('/',')')

    ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
    for i, curv in enumerate(curve_change):
        curv_m,curv_l_std,curv_r_std=compute_one_sided_statistics(curv)
        curv_m, curv_l_std, curv_r_std = curv_m * 180 / np.pi, curv_l_std * 180 / np.pi, curv_r_std * 180 / np.pi
        ax.scatter(i, curv_m, s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar([i, i], [curv_m, curv_m], yerr=[curv_l_std, curv_r_std], linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
    # plot a line for the average
    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(0, 0, 0), linewidth=1,
            zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  #
    ax.set_ylim((-15, 5))
    ax.set_ylabel(f'curvature change$')

    fig.show()
    # save figure in ANALYZE_DIR folder
    #fig.savefig(os.path.join(ANALYZE_DIR, f'{model_name}_curvature.pdf'), format='pdf', dpi=200, bbox_inches='tight')
    # select atlas == language in perire dat
    #%%
    Pereira_dat_lang=Pereira_dat.sel(neuroid=(Pereira_dat.atlas=='language').values)
    # sort Pereira_dat_lang by values in sentence_id
    alinments=np.asarray([np.argwhere(Pereira_dat_lang.stimulus_id.values==x) for x in sentence_id]).squeeze()
    Pereira_dat_lang=Pereira_dat_lang.isel(presentation=alinments)

    # drop nans from Pereira_dat_lang
    Pereira_dat_lang=Pereira_dat_lang.dropna('neuroid')

    # find correlation between curvature value and neuroid value for each voxel
    for idx,curv_type in enumerate(['sentence','passage']):
        if curv_type=='sentence':
            all_layer_curve=curvature_dict_sent['curve']
        else:
            all_layer_curve=curvature_dict_para['curve']
        curve_vox_corrs=[]
        curve_ = np.stack(all_layer_curve)
        #curve_change = (curve_[1:, :] - curve_[1, :])
        for curv in tqdm(curve_):
            # compute the correaltion between curv and Pereira_dat_lang
            layer_corr=[]
            for x in Pereira_dat_lang.values.T:
                #layer_corr.append(np.corrcoef(curv,x)[0,1])
                # do pearson correlation between curv and x
                [r,p]=stats.pearsonr(curv,x)
                #if p<.05:
                layer_corr.append(r)
                #else:
                #    layer_corr.append(np.nan)
            curve_vox_corrs.append(np.asarray(layer_corr))
        curve_vox_corrs=np.stack(curve_vox_corrs)
        fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
        ax = plt.axes((.1, .5, .65, .35 * pap_ratio))
        for i, curv in enumerate(curve_vox_corrs):
            ax.scatter(i, np.nanmean(curv) , s=25, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                       linewidth=.5, alpha=1)
            ax.errorbar(i, np.nanmean(curv) , yerr=np.nanstd(curv), linewidth=0, elinewidth=1,
                        color=line_cols[i, :], zorder=0, alpha=1)
        # plot a line for the average
        ax.plot(np.arange(curve_vox_corrs.shape[0]), np.nanmean(curve_vox_corrs, axis=1), color=(0, 0, 0), linewidth=1,
                zorder=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  #
        ax.set_ylim((-.1, .1))
        ax.set_ylabel(f'curvature change$')
        fig.show()
        fig.savefig(os.path.join(ANALYZE_DIR, f'{model_name}_curvature_{curv_type}_correlation_with_Pereira2018_all_ROIs.pdf'), format='pdf', dpi=200, bbox_inches='tight')
        curve_vox_corrs_roi = []
        for curv in tqdm(curve_):
            # compute the correaltion between curv and Pereira_dat_lang
            layer_corr_dict = dict()
            for g, grp in Pereira_dat_lang.groupby('roi'):
                layer_corr = []
                for x in grp.values.T:
                    # layer_corr.append(np.corrcoef(curv,x)[0,1])
                    #do pearson correlation between curv and x
                    [r, p] = stats.pearsonr(curv, x)
                    # if p<.05:
                    layer_corr.append(r)
                layer_corr_dict[g]= np.asarray(layer_corr)
                # else:
                #    layer_corr.append(np.nan)
            curve_vox_corrs_roi.append(layer_corr_dict)
        #
        AA=[[x[key] for x in curve_vox_corrs_roi] for key in curve_vox_corrs_roi[0].keys()]
        # create figure wiht 12 subplots, and plot the elements in AA
        fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
        pap_ratio=8/11
        for i,c in enumerate(AA):
            c=np.stack(c)
            ax = fig.add_subplot(4, 3, i+1)
            for j,c_ in enumerate(c):
                c_m,c_l_std,c_r_std=compute_one_sided_statistics(c_)
                ax.scatter(j, c_m, s=25, color=line_cols[j, :], zorder=2, edgecolor=(0, 0, 0),
                       linewidth=.5, alpha=1)
                ax.errorbar([j,j], [c_m,c_m] , yerr=[c_l_std,c_r_std], linewidth=0, elinewidth=1,
                        color=line_cols[j, :], zorder=0, alpha=1)
            # plot a line for the average
            ax.plot(np.arange(len(c)), np.nanmean(c, axis=1), color=(0, 0, 0), linewidth=1,
                zorder=1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # ax title
            ax.set_title(f'{list(curve_vox_corrs_roi[0].keys())[i]}')
            ax.set_ylim((-.075,.15))
        plt.tight_layout()
        fig.show()
        fig.savefig(
            os.path.join(ANALYZE_DIR, f'{model_name}_curvature_{curv_type}_correlation_with_Pereira2018_individual_ROIs.pdf'),
            format='pdf', dpi=200, bbox_inches='tight')

    #%%
    # group sentences in each layer by their curvature change and find the the level of activation is for those sentances
    # in perereia dat lang
    splits=6
    ranges=np.linspace(0,100,splits)[1:-1]
    range_=np.linspace(0,100,splits)
    # do a sum of adjacent elements in ranges so that it is the mean of the two adjacent elements
    range_=[(range_[i]+range_[i+1])/2 for i in range(len(range_)-1)]

    # zscore neuroid responses across presenteations
    Pereira_dat_lang=Pereira_dat_lang.dropna('neuroid')
    Pereira_dat_lang_norm=(Pereira_dat_lang-Pereira_dat_lang.mean('presentation'))/Pereira_dat_lang.std('presentation')
    for idx,curv_type in enumerate(['sentence','passage']):
        if curv_type=='sentence':
            all_layer_curve=curvature_dict_sent['curve']
        else:
            all_layer_curve=curvature_dict_para['curve']

        curve_vox_corrs=[]
        pereira_layer_group = []
        curve_ = np.stack(all_layer_curve)
        for curv in curve_:
            # divide the curv to 3 groups and find index of sentences in each group
            group_values=np.percentile(curv, ranges)
            #low_threshold = np.percentile(curv, 33)
            #high_threshold = np.percentile(curv, 66)

            # Group the data using np.digitize
            group_inds = np.digitize(curv, list(group_values))
            #group_inds=np.digitize(curv, group_values)
            pereira_goup=[]
            for i in range(len(np.unique(group_inds))):
                idx=np.argwhere(group_inds==i).squeeze()
                # find the mean activation for each voxel in Pereira_dat_lang
                pereira_goup.append(Pereira_dat_lang_norm.isel(presentation=idx).mean('presentation'))
            # combine them into a single array
            pereira_goup=xr.concat(pereira_goup,dim='group')
            pereira_layer_group.append(pereira_goup)
        # now plot the mean activation for each group in each layer
        fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
        for i,pier_corr in enumerate(pereira_layer_group):
            ax = fig.add_subplot(8, int(np.ceil(len(pereira_layer_group)/8)), i+1)
            #curv=curv.groupby('subject').mean('neuroid')
            # plot indivdual subjects as a line
            c_m, c_std = pier_corr.mean('neuroid').values, pier_corr.std('neuroid').values
            ax.scatter(range_, c_m, s=10, color=line_cols[j, :], zorder=2, edgecolor=(0, 0, 0),linewidth=.5, alpha=1)
            ax.errorbar(range_, c_m, yerr=c_std, linewidth=0, elinewidth=1,color=line_cols[j, :], zorder=0, alpha=1)
            # plot a line for the average
            ax.plot(range_, c_m, color=(0, 0, 0), linewidth=1, zorder=1)
            # plot a horizontal line at 0
            ax.axhline(0, color='black', linewidth=1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # ax title
            ax.set_title(f' {i}')
            ax.set_ylim((-.12,.12))
            ax.set_xlim((0,100))
            ax.set_yticks([-.1, 0, .1])
            ax.set_xticks(range_)
            if i==len(pereira_layer_group)-1:
                # make print ranges with no decimal points
                range_=[int(x) for x in range_]
                ax.set_xticks(range_)
                ax.set_xticklabels(range_)
                # make y ticks min, 0 and max of the ylims
                ax.set_xlabel('curvature \n (quantiles)')
                # make yticklabels to be only 1 decimal point
                yticks = ax.get_yticks()
                ax.set_yticklabels([f'{x:.1f}' for x in yticks])
                ax.set_ylabel('norm voxel act')
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        plt.tight_layout()
        fig.show()
        # save figure
        fig.savefig(os.path.join(ANALYZE_DIR, f'{modelname}_curvature_{curv_type}_vs_voxel_activation_Pereira.pdf'), transparent=True)


