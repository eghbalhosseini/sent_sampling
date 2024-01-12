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
from glob import glob
import re
import xarray as xr
from scipy.spatial.distance import pdist, squareform
if __name__ == '__main__':
    modelnames = ['roberta-base',  'xlnet-large-cased',  'bert-large-uncased','xlm-mlm-en-2048', 'gpt2-xl', 'albert-xxlarge-v2','ctrl']
    model_layers = [('roberta-base', 'encoder.layer.1'),
                    ('xlnet-large-cased', 'encoder.layer.23'),
                    ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                    ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                    ('gpt2-xl', 'encoder.h.43'),
                    ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                    ('ctrl', 'h.46')]
    expr='Pereira2018rand'
    model_activations=[]
    for (model_id,model_layer) in  model_layers:
        True
        pattern = os.path.join(
            '/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored',
            f'identifier={model_id},stimuli_identifier={expr}*.pkl')
        sent_set = glob(pattern)
        len(sent_set)
        # seperate sentences into 384sentences and 243sentences
        sent_set_384 = [sent for sent in sent_set if '384sentences' in sent]
        sent_set_243 = [sent for sent in sent_set if '243sentences' in sent]
        model_activation=[]
        for sent_set in [sent_set_384, sent_set_243]:
            a_list = []
            for kk in range(len(sent_set)):
                a = pd.read_pickle(sent_set[kk])
                a = a['data']
                a_list.append(a)
            a_concat = xr.concat(a_list, dim='presentation')
            # find model_layer in a_concat
            a_concat = a_concat.sel(layer=model_layer)
            model_activation.append(a_concat)
        model_activations.append(model_activation)



    prereira_243=[x[1] for x in model_activations]
    prereira_384=[x[0] for x in model_activations]
    # for each model compute the
    # make sure the order of sentences is the ame in pereria_243 and pereria_384
    # sort the sentences in each element of pereira_243 and pereira_384
# sort the sentences in each element of pereira_243 and pereira_384
    preira_243_sorted=[]
    preira_384_sorted=[]
    for i in range(len(prereira_243)):
        preira_243_sorted.append(prereira_243[i].sortby('stimulus_sentence'))
        preira_384_sorted.append(prereira_384[i].sortby('stimulus_sentence'))


    [np.array_equal(x.stimulus_sentence,preira_384_sorted[0].stimulus_sentence) for x in preira_384_sorted]
    model_dist=np.stack([pdist(x, metric='correlation') for x in preira_384_sorted])
    np.mean(pdist(model_dist, metric='correlation'))
    model_dist = np.stack([pdist(x, metric='correlation') for x in preira_243_sorted])
    np.mean(pdist(model_dist, metric='correlation'))

    ds_all= []
    RDM_all = []
    for extract_id in extract_ids:
        ext_obj=extract_pool[extract_id]()
        ext_obj.load_dataset()
        ext_obj()
        optimizer_obj = optim_pool[optim_id[0]]()
        optimizer_obj.load_extractor(ext_obj)
        optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=False,
                                                 save_results=False)
        n_samples = optimizer_obj.N_S
        print(f'{n_samples}\n')
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_S, replace=False))
        d_s_r, RDM_r = optimizer_obj.gpu_object_function_debug(sent_random)
        ds_all.append(d_s_r)
        RDM_all.append(RDM_r)

    # get n_samples from optimizer_obj
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.2, .7, .08, .25))
    ax.scatter(0, ds_all[2], color=np.divide((55, 76, 128), 256), s=50,
               label=f'random= {ds_all[2]:.4f}', edgecolor='k')
    ax.scatter(0, ds_all[1], color=np.divide((188, 80, 144), 255), s=50, label=f'Ds_min={ds_all[1]:.4f}', edgecolor='k')

    ax.scatter(0, ds_all[0], color=np.divide((255, 128, 0), 255), s=50, label=f'Ds_max={ds_all[0]:.4f}', edgecolor='k')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.4, 0.4))
    ax.set_ylim((0.0, 1.2))
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1.1, .2), frameon=True)
    ax.set_ylabel(r'$D_s$')
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    #dataset=optim_results[0]['extractor_name']
    #optim=optim_results[0]['optimizatin_name']
    #ax_title = f'ds,{dataset},{optim}'
    #ax.set_title(ax_title.replace(',', ',\n'),fontsize=8)

    ax=plt.axes((.6, .73, .25, .25))
    RDM_rand_mean=RDM_all[2]
    RDM_max=RDM_all[0]
    RDM_min=RDM_all[1]
    im=ax.imshow(RDM_all[2].cpu(), cmap='viridis',vmax=RDM_all[0].cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_rand_mean.shape[0]):
        for j in range(RDM_rand_mean.shape[1]):
            text = ax.text(j, i, f"{RDM_rand_mean[i, j]:.2f}",
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_title('RDM_rand')
    # set ytick labels to ext_obj.model_spec
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(modelnames,fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(modelnames, fontsize=6,rotation=90)

    ax=plt.axes((.6, .4, .25, .25))
    im=ax.imshow(RDM_max.cpu(), cmap='viridis',vmax=RDM_max.cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_max.shape[0]):
        for j in range(RDM_max.shape[1]):
            text = ax.text(j, i, f'{RDM_max[i, j]:.2f}',
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(modelnames, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(modelnames, fontsize=6, rotation=90)

    ax.set_title('RDM_max')

    ax = plt.axes((.6, .05, .25, .25))
    im = ax.imshow(RDM_min.cpu(), cmap='viridis',vmax=RDM_max.cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_min.shape[0]):
        for j in range(RDM_min.shape[1]):
            text = ax.text(j, i, f'{RDM_min[i, j]:.2f}',
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(modelnames, fontsize=8)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(modelnames, fontsize=6, rotation=90)

    ax.set_title('RDM_min')
    ax = plt.axes((.9, .05, .01, .25))
    plt.colorbar(im, cax=ax)

    rdm_rand_vec=RDM_rand_mean[np.triu_indices(RDM_min.shape[0], k=1)].to('cpu').numpy()
    rdm_max_vec=RDM_max[np.triu_indices(RDM_min.shape[0], k=1)].to('cpu').numpy()
    rdm_min_vec=RDM_min[np.triu_indices(RDM_min.shape[0], k=1)].to('cpu').numpy()
    # plot rdm vectors connecting points from rdom_rand to rdm max to rdm min
    #fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax=plt.axes((.1, .05, .15, .35))

    color_set=[ np.divide((188, 80, 144), 255),np.divide((55, 76, 128), 256),np.divide((255, 128, 0), 255)]

    rdm_vec=np.vstack((rdm_min_vec,rdm_rand_vec,rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1,2,3],rdm_vec[:,i],color='k',alpha=.3,linewidth=.5)
        # plot a scatter with each point color same as color_set
        ax.scatter([1,2,3],rdm_vec[:,i],color=color_set,s=10,marker='o',alpha=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(),vert=True,showfliers=False,showmeans=False,meanprops={'marker':'o','markerfacecolor':'r','markeredgecolor':'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['ds_min','ds_rand','ds_max'],fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim((0,1.4))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 3.25))
    #ax.violinplot([0,1,2],rdm_vec.transpose(),showmeans=True,showextrema=False,showmedians=False)

    fig.show()


    save_path = Path(ANALYZE_DIR)
    (ext_sh,optim_sh)=make_shorthand(extract_ids[0], optim_id[0])
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)

    select_sentneces=[]
    ext_objs=[]
    df_acts=[]
    df_datas=[]
    for extract_id in extract_ids:
        ext_obj = extract_pool[extract_id]()
        ext_obj.load_dataset()
        ext_obj()
        ext_objs.append(ext_obj)
        select_sentences=[]
        select_activations=[]
        for model_act in ext_obj.model_group_act:

            select_sentences.append([model_act['activations'][s][1] for s in range(len(model_act['activations']))])
            select_activations.append([model_act['activations'][s][0] for s in range(len(model_act['activations']))])

        df_data = pd.DataFrame(np.asarray(select_sentences).transpose(), columns=ext_obj.model_spec)
        df_act = pd.DataFrame(np.asarray(select_activations).transpose(), columns=ext_obj.model_spec)
        df_acts.append(df_act)
        df_datas.append(df_data)

        (ext_sh, optim_sh) = make_shorthand(extract_id,optim_id[0])
        ax_title = f'sent,{ext_sh},{optim_sh}_final'
        df_data.to_csv(Path(ANALYZE_DIR, f'{ax_title}.csv'))
    # comine df_datas
    df_data = pd.concat(df_datas, axis=0)
    # check if there an overlap between columns of df_max and df_min
    # get elements from ext_obj.data_ that are selected by sent_max_loc
    sent_max_data=[ext_objs[0].data_[x] for x in range(80)]
    sent_min_data = [ext_objs[1].data_[x] for x in range(80)]
    sent_all_data = [ext_objs[2].data_[x] for x in range(80)]


    lex_names = [x['name'] for x in LEX_PATH_SET]
    sent_max_lex_values=[[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_max_data]
    sent_min_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_min_data]
    sent_all_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_all_data]
    # add num_words to the beginning of each list
    sent_max_num_words=[len(x['word_id']) for x in sent_max_data]
    sent_min_num_words = [len(x['word_id']) for x in sent_min_data]
    sent_all_num_words = [len(x['word_id']) for x in sent_all_data]
    # add sent_max_num_words to the beginning of each sent_max_lex_values
    sent_max_lex_values=np.concatenate([np.asarray(sent_max_num_words).reshape(-1,1),np.asarray(sent_max_lex_values)],axis=1)
    sent_min_lex_values = np.concatenate([np.asarray(sent_min_num_words).reshape(-1, 1), np.asarray(sent_min_lex_values)], axis=1)
    sent_all_lex_values = np.concatenate([np.asarray(sent_all_num_words).reshape(-1, 1), np.asarray(sent_all_lex_values)], axis=1)
    # add 'num_words' to the begginig of lex_names
    lex_names=['num_words']+lex_names
    assert len(lex_names)==sent_max_lex_values.shape[1]
    # create a figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    axes=axes.flatten()
    for i in range(len(lex_names)):
        # plot a histogram for sent_max_lex_values using seaborn.distplot on axes[i]
        seaborn.distplot(sent_all_lex_values[:, i], bins=50, label='Ds_rand', norm_hist=True, hist=False,ax=axes[i],kde_kws={"lw": 3, "color": np.divide([150, 150, 150], 255)})
        seaborn.distplot(sent_max_lex_values[:, i], bins=50, label='Ds_max', norm_hist=True, hist=False,ax=axes[i], kde_kws={"lw": 3, "color": np.divide((255, 128, 0), 255)})
        seaborn.distplot(sent_min_lex_values[:, i], bins=50, label='Ds_min', norm_hist=True, hist=False,ax=axes[i],kde_kws={"lw": 3,   "color": np.divide((188, 80, 144), 255)})
        # put tick in the begining and end of x axis
        #axes[i].set_xticks([np.min(sent_all_lex_values[:, i]), np.max(sent_all_lex_values[:, i])])
        if i == (len(lex_names)-1):
            axes[i].legend(loc='upper right')
        axes[i].set_ylabel(lex_names[i], fontsize=8)
        # remove top and right spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        # turn off yticks values
        axes[i].set_yticks([])

    plt.tight_layout()
    ax_title = f'sent_features,{ext_sh},{optim_sh}'

    # add a suptitle to the figure
    fig.suptitle(ax_title, fontsize=10,y=.99)
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # save figure as pdf and png
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    fig.show()

    # for each key in df_act_min and df_act_max, compute a pca over the rows and check the explained variance ratio
    dims_for_min = []
    dims_for_max = []
    dims_for_rand= []
    for key in tqdm(df_acts[1].keys()):
        a=np.stack(df_acts[1][key].values)
        pca=PCA()
        pca.fit(a)
        dim_for_90_min=np.where(np.cumsum(pca.explained_variance_ratio_)<.85)[0][-1]/len(pca.explained_variance_ratio_)
        dims_for_min.append(dim_for_90_min)
        b=np.stack(df_acts[0][key].values)
        pca=PCA()
        pca.fit(b)
        dim_for_90_max=np.where(np.cumsum(pca.explained_variance_ratio_)<.85)[0][-1]/len(pca.explained_variance_ratio_)
        dims_for_max.append(dim_for_90_max)

        c=np.stack(df_acts[2][key].values)
        pca=PCA()
        pca.fit(c)
        dim_for_90_rand=np.where(np.cumsum(pca.explained_variance_ratio_)<.85)[0][-1]/len(pca.explained_variance_ratio_)
        dims_for_rand.append(dim_for_90_rand)


    # plot the results
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)

    ax = plt.axes((.2, .2, .35, .25))
    # plot a bar graph for dim_for_90_min and one for dim_for_90_max side by side
    ax.bar(np.arange(len(dims_for_min))-.22, dims_for_min, width=.2,linewidth=1,edgecolor='k', label='min')
    ax.bar(np.arange(len(dims_for_max)) + .22, dims_for_max, width=.2,linewidth=1,edgecolor='k', label='max')
    ax.bar(np.arange(len(dims_for_rand)), dims_for_rand, width=.2,linewidth=1,edgecolor='k', label='rand')
    ax.set_xticks(np.arange(len(dims_for_min)) + .2)
    ax.set_xticklabels(modelnames, rotation=90)
    ax.set_ylabel('fraction of dimension needed \n to explain 90% of variance')
    ax.set_title('PCA dimensionality for min and max activations')
    # put legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.show()

    ax_title = f'dimenstionalty_comp_ds_min_ds_max,ds_rand,low_dim={optimizer_obj.low_dim},{extract_ids[0]}'
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)
    #%%
    # check the projection of sampled activations on the PCA of the full activations
    # get n_samples from each element in optim_id
    Ds_modified_sentences = 'ANNSET_DS_MIN_MAX_from_100ev_eh.csv'
    Ds_modified_sentences = Path(ANALYZE_DIR, Ds_modified_sentences)

    Ds_modified_sentences = pd.read_csv(Ds_modified_sentences)

    # find location of DS_MAX sentences in ext_obj.data_
    Ds_max_sentences = Ds_modified_sentences['DS_MAX']
    Ds_max_selected = Ds_modified_sentences['max_include']

    Ds_min_sentences = Ds_modified_sentences['DS_MIN']
    Ds_min_selected = Ds_modified_sentences['min_include']
    Ds_random_sentences = Ds_modified_sentences['DS_RAND']
    Ds_random_selected = Ds_modified_sentences['rand_include']

    # remove index after 102
    Ds_max_sentences = Ds_max_sentences[:100]
    Ds_max_selected = Ds_max_selected[:100]
    Ds_max_sentences_included = Ds_max_sentences[:100]

    Ds_min_sentences = Ds_min_sentences[:100]
    Ds_min_selected = Ds_min_selected[:100]
    Ds_min_sentences_included = Ds_min_sentences[:100]

    Ds_random_sentences = Ds_random_sentences[:100]
    Ds_random_selected = Ds_random_selected[:100]
    Ds_random_sentences_included= Ds_random_sentences[:100]

    # create a flat list of the included sentences
    Ds_sentences_included = [item for sublist in [Ds_max_sentences_included,Ds_min_sentences_included,Ds_min_sentences_included] for item in sublist]

    # load full dataset


    extract_id_full = f'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'

    # load ext_object
    ext_obj_full = extract_pool[extract_id_full]()
    ext_obj_full.load_dataset()
    ext_obj_full()
    optimizer_obj_full = optim_pool[optim_id[0]]()
    optimizer_obj_full.load_extractor(ext_obj_full)

    optimizer_obj.load_extractor(ext_obj)
    proj_list_min=[]
    proj_list_max=[]
    proj_dim_list=[]
    variance_shape_list=[]
    for idx,act_dict in tqdm(enumerate(ext_obj_full.model_group_act)):
        True
        sentences=[x[1] for x in act_dict['activations']]
        act_ = [x[0] for x in act_dict['activations']]
        # find location of Ds_sentences_included in sentences
        locs=[sentences.index(x) for x in Ds_sentences_included]
        # drop elements of act_ that are in locs
        act_ = [x for i,x in enumerate(act_) if i not in locs]

        act_=np.stack(act_)
        pca_full=PCA()
        pca_full.fit(act_)
        proj_dims=[]
        variance_shape=[]
        for df_act in df_acts:
            a=np.stack(df_act[act_dict['model_name']].values)
            # for each colum in a compute the projection on the pca components
            proj=np.matmul(a,pca_full.components_.T)
            # normalize the projection
            pca = PCA()
            pca.fit(proj)
            # find where variance explained in pca is 90%
            variance_shape.append(pca.explained_variance_ratio_)
            dim = np.where(np.cumsum(pca.explained_variance_ratio_) > .9)[0][0]
            proj_dims.append(dim)
            # find the norm of proj values on each columns



        proj_dim_list.append(proj_dims)
        variance_shape_list.append(variance_shape)


        #
        #b=np.stack(df_act_max[ext_obj.model_spec[idx]])
        # for each colum in a compute the projection on the pca components
        #proj=np.matmul(b,pca.components_.T)
        #proj_list_max.append(proj)
    proj_dim_list=np.stack(proj_dim_list)/80

    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)

    ax = plt.axes((.2, .2, .35, .25))
    # plot a bar graph for dim_for_90_min and one for dim_for_90_max side by side
    ax.bar(np.arange(len(proj_dim_list)) - .22, proj_dim_list[:,1], width=.2, linewidth=1, edgecolor='k', label='min')
    ax.bar(np.arange(len(proj_dim_list)) + .22, proj_dim_list[:,0], width=.2, linewidth=1, edgecolor='k', label='max')
    ax.bar(np.arange(len(proj_dim_list)), proj_dim_list[:,2], width=.2, linewidth=1, edgecolor='k', label='rand')
    ax.set_xticks(np.arange(len(proj_dim_list)) + .2)
    ax.set_xticklabels(modelnames, rotation=90)
    ax.set_ylabel('fraction of dimension needed \n to explain 90% of variance')
    ax.set_title('PCA dimensionality for min and max activations')
    # put legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.show()

    # creat3 7 suplots and in each plot the variance_shape element
    fig, axs = plt.subplots(7, 1, figsize=(8, 11), dpi=300, frameon=False)
    for idx,ax in enumerate(axs):
        ax.plot(np.arange(len(variance_shape_list[idx][0]))[:10], np.cumsum(variance_shape_list[idx][0][:10]), label='max')
        ax.plot(np.arange(len(variance_shape_list[idx][1]))[:10], np.cumsum(variance_shape_list[idx][1][:10]), label='min')
        ax.plot(np.arange(len(variance_shape_list[idx][2]))[:10], np.cumsum(variance_shape_list[idx][2][:10]), label='rand')
        ax.set_title(modelnames[idx])
        ax.set_ylabel('variance explained')
        ax.set_xlabel('PCA component')
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.show()
    # fig=plt.figure(figsize=(11,8),dpi=300,frameon=False)
    # # create a grid of 10 columns and 7 rows
    # gs=GridSpec(7,10)
    # for idx,proj in enumerate(proj_dim_list):
    #     # in each row plot a histogram of the projection values for each column in proj
    #     for idy in range(10):
    #         # select the column in proj
    #         a=proj[:,idy]
    #         # sort the values in a and plot a bar graph
    #         a=np.sort(a)
    #         ax=fig.add_subplot(gs[idx,idy])
    #         ax.scatter(np.arange(len(a)),a,color='b',s=1)
    #         # plot a line at 0
    #         #ax.axhline(0,color='k')
    #         b=proj_list_max[idx][:,idy]
    #         b=np.sort(b)
    #         ax.scatter(np.arange(len(b)),b,color='r',s=1)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.spines["top"].set_visible(False)
    #         ax.spines["right"].set_visible(False)
    #         ax.spines["bottom"].set_visible(False)
    #         ax.spines["left"].set_visible(False)
    #         if idy==0:
    #             ax.set_ylabel(modelnames[idx],fontsize=8,rotation=0,horizontalalignment='right',verticalalignment='center',labelpad=10)
    #         if idx==0:
    #             ax.set_title(f'PC#{idy+1}',fontsize=8)
    # ax_title = f'proj_on_pca_ds_min,low_dim={optimizer_obj.low_dim},{dataset}'
    # save_path = Path(ANALYZE_DIR)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
    #             facecolor='auto',edgecolor='auto', backend=None)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    # fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
    #             facecolor='auto',edgecolor='auto', backend=None)
    #
    #
    #     #act = torch.tensor(act_, dtype=float, device=optimizer_obj.device, requires_grad=False)
    # compute the DS scores for sentences that are selected
    Ds_rand_file_selected='sent,G=best_performing_pereira_1-D=ud_sentencez_token_filter_v3_minus_ev_sentences_len_7_14_textNoPeriod-activation-B=None-AVE=False,coordinate_ascent_eh-O=D_rand-Nit=500-Ns=100-Nin=1-LD=False-V=0.9-T=sklearn-GPU=True_eh_edit.csv'
    Ds_min_file_selected='sent,G=best_performing_pereira_1-D=ud_sentencez_token_filter_v3_minus_ev_sentences_len_7_14_textNoPeriod-activation-B=None-AVE=False,coordinate_ascent_eh-O=2-D_s-Nit=500-Ns=100-Nin=1-LD=False-V=0.9-T=sklearn-GPU=True_eh_edit.csv'
    Ds_max_file_selected='sent,G=best_performing_pereira_1-D=ud_sentencez_token_filter_v3_minus_ev_sentences_len_7_14_textNoPeriod-activation-B=None-AVE=False,coordinate_ascent_eh-O=D_s-Nit=500-Ns=100-Nin=1-LD=False-V=0.9-T=sklearn-GPU=True_ev_edit.csv'
    Ds_rand_file_selected=Path(ANALYZE_DIR,Ds_rand_file_selected)
    Ds_max_file_selected=Path(ANALYZE_DIR,Ds_max_file_selected)
    Ds_min_file_selected=Path(ANALYZE_DIR,Ds_min_file_selected)

    # read the Ds_rand_file_selected csv
    df_rand_selected=pd.read_csv(Ds_rand_file_selected)
    df_max_selected=pd.read_csv(Ds_max_file_selected)
    df_min_selected=pd.read_csv(Ds_min_file_selected)
    # select the sentences that are in the included in df_rand_selected
    df_rand_selected=df_rand_selected[df_rand_selected['include']==1]
    df_max_selected=df_max_selected[df_max_selected['Include']==1]
    df_min_selected=df_min_selected[df_min_selected['Include']==1]

    # find the location of sentences in
    S_id_rand_selected=[int(x) for x in list(df_rand_selected['Unnamed: 0'])]
    S_id_max_selected=[int(x) for x in list(df_max_selected['Unnamed: 0'])]
    S_id_min_selected=[int(x) for x in list(df_min_selected['Unnamed: 0'])]
    # sample S_id_rand_selected and s_id_min_selected with the size of S_id_max_selected
    S_id_rand_selected=np.random.choice(S_id_rand_selected,size=len(S_id_max_selected),replace=False)
    S_id_min_selected=np.random.choice(S_id_min_selected,size=len(S_id_max_selected),replace=False)
    # find sentences baased on S_id_max_selected in model_activations

    #assert len(S_id_rand_selected)==len(S_id_max_selected)==len(S_id_min_selected)==80
    assert len(S_id_rand_selected) == len(S_id_max_selected) == len(S_id_min_selected) == 66
    DS_max_selected, RDM_max_selected = optimizer_obj.gpu_object_function_debug(S_id_max_selected)
    DS_min_selected, RDM_min_selected = optimizer_obj.gpu_object_function_debug(S_id_min_selected)
    DS_rand_selected, RDM_rand_selected = optimizer_obj.gpu_object_function_debug(S_id_rand_selected)
    # find sentences that corresponds
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.2, .7, .08, .25))
    ax.scatter(0, DS_rand_selected, color=np.divide((55, 76, 128), 256), s=50,label=f'random= {DS_rand_selected:.4f}', edgecolor='k')
    ax.scatter(0, DS_min_selected, color=np.divide((188, 80, 144), 255), s=50, label=f'Ds_min={DS_min_selected:.4f}', edgecolor='k')
    ax.scatter(0, DS_max_selected, color=np.divide((255, 128, 0), 255), s=50, label=f'Ds_max={DS_max_selected:.4f}', edgecolor='k')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.4, 0.4))
    ax.set_ylim((0.0, 1.2))
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1.1, .2), frameon=True)
    ax.set_ylabel(r'$D_s$')
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    dataset = optim_results[0]['extractor_name']
    optim = optim_results[0]['optimizatin_name']
    ax_title = f'ds for selected sentences,{dataset},{optim}'
    ax.set_title(ax_title.replace(',', ',\n'), fontsize=8)

    ax = plt.axes((.6, .73, .25, .25))
    im = ax.imshow(RDM_rand_selected.cpu(), cmap='viridis', vmax=RDM_max_selected.cpu().numpy().max())
    # add values to image plot

    for i in range(RDM_rand_selected.shape[0]):
        for j in range(RDM_rand_selected.shape[1]):
            text = ax.text(j, i, f"{RDM_rand_selected[i, j]:.2f}",
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_title('RDM_rand_selected')
    # set ytick labels to ext_obj.model_spec
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax = plt.axes((.6, .4, .25, .25))
    im = ax.imshow(RDM_max_selected.cpu(), cmap='viridis', vmax=RDM_max_selected.cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_max_selected.shape[0]):
        for j in range(RDM_max_selected.shape[1]):
            text = ax.text(j, i, f'{RDM_max_selected[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax.set_title('RDM_max')

    ax = plt.axes((.6, .05, .25, .25))
    im = ax.imshow(RDM_min_selected.cpu(), cmap='viridis', vmax=RDM_max_selected.cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_min_selected.shape[0]):
        for j in range(RDM_min_selected.shape[1]):
            text = ax.text(j, i, f'{RDM_min_selected[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=8)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax.set_title('RDM_min')
    ax = plt.axes((.9, .05, .01, .25))
    plt.colorbar(im, cax=ax)

    rdm_rand_vec = RDM_rand_selected[np.triu_indices(RDM_min_selected.shape[0], k=1)].to('cpu').numpy()
    rdm_max_vec = RDM_max_selected[np.triu_indices(RDM_min_selected.shape[0], k=1)].to('cpu').numpy()
    rdm_min_vec = RDM_min_selected[np.triu_indices(RDM_min_selected.shape[0], k=1)].to('cpu').numpy()
    # plot rdm vectors connecting points from rdom_rand to rdm max to rdm min
    # fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.1, .05, .15, .35))

    color_set = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255)]

    rdm_vec = np.vstack((rdm_min_vec, rdm_rand_vec, rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2, 3], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2, 3], rdm_vec[:, i], color=color_set, s=10, marker='o', alpha=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['ds_min', 'ds_rand', 'ds_max'], fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim((0, 1.4))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 3.25))
    # ax.violinplot([0,1,2],rdm_vec.transpose(),showmeans=True,showextrema=False,showmedians=False)
    ax_title = f'ds,{dataset},{optim}'
    fig.show()
    (ext_sh, optim_sh) = make_shorthand(ext, optim)
    ax_title = f'sent_selected,{ext_sh},{optim_sh}'
    # add a suptitle to the figure
    fig.suptitle(ax_title, fontsize=10, y=.99)
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')

    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',edgecolor='auto', backend=None)

