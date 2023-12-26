import os
import numpy as np
import sys
from pathlib import Path
import getpass
if getpass.getuser() == 'eghbalhosseini':
    SAMPLING_PARENT = '/Users/eghbalhosseini/MyCodes/sent_sampling'
    SAMPLING_DATA = '/Users/eghbalhosseini/MyCodes//fmri_DNN/ds_parametric/'

elif getpass.getuser() == 'ehoseini':
    SAMPLING_PARENT = '/om/user/ehoseini/sent_sampling'
    SAMPLING_DATA = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'



sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
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
if __name__ == '__main__':
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id='coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    # read the excel that contains the selected sentences
    # %%  RUN SANITY CHECKS
    ds_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN/ds_parametric/ANNSET_DS_MIN_MAX_from_100ev_eh_FINAL.csv')
    # read also the actuall experiment stimuli
    stim_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN//ds_parametric/fMRI_final/stimuli_order_ds_parametric.csv',
                           delimiter='\t')
    # find unique conditions
    unique_cond = np.unique(stim_csv.Condition)
    # for each unique_cond find sentence transcript
    unique_cond_transcript = [stim_csv.Stim_transcript[stim_csv.Condition == x].values for x in unique_cond]
    # remove duplicate sentences in unique_cond_transcript
    unique_cond_transcript = [list(np.unique(x)) for x in unique_cond_transcript]
    ds_min_list = unique_cond_transcript[1]
    ds_max_list = unique_cond_transcript[0]
    ds_rand_list = unique_cond_transcript[2]
    # extract the ds_min sentence that are in min_included column
    ds_min_ = ds_csv.DS_MIN_edited[(ds_csv['min_include'] == 1)]
    ds_max_ = ds_csv.DS_MAX_edited[(ds_csv['max_include'] == 1)]
    ds_rand_ = ds_csv.DS_RAND_edited[(ds_csv['rand_include'] == 1)]
    # check if ds_min_ and ds_min_list have the same set of sentences regardless of the order
    assert len([ds_min_list.index(x) for x in ds_min_]) == len(ds_min_)
    assert len([ds_max_list.index(x) for x in ds_max_]) == len(ds_max_)
    assert len([ds_rand_list.index(x) for x in ds_rand_]) == len(ds_rand_)
    # %% MORE SANITY CHECKS FOR THE ACTIVATIONS
    # get the
    ds_min_sent = ds_csv.DS_MIN[(ds_csv['min_include'] == 1)]
    ds_max_sent = ds_csv.DS_MAX[(ds_csv['max_include'] == 1)]
    ds_rand_sent = ds_csv.DS_RAND[(ds_csv['rand_include'] == 1)]
    # laod the extractor
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    ext_obj()
    # find location of sentences in ext_obj.model_group_act
    ds_min_list = []
    ds_max_list = []
    ds_rand_list = []
    for idx, act_dict in enumerate(ext_obj.model_group_act):
        True
        sentences = [x[1] for x in act_dict['activations']]
        # find the location of ds_min_sent in sentences
        ds_min_loc = [sentences.index(x) for x in ds_min_sent]
        ds_max_loc = [sentences.index(x) for x in ds_max_sent]
        ds_rand_loc = [sentences.index(x) for x in ds_rand_sent]
        ds_min_list.append(ds_min_loc)
        ds_max_list.append(ds_max_loc)
        ds_rand_list.append(ds_rand_loc)

    ds_min_list = np.asarray(ds_min_list).transpose()
    ds_max_list = np.asarray(ds_max_list).transpose()
    ds_rand_list = np.asarray(ds_rand_list).transpose()
    # make sure the row are the same in ds_min_list
    assert np.all([np.all(x == x[0]) for x in ds_min_list])
    assert np.all([np.all(x == x[0]) for x in ds_max_list])
    assert np.all([np.all(x == x[0]) for x in ds_rand_list])
    ds_min_loc = ds_min_list[:, 0]
    ds_max_loc = ds_max_list[:, 0]
    ds_rand_loc = ds_rand_list[:, 0]
    model_names = [x['model_name'] for x in ext_obj.model_group_act]
    #%%
    optim_obj = optim_pool[optim_id]()
    optim_obj.load_extractor(ext_obj)
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=False,
                                             save_results=False)

    d_s_min, RDM_min = optim_obj.gpu_object_function_debug(ds_min_loc)
    d_s_rand, RDM_rand = optim_obj.gpu_object_function_debug(ds_rand_loc)
    d_s_max, RDM_max = optim_obj.gpu_object_function_debug(ds_max_loc)
    RDM_min=RDM_min.cpu()
    RDM_rand=RDM_rand.cpu()
    RDM_max = RDM_max.cpu()
    grays = (.8, .8, .8, .5)
    #colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255),
    #          np.divide((55, 76, 128), 256)]
    colors = [np.divide((0, 157, 255), 255), np.divide((128, 128, 128), 256),np.divide((255, 98, 0), 255)]

    # get obj= from optim_id
    obj_id = ['Ds_max', '2-Ds_max']
    # get n_samples from each element in optim_id
    low_resolution=False

    ds_all= []
    RDM_all = []

#%%
    # get n_samples from optimizer_obj
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio=8/11
    ax = plt.axes((.2, .6, .08, .25*pap_ratio))
    ax.scatter(0, d_s_rand, color=colors[1], s=50,
               label=f'random= {d_s_min:.4f}', edgecolor='k')
    ax.scatter(0, d_s_min, color=colors[0], s=50, label=f'Ds_min={d_s_rand:.4f}', edgecolor='k')

    ax.scatter(0, d_s_max, color=colors[2], s=50, label=f'Ds_max={d_s_max:.4f}', edgecolor='k')

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


    ax=plt.axes((.6, .73, .25, .25*pap_ratio))
    im=ax.imshow(RDM_rand, cmap='viridis',vmax=RDM_max.numpy().max())
    # add values to image plot
    for i in range(RDM_rand.shape[0]):
        for j in range(RDM_rand.shape[1]):
            text = ax.text(j, i, f"{RDM_rand[i, j]:.2f}",
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_title('RDM_rand')
    # set ytick labels to ext_obj.model_spec
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(model_names,fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(model_names, fontsize=6,rotation=90)

    ax=plt.axes((.6, .4, .25, .25*pap_ratio))
    im=ax.imshow(RDM_max, cmap='viridis',vmax=RDM_max.numpy().max())
    # add values to image plot
    for i in range(RDM_max.shape[0]):
        for j in range(RDM_max.shape[1]):
            text = ax.text(j, i, f'{RDM_max[i, j]:.2f}',
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(model_names, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(model_names, fontsize=6, rotation=90)

    ax.set_title('RDM_max')
    np.fill_diagonal(RDM_min.numpy(),np.nan)
    ax = plt.axes((.6, .05, .25, .25*pap_ratio))
    im = ax.imshow(RDM_min, cmap='viridis',vmax=RDM_max.numpy().max())
    # add values to image plot
    for i in range(RDM_min.shape[0]):
        for j in range(RDM_min.shape[1]):
            text = ax.text(j, i, f'{RDM_min[i, j]:.2f}',
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(model_names, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(model_names, fontsize=6, rotation=90)

    ax.set_title('RDM_min')
    ax = plt.axes((.9, .05, .01, .25*pap_ratio))
    plt.colorbar(im, cax=ax)

    rdm_rand_vec=RDM_rand[np.triu_indices(RDM_min.shape[0], k=1)].numpy()
    rdm_max_vec=RDM_max[np.triu_indices(RDM_min.shape[0], k=1)].numpy()
    rdm_min_vec=RDM_min[np.triu_indices(RDM_min.shape[0], k=1)].numpy()
    # plot rdm vectors connecting points from rdom_rand to rdm max to rdm min
    #fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax=plt.axes((.1, .05, .15, .3*pap_ratio))

    rdm_vec=np.vstack((rdm_min_vec,rdm_rand_vec,rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1,2,3],rdm_vec[:,i],color='k',alpha=.3,linewidth=.5)
        # plot a scatter with each point color same as color_set
        ax.scatter([1,2,3],rdm_vec[:,i],color=colors,s=10,marker='o',alpha=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(),vert=True,showfliers=False,showmeans=False,meanprops={'marker':'o','markerfacecolor':'r','markeredgecolor':'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['ds_min','ds_rand','ds_max'],fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim((0,1.3))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 3.25))
    #ax.violinplot([0,1,2],rdm_vec.transpose(),showmeans=True,showextrema=False,showmedians=False)

    fig.show()


    save_path = Path(ANALYZE_DIR)
    (ext_sh,optim_sh)=make_shorthand(extract_id, optim_id)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_FINAL.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_FINAL.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)
    #%%
    data_text=[x['text'] for x in ext_obj.data_]
    data_textNoPeriod=[]
    for x in data_text:
        if '.' in x[-1] :
            data_textNoPeriod.append(x[:-1])
        else:
            data_textNoPeriod.append(x)

    ds_min_loc_in_dat = [data_textNoPeriod.index(x) for x in ds_min_sent]
    ds_max_loc_in_dat = [data_textNoPeriod.index(x) for x in ds_max_sent]
    ds_rand_loc_in_dat = [data_textNoPeriod.index(x) for x in ds_rand_sent]

    sent_max_data=[ext_obj.data_[x] for x in ds_max_loc_in_dat]
    sent_min_data = [ext_obj.data_[x] for x in ds_min_loc_in_dat]
    sent_rand_data = [ext_obj.data_[x] for x in ds_rand_loc_in_dat]
    sent_all_data=ext_obj.data_


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
        seaborn.distplot(sent_all_lex_values[:, i], bins=50, label='Ds_all', norm_hist=True, hist=False,ax=axes[i],kde_kws={"lw": 3, "color": np.divide([150, 150, 150], 255)})
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
    fig.show()
    ax_title = f'sent_features,{ext_sh},{optim_sh}'

    # add a suptitle to the figure
    fig.suptitle(ax_title, fontsize=10,y=.99)
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL.png')
    # save figure as pdf and png
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    fig.show()

    #%%
    # plot model RDMS
    # for each matrix in optimizer_obj.XY_corr_list select rows and colums based on a list S
    # and plot the resulting matrix
    X_Max = []
    S_id = ds_max_loc
    for XY_corr in optim_obj.XY_corr_list:
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_Max.append(X_sample)

    X_Min = []
    S_id = ds_min_loc
    for XY_corr in optim_obj.XY_corr_list:
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_Min.append(X_sample)

    X_rand = []
    S_id = ds_rand_loc
    for XY_corr in optim_obj.XY_corr_list:
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_rand.append(X_sample)
    # create a figure with 7 rows and 3 columns and plot x_samples in each row

    fig = plt.figure(figsize=(11, 8))
    for i in range(len(X_Max)):
        ax = plt.subplot(3, 7, i + 1 + 7)
        im = ax.imshow(X_Max[i], cmap='viridis', vmax=X_Max[i].max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_max')
        # turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(X_Min)):
        ax = plt.subplot(3, 7, i + 1 + 14)
        im = ax.imshow(X_Min[i], cmap='viridis', vmax=X_Min[i].max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_min')
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(X_rand)):
        ax = plt.subplot(3, 7, i + 1)
        im = ax.imshow(X_rand[i], cmap='viridis', vmax=X_rand[i].max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_rand')
        ax.set_xticks([])
        ax.set_yticks([])

    # ax = plt.axes((.95, .05, .01, .25))
    # plt.colorbar(im, cax=ax)

    fig.show()
    ax_title = f'sent_rdms,{ext_sh},{optim_sh}'
    # save the figure
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    #%%
    # plot average distances between setnences for each model
    x_max_=[]
    for a in X_Max:
        # get the upper diagonal part of a
        a_upper=a[np.triu_indices(a.shape[0],k=1)]
        x_max_.append(a_upper.squeeze())

    x_min_=[]
    for a in X_Min:
        # get the upper diagonal part of a
        a_upper=a[np.triu_indices(a.shape[0],k=1)]
        x_min_.append(a_upper.squeeze())

    x_rand_=[]
    for a in X_rand:
        # get the upper diagonal part of a
        a_upper=a[np.triu_indices(a.shape[0],k=1)]
        x_rand_.append(a_upper.squeeze())
    #
    # save a dictionary of x_min, x_rand and x_max
    model_names = [x['model_name'] for x in ext_obj.model_group_act]
    similirity_dict={'x_min':x_min_,'x_rand':x_rand_,'x_max':x_max_}
    similiary_path=Path(ANALYZE_DIR,'similarity_dict_DsParametric.pkl')
    save_obj(similirity_dict,similiary_path.__str__())
    fig = plt.figure(figsize=(11, 8))
    for i in tqdm(range(len(X_Max))):
        ax = plt.subplot(2, 4, i + 1)
        # create a df with 2 columns, [x_min_[i],x_rand_[i],x_max_[i]]] and a second column with 'min','rand','max'
        df=pd.DataFrame(2-np.vstack((x_min_[i],x_rand_[i],x_max_[i])).transpose(),columns=['min','rand','max'])
        # change the colors to match the colors in the previous plot
        # melt the df
        df=pd.melt(df)
        # plot a swarm plot of df
        seaborn.violinplot(x='variable',y='value',data=df,ax=ax,palette=colors,scale='width')
        #seaborn.swarmplot(x="variable", y="value", data=df,ax=ax,palette=colors)
        ax.set_title(f'{model_names[i]}', fontsize=8)
        ax.set_ylabel('Sentence alignment', fontsize=8)
        ax.set_xlabel('')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['Ds_min','Ds_rand','Ds_max'],fontsize=8,rotation=90)
        # turn off spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # turn off ticks
        #ax.set_xticks([])
        #ax.set_yticks([])
    plt.tight_layout()
    fig.show()
    # create a figure title
    ax_title = f'sent_alignment,{ext_sh},{optim_sh}'
    # save the figure
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    #%%
