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
import pandas as pd
from pathlib import Path
import torch
from scipy.spatial.distance import pdist, squareform
# check if gpu is available
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import umap
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.colors as mcolors
from cuml.manifold import TSNE
import cuml
from joint_mds import JointMDS
import torch
from scipy import stats




if __name__ == '__main__':
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
                'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']
    #
    # read the excel that contains the selected sentences
    #%%  RUN SANITY CHECKS
    ds_csv=pd.read_csv('/om2/user/ehoseini/fmri_DNN/ds_parametric/ANNSET_DS_MIN_MAX_from_100ev_eh_FINAL.csv')
    # read also the actuall experiment stimuli
    stim_csv=pd.read_csv('/om2/user/ehoseini/fmri_DNN//ds_parametric/fMRI_final/stimuli_order_ds_parametric.csv',delimiter='\t')
    # find unique conditions
    unique_cond=np.unique(stim_csv.Condition)
    # for each unique_cond find sentence transcript
    unique_cond_transcript=[stim_csv.Stim_transcript[stim_csv.Condition==x].values for x in unique_cond]
    # remove duplicate sentences in unique_cond_transcript
    unique_cond_transcript=[list(np.unique(x)) for x in unique_cond_transcript]
    ds_min_list=unique_cond_transcript[1]
    ds_max_list=unique_cond_transcript[0]
    ds_rand_list=unique_cond_transcript[2]
    # extract the ds_min sentence that are in min_included column
    ds_min_=ds_csv.DS_MIN_edited[(ds_csv['min_include']==1)]
    ds_max_=ds_csv.DS_MAX_edited[(ds_csv['max_include']==1)]
    ds_rand_=ds_csv.DS_RAND_edited[(ds_csv['rand_include']==1)]
    # check if ds_min_ and ds_min_list have the same set of sentences regardless of the order
    assert len([ds_min_list.index(x) for x in ds_min_])== len(ds_min_)
    assert len([ds_max_list.index(x) for x in ds_max_])== len(ds_max_)
    assert len([ds_rand_list.index(x) for x in ds_rand_])== len(ds_rand_)
    #%% MORE SANITY CHECKS FOR THE ACTIVATIONS
    # get the
    ds_min_sent = ds_csv.DS_MIN[(ds_csv['min_include'] == 1)]
    ds_max_sent = ds_csv.DS_MAX[(ds_csv['max_include'] == 1)]
    ds_rand_sent = ds_csv.DS_RAND[(ds_csv['rand_include'] == 1)]
    # laod the extractor
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    ext_obj()
    # find location of sentences in ext_obj.model_group_act
    ds_min_list=[]
    ds_max_list=[]
    ds_rand_list=[]
    for idx,act_dict in enumerate(ext_obj.model_group_act):
        True
        sentences=[x[1] for x in act_dict['activations']]
        # find the location of ds_min_sent in sentences
        ds_min_loc=[sentences.index(x) for x in ds_min_sent]
        ds_max_loc=[sentences.index(x) for x in ds_max_sent]
        ds_rand_loc=[sentences.index(x) for x in ds_rand_sent]
        ds_min_list.append(ds_min_loc)
        ds_max_list.append(ds_max_loc)
        ds_rand_list.append(ds_rand_loc)

    ds_min_list=np.asarray(ds_min_list).transpose()
    ds_max_list=np.asarray(ds_max_list).transpose()
    ds_rand_list=np.asarray(ds_rand_list).transpose()
    # make sure the row are the same in ds_min_list
    assert np.all([np.all(x==x[0]) for x in ds_min_list])
    assert np.all([np.all(x==x[0]) for x in ds_max_list])
    assert np.all([np.all(x==x[0]) for x in ds_rand_list])
    ds_min_loc=ds_min_list[:,0]
    ds_max_loc=ds_max_list[:,0]
    ds_rand_loc=ds_rand_list[:,0]
    model_names = [x['model_name'] for x in ext_obj.model_group_act]
    #%% compute the pca
    pca=PCA(n_components=0.95)
    model_pca_loads=[]
    for idx, act_dict in tqdm(enumerate(ext_obj.model_group_act)):
        # backward compatibility
        act_ = np.asarray([x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']])

        # find rows corresponds to ds_min_loc and put the in act_min
        act_min=act_[ds_min_loc,:]
        act_max=act_[ds_max_loc,:]
        act_rand=act_[ds_rand_loc,:]
        # remove act_min, act_max, act_rand from act_
        act_=np.delete(act_,[ds_min_loc,ds_max_loc,ds_rand_loc],axis=0)
        pca.fit(act_)
        act_pca=pca.transform(act_)
        act_min_pca=pca.transform(act_min)
        act_max_pca=pca.transform(act_max)
        act_rand_pca=pca.transform(act_rand)
        model_pca_loads.append({'act':act_pca,'act_min':act_min_pca,'act_max':act_max_pca,'act_rand':act_rand_pca,'exp_variance':pca.explained_variance_ratio_})

    # create a plot with the number of models
    # get model names from ext_obj.model_group_act

    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)
    #colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255),
    #          np.divide((55, 76, 128), 256)]
    colors = [np.divide((0, 157, 255), 255), np.divide((255, 98, 0), 255),
              np.divide((55, 76, 128), 256)]

    for idx, model_loads in tqdm(enumerate(model_pca_loads)):
        True
        ax = plt.subplot(2, 4, idx + 1)
        ax.set_box_aspect(1)
        ax.set_title(model_names[idx])
        l_all = model_loads['act'][:, :2]
        # ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
        ax.scatter(l_all[:, 0], l_all[:, 1], s=.5, c=grays)
        # ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
        l_min = model_loads['act_min'][:, :2]
        ax.scatter(l_min[:, 0], l_min[:, 1], s=.5, c=colors[0])
        l_max = model_loads['act_max'][:, :2]
        ax.scatter(l_max[:, 0], l_max[:, 1], s=.5, c=colors[1])
        right_side = ax.spines["right"]
        right_side.set_visible(False)
        right_side = ax.spines["top"]
        right_side.set_visible(False)
        right_side = ax.spines["left"]
        right_side.set_visible(False)
        right_side = ax.spines["bottom"]
        right_side.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(ax.get_xlim(), [0, 0], '-', zorder=2, linewidth=1, color=(.5, .5, .5))
        ax.plot([0, 0], ax.get_ylim(), '-', zorder=2, linewidth=1, color=(.5, .5, .5))
        exp_varaince=100*sum(model_loads['exp_variance'][:2])
        ax.set_title(f"{ext_obj.model_spec[idx]}\n{exp_varaince:.2f}%", rotation=0,
                      fontsize=8)
    fig.show()

    save_loc = Path(ANALYZE_DIR, f'pca_loadings_Ds_min_max_final_{extract_id}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'pca_loadings_Ds_min_max_final_{extract_id}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)


#%% Do kde plot
    g_colors = [(1, 1, 1), np.divide((153, 153, 153), 256)]
    cmap_name = 'custom_colormap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, g_colors)
    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)
    #colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255),
    #          np.divide((55, 76, 128), 256)]


    for idx, model_loads in tqdm(enumerate(model_pca_loads)):
        True
        ax = plt.subplot(2, 4, idx + 1)
        ax.set_box_aspect(1)
        ax.set_title(model_names[idx])
        l_all = model_loads['act'][:, :2]
        data = pd.DataFrame({'X': l_all[:, 0], 'Y': l_all[:, 1]})
        # create kde plot for l_all using sns
        #sns.kdeplot(data=data, x='X', y='Y', cmap=cm,ax=ax, shade=True)
        sns.kdeplot(data=data, x='X', y='Y', cmap=cm, shade=True, linewidths=0.5, linecolor='k')
        #sns.kdeplot(l_all[:, 0], l_all[:, 1], shade=True, shade_lowest=False, ax=ax, alpha=.3, color=grays)
        # ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
        #ax.scatter(l_all[:, 0], l_all[:, 1], s=.5, c=grays)
        # ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
        l_min = model_loads['act_min'][:, :2]
        #ax.scatter(l_min[:, 0], l_min[:, 1], s=2, c=colors[0],linewidths=0.5, linecolor='k')
        sns.scatterplot(x=l_min[:, 0], y=l_min[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[0])
        l_max = model_loads['act_max'][:, :2]
       # ax.scatter(l_max[:, 0], l_max[:, 1], s=2, c=colors[1],linewidths=0.5, linecolor='k')
        sns.scatterplot(x=l_max[:, 0], y=l_max[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[1])
        right_side = ax.spines["right"]
        right_side.set_visible(False)
        right_side = ax.spines["top"]
        right_side.set_visible(False)
        right_side = ax.spines["left"]
        right_side.set_visible(False)
        right_side = ax.spines["bottom"]
        right_side.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(ax.get_xlim(), [0, 0], '-', zorder=2, linewidth=1, color=(.5, .5, .5))
        ax.plot([0, 0], ax.get_ylim(), '-', zorder=2, linewidth=1, color=(.5, .5, .5))
        exp_varaince=100*sum(model_loads['exp_variance'][:2])
        ax.set_title(f"{ext_obj.model_spec[idx]}\n{exp_varaince:.2f}%", rotation=0,
                      fontsize=8)
    fig.show()

    save_loc = Path(ANALYZE_DIR, f'pca_loadings_Ds_min_max_final_{extract_id}_KDE.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'pca_loadings_Ds_min_max_final_{extract_id}_KDE.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)

    #%% UMAP analysis
    min_dist=.1
    n_neighbors=50
    metric='euclidean'

    model_umap_loads = []
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    for idx, act_dict in tqdm(enumerate(ext_obj.model_group_act)):
        # backward compatibility
        act_ = np.asarray(
            [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']])

        # find rows corresponds to ds_min_loc and put the in act_min
        act_min = act_[ds_min_loc, :]
        act_max = act_[ds_max_loc, :]
        act_rand = act_[ds_rand_loc, :]
        # remove act_min, act_max, act_rand from act_
        act_ = np.delete(act_, [ds_min_loc, ds_max_loc, ds_rand_loc], axis=0)
        act_umap = umap_obj.fit_transform(act_)
        act_min_umap = umap_obj.transform(act_min)
        act_max_umap = umap_obj.transform(act_max)
        act_rand_umap = umap_obj.transform(act_rand)
        model_umap_loads.append({'act': act_umap, 'act_min': act_min_umap, 'act_max': act_max_umap, 'act_rand': act_rand_umap,'umap_configs':[min_dist,n_neighbors,metric]})




    #%% plot UMAPmodel_names=[x['model_name'] for x in ext_obj.model_group_act]
    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)

    for idx, model_loads in tqdm(enumerate(model_umap_loads)):
        True
        ax = plt.subplot(2, 4, idx + 1)
        ax.set_box_aspect(1)
        ax.set_title(model_names[idx])
        l_all = model_loads['act'][:, :2]
        data = pd.DataFrame({'X': l_all[:, 0], 'Y': l_all[:, 1]})
        sns.kdeplot(data=data, x='X', y='Y', cmap=cm, shade=True, linewidths=0.5, linecolor='k')
        l_min = model_loads['act_min'][:, :2]

        sns.scatterplot(x=l_min[:, 0], y=l_min[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[0])
        l_max = model_loads['act_max'][:, :2]

        sns.scatterplot(x=l_max[:, 0], y=l_max[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[1])
        right_side = ax.spines["right"]
        right_side.set_visible(False)
        right_side = ax.spines["top"]
        right_side.set_visible(False)
        right_side = ax.spines["left"]
        right_side = ax.spines["bottom"]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ext_obj.model_spec[idx]}", rotation=0,
                      fontsize=8)
    fig.show()

    save_loc = Path(ANALYZE_DIR, f'UMAP_loadings_Ds_min_max_final_{extract_id}_neigh_{n_neighbors}_dist_{min_dist}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'UMAP_loadings_Ds_min_max_final_{extract_id}_neigh_{n_neighbors}_dist_{min_dist}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)


    #%%
    # %% TSNE analysis
    n_components = 2
    perplexity = 50
    learning_rate = 50

    model_tsne_loads = []
    tsne_obj = TSNE(n_components=n_components, learning_rate=learning_rate, perplexity = perplexity)
    for idx, act_dict in tqdm(enumerate(ext_obj.model_group_act)):
        # backward compatibility
        act_ = np.asarray([x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']])

        # find rows corresponds to ds_min_loc and put the in act_min
        #act_min = act_[ds_min_loc, :]
        #act_max = act_[ds_max_loc, :]
        #act_rand = act_[ds_rand_loc, :]
        # remove act_min, act_max, act_rand from act_
        #act_ = np.delete(act_, [ds_min_loc, ds_max_loc, ds_rand_loc], axis=0)
        act_ = cudf.DataFrame(act_)
        # check if act_ is on gpu


        act_tsne = tsne_obj.fit_transform(act_)
        #act_min_tsne = tsne_obj.transform(act_min)
        #act_max_tsne = tsne_obj.transform(act_max)
        #act_rand_tsne = tsne_obj.transform(act_rand)
        model_tsne_loads.append(
            {'act': act_tsne.values, })#'act_min': act_min_tsne, 'act_max': act_max_tsne, 'act_rand': act_rand_tsne,'tsne_configs': [n_components, perplexity]})



    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)

    for idx, model_loads in tqdm(enumerate(model_tsne_loads)):
        True
        ax = plt.subplot(2, 4, idx + 1)
        ax.set_box_aspect(1)
        ax.set_title(model_names[idx])
        l_all = model_loads['act'].get()[:, :2]
        data = pd.DataFrame({'X': l_all[:, 0], 'Y': l_all[:, 1]})
        # do scatter plot of all datapoints using seaborn
        #sns.scatterplot(data=data, x='X', y='Y', color=grays, s=1, linewidth=0)

        sns.kdeplot(data=data, x='X', y='Y', cmap=cm, shade=True, linewidths=0.5, linecolor='k')
        l_min = model_loads['act'].get()[ds_min_loc, :2]
        #ax.scatter(l_min[:, 0], l_min[:, 1], s=1, c=colors[0])
        sns.scatterplot(x=l_min[:, 0], y=l_min[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[0])
        l_max = model_loads['act'].get()[ds_max_loc, :2]
        #ax.scatter(l_max[:, 0], l_max[:, 1], s=1, c=colors[1])
        sns.scatterplot(x=l_max[:, 0], y=l_max[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[1])
        right_side = ax.spines["right"]
        right_side.set_visible(False)
        right_side = ax.spines["top"]
        right_side.set_visible(False)
        right_side = ax.spines["left"]
        right_side.set_visible(False)
        right_side = ax.spines["bottom"]
        right_side.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ext_obj.model_spec[idx]}", rotation=0,
                     fontsize=8)
    fig.show()

    save_loc = Path(ANALYZE_DIR, f'tsne_loadings_Ds_min_max_final_{extract_id}_perp_{perplexity}_lr_{learning_rate}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'tsne_loadings_Ds_min_max_final_{extract_id}_perp_{perplexity}_lr_{learning_rate}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)

    #%% do an allignment
    pairs = torch.combinations(torch.arange(len(ext_obj.model_group_act)), with_replacement=False)
        # backward compatibility
    JMDS_max_list=[]
    JMDS_min_list = []
    for p1,p2 in tqdm(pairs,total=len(pairs)):
        act_1 = np.asarray([x[0] for x in ext_obj.model_group_act[int(p1)]['activations']])
        act_2 = np.asarray([x[0] for x in ext_obj.model_group_act[int(p2)]['activations']])
        act_1_min=torch.from_numpy(act_1[ds_min_loc,:])
        act_2_min = torch.from_numpy(act_2[ds_min_loc, :])
        act_1_rdm_min=1-torch.corrcoef(act_1_min)
        act_2_rdm_min=1-torch.corrcoef(act_2_min)
        act_1_max = torch.from_numpy(act_1[ds_max_loc, :])
        act_2_max = torch.from_numpy(act_2[ds_max_loc, :])
        act_1_rdm_max = 1 - torch.corrcoef(act_1_max)
        act_2_rdm_max = 1 - torch.corrcoef(act_2_max)



    # do for max locations
        JMDS=JointMDS(n_components=2, dissimilarity='precomputed',return_stress=True)
        Z1_min,Z2_min,P_min,S_min=JMDS.fit_transform(act_1_rdm_min, act_2_rdm_min)
        JMDS_min_list.append([(int(p1), int(p2)), Z1_min, Z2_min, P_min, S_min])
        # do for max
        JMDS=JointMDS(n_components=2, dissimilarity='precomputed',return_stress=True)
        Z1_max,Z2_max,P_max,S_max=JMDS.fit_transform(act_1_rdm_max, act_2_rdm_max)
        JMDS_max_list.append([(int(p1),int(p2)),Z1_max,Z2_max,P_max,S_max])

    Stress_values=np.zeros((len(ext_obj.model_group_act),len(ext_obj.model_group_act)))

    np.fill_diagonal(Stress_values,np.nan)
    for k in JMDS_max_list:
        True
        Stress_values[k[0]]=k[-1]
    for k in JMDS_min_list:
        Stress_values[(k[0][1],k[0][0])]=k[-1]

    stress_max = np.asarray([x[-1] for x in JMDS_max_list])
    stress_min = np.asarray([x[-1] for x in JMDS_min_list])
    np.argmax(stress_max-stress_min)
    t_statistic, p_value = stats.ttest_rel(stress_max, stress_min)



    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio=8/11
    ax = plt.axes((.1, .05, .1, .25*pap_ratio))

    color_set = [colors[0],colors[1]]

    rdm_vec = np.vstack((stress_min, stress_max))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2,], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2], rdm_vec[:, i], color=color_set, s=10, marker='o', alpha=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['S_min', 'S_max'], fontsize=8)
    ax.set_ylabel('Stress')
    ax.set_ylim((0, .12))
    ax.set_title('Alignment')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 2.25))

    model_1=[[i,[model_names[x] for x in JMDS_max_list[i][0]]] for i in range(21)]
    kk=12
    # do scatter plot
    ax = plt.axes((.6, .05, .25, .25*pap_ratio))

    # ax.scatter(l_min[:, 0], l_min[:, 1], s=2, c=colors[0],linewidths=0.5, linecolor='k')
    l_1=JMDS_min_list[kk][1]
    sns.scatterplot(x=l_1[:, 0], y=l_1[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[0])
    l_2 = JMDS_min_list[kk][2]
    # ax.scatter(l_max[:, 0], l_max[:, 1], s=2, c=colors[1],linewidths=0.5, linecolor='k')
    sns.scatterplot(x=l_2[:, 0], y=l_2[:, 1], edgecolor='w', linewidth=0.5, s=4,marker='s', color=colors[0])
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    right_side = ax.spines["top"]
    right_side.set_visible(False)
    right_side = ax.spines["left"]
    right_side.set_visible(False)
    right_side = ax.spines["bottom"]
    right_side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(ax.get_xlim(), [0, 0], '-', zorder=2, linewidth=1, color=(.5, .5, .5))
    ax.plot([0, 0], ax.get_ylim(), '-', zorder=2, linewidth=1, color=(.5, .5, .5))

    ax = plt.axes((.3, .05, .25, .25*pap_ratio))

    # ax.scatter(l_min[:, 0], l_min[:, 1], s=2, c=colors[0],linewidths=0.5, linecolor='k')

    l_1 = JMDS_max_list[kk][1]
    sns.scatterplot(x=l_1[:, 0], y=l_1[:, 1], edgecolor='black', linewidth=0.5, s=4, color=colors[1])
    l_2 = JMDS_max_list[kk][2]
    # ax.scatter(l_max[:, 0], l_max[:, 1], s=2, c=colors[1],linewidths=0.5, linecolor='k')
    sns.scatterplot(x=l_2[:, 0], y=l_2[:, 1], edgecolor='w', linewidth=0.5, s=4, marker='s', color=colors[1])
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    right_side = ax.spines["top"]
    right_side.set_visible(False)
    right_side = ax.spines["left"]
    right_side.set_visible(False)
    right_side = ax.spines["bottom"]
    right_side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(ax.get_xlim(), [0, 0], '-', zorder=2, linewidth=1, color=(.5, .5, .5))
    ax.plot([0, 0], ax.get_ylim(), '-', zorder=2, linewidth=1, color=(.5, .5, .5))

    save_loc = Path(ANALYZE_DIR, f'JointMDS_loadings_{model_1[kk][1][0]}_{model_1[kk][1][1]}_Ds_min_max_final.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'JointMDS_loadings_{model_1[kk][1][0]}_{model_1[kk][1][1]}_Ds_min_max_final.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)

