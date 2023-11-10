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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def pca_loadings(act,var_explained=0.90):
    # act must be in m sample * n feature shape ,
    # from https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    q = min(1000, min(act.shape))
    u, s, v = torch.pca_lowrank(act, q=q,center=True)
    var_explained_curve=torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2)
    idx_cutoff = var_explained_curve < var_explained
    pca_loads=torch.matmul(u[:,idx_cutoff],torch.diag(s[idx_cutoff]))
    var_explained=s ** 2 / torch.sum(s ** 2)
    return pca_loads, var_explained[idx_cutoff], v # v is the eigenvectors

def make_rotational_colors(X_list):
    loadings_p12 = [x[:, 0:2] for x in X_list]
    loadings_p12_norm = [x / torch.norm(x, dim=1, keepdim=True) for x in loadings_p12]
    loadings_p12_len = [torch.norm(x, dim=1, keepdim=True) for x in loadings_p12]
    rot_list = []
    all_angle_fixed = []
    for idx, load_norm in enumerate(loadings_p12_norm):
        # try/except pushing to cpu
        try:
            load_norm = load_norm.cpu()
        except:
            pass
        angle = np.arccos(load_norm[:, 0].numpy())
        y_cos = load_norm[:, 1].numpy()
        angle_fixed = [angle[idy] if y > 0 else np.pi * 2 - angle[idy] for idy, y in enumerate(y_cos)]
        all_angle_fixed.append(angle_fixed)
        rot = np.argsort(angle_fixed)

        rot_list.append(rot)
    return rot_list


if __name__ == '__main__':
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
                'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True']
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
    model_names=[x['model_name'] for x in ext_obj.model_group_act]
    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)
    #colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255),
    #          np.divide((55, 76, 128), 256)]
    colors = [np.divide((188, 80, 144), 255), np.divide((255, 128, 0), 255),
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

    #act = torch.tensor(act_, dtype=float, device=device, requires_grad=False)
        #pca_loads, var_explained = pca_loadings(act)

    #optim_id=['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
    # 'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']
    low_resolution = 'False'
    optim_files = []
    optim_results = []
    for ext in extract_id:
        for optim in optim_id:
            optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
            assert(optim_file.exists())
            optim_files.append(optim_file.__str__())
            optim_results.append(load_obj(optim_file.__str__()))


    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    ext_obj()
    # compute pca loadings for ext_obj activations
    model_pca_load=[]
    model_var_explained=[]
    for idx, act_dict in (enumerate(ext_obj.model_group_act)):
        # backward compatibility
        act_ = [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
        act = torch.tensor(act_, dtype=float, device=device, requires_grad=False)
        pca_loads, var_explained = pca_loadings(act)
        model_pca_load.append(pca_loads)
        model_var_explained.append(var_explained)

    loadings_p12 = [x[:, 0:2] for x in model_pca_load]
    loadings_p12_norm = [x / torch.norm(x, dim=1, keepdim=True) for x in loadings_p12]
    loadings_p12_len = [torch.norm(x, dim=1, keepdim=True) for x in loadings_p12]

    rot_list=make_rotational_colors(model_pca_load)

    num_models = len(model_pca_load)
    h0 = sns.color_palette("hls", pca_loads.shape[0], as_cmap=True)
    line_cols = np.flipud(h0(np.arange(pca_loads.shape[0]) / pca_loads.shape[0]))
    fig = plt.figure(figsize=(8, 8))
    counter = 0
    for idx, _ in tqdm(enumerate(range(len(rot_list)))):
        rot = rot_list[idx]
        for idy in range(len(loadings_p12)):
            # print(counter)
            l_v = loadings_p12[idy]
            counter = counter + 1
            ax = plt.subplot(num_models, num_models, counter)
            ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
            # ax.axis('off')
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
            if np.mod(counter, num_models) == 1:
                ax.set_ylabel(f"{ext_obj.model_spec[idx]}", rotation=0, fontsize=6)
            if counter > num_models ** 2 - num_models:
                var_explained = np.sum(model_var_explained[idy][:2].cpu().numpy())
                ax.set_xlabel(f"var explained ={var_explained * 100:0.1f}", rotation=0, fontsize=6)


    save_loc = Path(ANALYZE_DIR, f'pca_loadings_{extract_id[0]}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'pca_loadings_{extract_id[0]}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)

    fig = plt.figure(figsize=(8, 11))
    optim_set = optim_results[0]['optimized_S']
    optim_name=optim_results[0]['optimizatin_name']
    for idx, _ in tqdm(enumerate(range(len(rot_list)))):
        ax = plt.subplot(4, 2, idx + 1)
        ax.set_box_aspect(1)

        rot = rot_list[idx]
        l_v = loadings_p12[idx]

        #ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
        #ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
        grays=(.7,.7,.7,.5)
        ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=grays)
        #ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
        ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
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
        ax.set_title(f"{ext_obj.model_spec[idx]}", rotation=0,
                     fontsize=8)
    fig.show()

    save_loc = Path(ANALYZE_DIR, f'Ds_samples_on_pca_loadings_{extract_id[0]}_{optim_name}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'Ds_samples_on_pca_loadings_{extract_id[0]}_{optim_name}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)



    fig = plt.figure(figsize=(8, 11))
    optim_set = optim_results[1]['optimized_S']
    optim_name=optim_results[1]['optimizatin_name']
    for idx, _ in tqdm(enumerate(range(len(rot_list)))):
        ax = plt.subplot(4, 2, idx + 1)
        ax.set_box_aspect(1)

        rot = rot_list[idx]
        l_v = loadings_p12[idx]
        grays=(.7,.7,.7,.5)
        #ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
        ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=grays)
        #ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
        ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
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
        ax.set_title(f"{ext_obj.model_spec[idx]}", rotation=0,
                     fontsize=8)
    fig.show()

    save_loc = Path(ANALYZE_DIR, f'Ds_samples_on_pca_loadings_{extract_id[0]}_{optim_name}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'Ds_samples_on_pca_loadings_{extract_id[0]}_{optim_name}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)




