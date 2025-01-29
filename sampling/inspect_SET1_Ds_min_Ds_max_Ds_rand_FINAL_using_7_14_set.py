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
import pickle
colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]
if __name__ == '__main__':
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_len_7_14_textNoPeriod-activation-bench=None-ave=False'
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
    sentence_data = ext_obj.data_
    sentences_from_data = [x['text'] for x in sentence_data]
    # drop the period from the end of each sentence
    sentences_from_data = [x[:-1] if x[-1] == '.' else x for x in sentences_from_data]
    #
    # find the location of ds_min_sent in sentences_from_data
    ds_min_loc_in_dat = [sentences_from_data.index(x) for x in ds_min_sent]
    ds_max_loc_in_dat = [sentences_from_data.index(x) for x in ds_max_sent]
    ds_rand_loc_in_dat = [sentences_from_data.index(x) for x in ds_rand_sent]
    # get sentence data for each ds_min, ds_max and ds_rand
    sent_max_data = [sentence_data[x] for x in ds_max_loc_in_dat]
    UPOS=[x['word_UPOS'] for x in sentence_data]
    # flatten the UPOS
    UPOS_flat=[item for sublist in UPOS for item in sublist]
    set(UPOS_flat)
    sent_min_data = [sentence_data[x] for x in ds_min_loc_in_dat]
    sent_rand_data = [sentence_data[x] for x in ds_rand_loc_in_dat]
    # create a dictionary of sentence data for each ds_min, ds_max and ds_rand
    sent_data_dict = {'sent_max': sent_max_data, 'sent_min': sent_min_data, 'sent_rand': sent_rand_data}
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'sentence_data_dsparametric_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(sent_data_dict, f)


    model_names = [x['model_name'] for x in ext_obj.model_group_act]

    for idx, act_dict in tqdm(enumerate(ext_obj.model_group_act)):
        # backward compatibility
        act_ = np.asarray([x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']])

        # find rows corresponds to ds_min_loc and put the in act_min
        act_min=act_[ds_min_loc,:]
        act_max=act_[ds_max_loc,:]
        act_rand=act_[ds_rand_loc,:]
        # make dictionary of act_min, act_max and act_rand
        act_dict={'act_min':act_min,'act_max':act_max,'act_rand':act_rand,'model_name':model_names[idx],'sent_min':ds_min_sent,'sent_max':ds_max_sent,'sent_rand':ds_rand_sent}
        # save the act_dict in ANALYZE_DIR/DsParametric
        save_path = Path(ANALYZE_DIR, 'DsParametric', f'act_dict_dsparametric_{model_names[idx]}_7_14_dec2024.pkl')
        with open(save_path.__str__(), 'wb') as f:
            pickle.dump(act_dict, f)
        # create a lefout set of act_min, act_max and act_rand
        index_all=np.arange(act_.shape[0])
        d_id_leftout = list(
            set(index_all) - set(ds_min_loc) - set(ds_max_loc) - set(ds_rand_loc))
        #sent_leftout=[sentences for x in d_id_leftout]
        act_leftout=act_[d_id_leftout,:]
        save_path = Path(ANALYZE_DIR, 'DsParametric', f'act_leftout_dsparametric_{model_names[idx]}_7_14_dec2024.pkl')
        with open(save_path.__str__(), 'wb') as f:
            pickle.dump(act_leftout, f)
        # save all act_all
        act_all_dict={'act_all':act_,'model_name':model_names[idx],'min_loc':ds_min_loc,'max_loc':ds_max_loc,'rand_loc':ds_rand_loc,'sentences':sentences}
        save_path = Path(ANALYZE_DIR, 'DsParametric', f'act_all_dsparametric_{model_names[idx]}_7_14_dec2024.pkl')
        with open(save_path.__str__(), 'wb') as f:
            pickle.dump(act_all_dict, f)




    #%%
    optim_obj = optim_pool[optim_id]()
    optim_obj.load_extractor(ext_obj)
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                             save_results=False)

    d_s_min, RDM_min = optim_obj.gpu_object_function_debug(ds_min_loc)
    d_s_rand, RDM_rand = optim_obj.gpu_object_function_debug(ds_rand_loc)
    d_s_max, RDM_max = optim_obj.gpu_object_function_debug(ds_max_loc)
    RDM_min=RDM_min.cpu()
    RDM_rand=RDM_rand.cpu()
    RDM_max = RDM_max.cpu()
    model_names = optim_obj.extractor_obj.model_spec
    model_names_new_order=[0,2,5,3,1,6,4]
    model_names_new = [model_names[i] for i in model_names_new_order]
    # reorder the RDMs
    RDM_max = np.triu(RDM_max, k=1).T + np.triu(RDM_max, k=1)
    RDM_min = np.triu(RDM_min, k=1).T + np.triu(RDM_min, k=1)
    RDM_rand = np.triu(RDM_rand, k=1).T + np.triu(RDM_rand, k=1)

    RDM_max_new = RDM_max[model_names_new_order, :]
    RDM_max_new = RDM_max_new[:, model_names_new_order]
    RDM_min_new = RDM_min[model_names_new_order, :]
    RDM_min_new = RDM_min_new[:, model_names_new_order]
    RDM_rand_new = RDM_rand[model_names_new_order, :]
    RDM_rand_new = RDM_rand_new[:, model_names_new_order]
    #%%
    RDM_max = RDM_max_new
    RDM_min = RDM_min_new
    RDM_rand = RDM_rand_new
    # create a dictionary with figure_3_data
    mask = np.triu(np.ones_like(RDM_max, dtype=bool))
    mask = np.where(mask, np.nan, 1)
    rdm_rand_vec = RDM_rand[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_max_vec = RDM_max[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_min_vec = RDM_min[np.tril_indices(RDM_max.shape[0], k=-1)]
    model_pairs = []
    for i in range(len(model_names_new)):
        for j in range(i + 1, len(model_names_new)):
            model_pairs.append((model_names_new[i], model_names_new[j]))

    figure_3_data = {'RDM_max': RDM_max, 'RDM_min': RDM_min, 'RDM_rand': RDM_rand, 'rdm_rand_vec': rdm_rand_vec,
                        'rdm_max_vec': rdm_max_vec, 'rdm_min_vec': rdm_min_vec, 'model_pairs': model_pairs,
                        'model_names': model_names_new, 'mask': mask}
    # get the actuall RDMS
    X_Max_min_rand = []
    S_ids = [ds_max_loc, ds_min_loc, ds_rand_loc]
    for idx, S_id in enumerate(S_ids):
        X_=[]
        for XY_corr in optim_obj.XY_corr_list:
            pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
            X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            X_.append(X_sample)
        X_Max_min_rand.append(X_)
    RDM_max_dict = {model_name: [] for model_name in model_names}
    for model_name in RDM_max_dict.keys():
        # get the model id from the model names
        model_id = model_names.index(model_name)
        RDM_max_dict[model_name] = X_Max_min_rand[0][model_id]
    RDM_min_dict = {model_name: [] for model_name in model_names}
    for model_name in RDM_min_dict.keys():
        # get the model id from the model names
        model_id = model_names.index(model_name)
        RDM_min_dict[model_name] = X_Max_min_rand[1][model_id]
    RDM_rand_dict = {model_name: [] for model_name in model_names}
    for model_name in RDM_rand_dict.keys():
        # get the model id from the model names
        model_id = model_names.index(model_name)
        RDM_rand_dict[model_name] = X_Max_min_rand[2][model_id]
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'ds_data_Parametric_7_14_dec2024.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(figure_3_data, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'RDM_max_dict_parametric_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_max_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'RDM_min_dict_parametric_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_min_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'RDM_rand_dict_parametric_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_rand_dict, f)
    grays = (.8, .8, .8, .5)

    #colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255),
    #          np.divide((55, 76, 128), 256)]
    colors = [np.divide((0, 157, 255), 255), np.divide((128, 128, 128), 256),np.divide((255, 98, 0), 255)]

    # get obj= from optim
    obj_id = ['Ds_max', '2-Ds_max']
    # get n_samples from each element in optim_id
    low_resolution=False

    ds_all= []
    RDM_all = []


    ds_rand_set = []
    RDM_rand_set= []
    sent_rand_ids_set = []
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optim_obj.N_S, 80))
        sent_rand_ids_set.append(sent_random)
        d_s_r, RDM_r = optim_obj.gpu_object_function_debug(sent_random)
        ds_rand_set.append(d_s_r)
        RDM_rand_set.append(RDM_r)
    RDM_rand_set = [2 - x for x in RDM_rand_set]
    RDM_rand_set = [torch.triu(x,diagonal=1).T + torch.triu(x, diagonal=1) for x in RDM_rand_set]
    RDM_rand_set = [x.cpu().numpy() for x in RDM_rand_set]


    RDM_rand_dict = {model_name: [] for model_name in optim_obj.extractor_obj.model_spec}
    for idx, XY_corr in enumerate(optim_obj.XY_corr_list):
        X_samples = []
        model_name = optim_obj.extractor_obj.model_spec[idx]
        for S_id in sent_rand_ids_set:
            pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
            X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            X_samples.append(X_sample)
        RDM_rand_dict[model_name] = X_samples
    # add ds_rand_set to the RDM_rand_dict
    RDM_rand_dict['ds_rand_set'] = ds_rand_set
    RDM_rand_dict['sent_rand_ids_set'] = sent_rand_ids_set
    RDM_rand_dict['RDM_rand_set'] = RDM_rand_set
    save_path=Path(ANALYZE_DIR,'DsParametric', f'RDM_rand_set_dict_parametric_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
          pickle.dump(RDM_rand_dict, f)

    RDM_full_dict = {model_name: [] for model_name in optim_obj.extractor_obj.model_spec}
    for idx, XY_corr in enumerate(optim_obj.XY_corr_list):
        # get the upper diagonal of XY_corr by using torch.combinations
        model_name=optim_obj.extractor_obj.model_spec[idx]
        pairs = torch.combinations(torch.tensor(range(optim_obj.N_S)), with_replacement=False)
        # sort pairs
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
        # make squareform matrix
        #X_sample = squareform(X_sample)
        RDM_full_dict[model_name] = X_sample

    save_path=Path(ANALYZE_DIR,'DsParametric', f'RDM_full_set_dict_parametric_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
          pickle.dump(RDM_full_dict, f)


    #%%
    optim_id_jsd='coordinate_ascent_eh-obj=D_s_jsd-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_obj_jsd=optim_pool[optim_id_jsd]()
    optim_obj_jsd.load_extractor(ext_obj)
    optim_obj_jsd.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                                save_results=False)

    _, _,jsd_min = optim_obj_jsd.gpu_object_function_ds_plus_jsd(ds_min_loc,debug=True)



    jsd_range=[]
    js_min_range=[]
    js_max_range=[]
    js_rand_range=[]
    for kk in tqdm(range(1000)):
        S = np.random.choice(optim_obj_jsd.N_S, optim_obj_jsd.N_s, replace=False)
        # compute objective function for the random sample
        _,_,jsds=optim_obj_jsd.gpu_object_function_ds_plus_jsd(S,debug=True)
        _, _, jsd_min = optim_obj_jsd.gpu_object_function_ds_plus_jsd(ds_min_loc, debug=True)
        _, _, jsd_rand = optim_obj_jsd.gpu_object_function_ds_plus_jsd(ds_rand_loc, debug=True)
        _, _, jsd_max = optim_obj_jsd.gpu_object_function_ds_plus_jsd(ds_max_loc, debug=True)
        jsd_range.append(torch.stack(jsds).cpu().numpy())
        js_min_range.append(torch.stack(jsd_min).cpu().numpy())
        js_max_range.append(torch.stack(jsd_max).cpu().numpy())
        js_rand_range.append(torch.stack(jsd_rand).cpu().numpy())

    jsd_range=np.stack(jsd_range)
    jsd_min=np.stack(js_min_range)
    jsd_max=np.stack(js_max_range)
    jsd_rand=np.stack(js_rand_range)
    #%%
    # create a figure with 7 panels and each one plot a histogram of jsd_rand columns
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio=8/11
    for kk in range(7):
        ax = plt.axes((.1,.7*(1-kk/7),.4,.06))
        modl_jsd_rand=jsd_rand[:,kk]
        modl_jsd_min=jsd_min[:,kk]
        modl_jsd_max=jsd_max[:,kk]
        modl_jsd_range=jsd_range[:,kk]
        # find the max across all
        max_jsd=np.max([modl_jsd_rand.max(),modl_jsd_min.max(),modl_jsd_max.max()])
        # create edges from 0 to max_jsd
        edges=np.linspace(0,max_jsd,50)
        # plot histograms
        ax.hist(modl_jsd_rand, bins=edges, color=colors[1], alpha=0.5, label='rand')
        ax.hist(modl_jsd_min, bins=edges, color=colors[0], alpha=0.5, label='min')
        ax.hist(modl_jsd_max, bins=edges, color=colors[2], alpha=0.5, label='max')
        # plot range with no colors in side and only edges
        ax.hist(modl_jsd_range, bins=edges, color='w', edgecolor='k', alpha=0.5, label='range')
        # add model id
        ax.set_title(model_names[kk])

        # plot a vertical line at jsd_min, jsd_max and jsd_rand
    fig.show()


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
    im=ax.imshow(RDM_rand, cmap='viridis',vmax=RDM_max.max())
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
    im=ax.imshow(RDM_max, cmap='viridis',vmax=RDM_max.max())
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
    np.fill_diagonal(RDM_min,np.nan)
    ax = plt.axes((.6, .05, .25, .25*pap_ratio))
    im = ax.imshow(RDM_min, cmap='viridis',vmax=RDM_max.max())
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

    rdm_rand_vec=RDM_rand[np.triu_indices(RDM_min.shape[0], k=1)]
    rdm_max_vec=RDM_max[np.triu_indices(RDM_min.shape[0], k=1)]
    rdm_min_vec=RDM_min[np.triu_indices(RDM_min.shape[0], k=1)]
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
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_FINAL_7_14_dec2024.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_FINAL_7_14_dec2024.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)
    #%% create a list of random sets
    ds_rand = []
    RDM_rand = []
    sent_rand_ids = []
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optim_obj.N_S, 80))
        sent_rand_ids.append(sent_random)
        d_s_r, RDM_r = optim_obj.gpu_object_function_debug(sent_random)
        ds_rand.append(d_s_r)
        RDM_rand.append(RDM_r)

    # save random set
    RDM_rand_dict={'RDM_rand':RDM_rand,'ds_rand':ds_rand,'sent_rand_ids':sent_rand_ids,'model_names':model_names}
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'ds_set_rand_parametric_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_rand_dict, f)

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
    lex_dict = {lex_name: [] for lex_name in lex_names}
    # for each key in lex_dict, get the lexical feature for each sentence
    for lex_name in lex_dict.keys():
        lex_values = [np.nanmean(sent_dat[lex_name]) for sent_dat in ext_obj.data_]
        lex_dict[lex_name] = lex_values
    # add sentence, words, and word length to lex_dict
    lex_dict['text'] = [sent_dat['text'] for sent_dat in ext_obj.data_]
    lex_dict['sentence_length'] = [sent_dat['sentence_length'] for sent_dat in ext_obj.data_]
    lex_dict['word_string'] = [sent_dat['word_string'] for sent_dat in ext_obj.data_]


    sent_min_lex = {lex_name: [] for lex_name in lex_dict.keys()}
    sent_rand_lex = {lex_name: [] for lex_name in lex_dict.keys()}
    sent_max_lex = {lex_name: [] for lex_name in lex_dict.keys()}
    sent_all_lex = {lex_name: [] for lex_name in lex_dict.keys()}
    for lex_name in lex_dict.keys():
        lex_vals=lex_dict[lex_name]
        sent_max_lex[lex_name] = [lex_vals[x] for x in ds_max_loc_in_dat]
        sent_min_lex[lex_name] = [lex_vals[x] for x in ds_min_loc_in_dat]
        sent_rand_lex[lex_name] = [lex_vals[x] for x in ds_rand_loc_in_dat]
        sent_all_lex[lex_name] = lex_vals

    sent_lex_rand_set = {lex_name: [] for lex_name in lex_dict.keys()}
    for lex_name in lex_dict.keys():
        # get the lex_full from the lex_dict
        lex_vals = lex_dict[lex_name]
        lex_values=[]
        for sent_random in sent_rand_ids:
            lex_value = [lex_vals[id] for id in sent_random]
            lex_values.append(lex_value)
        sent_lex_rand_set[lex_name] = lex_values

    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_max_parametric_lex_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(sent_max_lex, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_min_parametric_lex_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(sent_min_lex, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_rand_parametric_lex_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(sent_rand_lex, f)

    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_set_rand_parametric_lex_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(sent_lex_rand_set, f)

    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_all_parametric_lex_7_14_dec2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(sent_all_lex, f)


    lex_names = [x['name'] for x in LEX_PATH_SET]
    sent_max_lex_values=[[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_max_data]
    sent_min_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_min_data]
    sent_all_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_all_data]
    sent_rand_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_rand_data]


    # add num_words to the beginning of each list
    sent_max_num_words=[len(x['word_id']) for x in sent_max_data]
    sent_min_num_words = [len(x['word_id']) for x in sent_min_data]
    sent_all_num_words = [len(x['word_id']) for x in sent_all_data]
    sent_rand_num_words = [len(x['word_id']) for x in sent_rand_data]
    # add sent_max_num_words to the beginning of each sent_max_lex_values
    sent_max_lex_values=np.concatenate([np.asarray(sent_max_num_words).reshape(-1,1),np.asarray(sent_max_lex_values)],axis=1)
    sent_min_lex_values = np.concatenate([np.asarray(sent_min_num_words).reshape(-1, 1), np.asarray(sent_min_lex_values)], axis=1)
    sent_all_lex_values = np.concatenate([np.asarray(sent_all_num_words).reshape(-1, 1), np.asarray(sent_all_lex_values)], axis=1)
    # add 'num_words' to the begginig of lex_names
    lex_names = [x['name'] for x in LEX_PATH_SET]
    lex_dict = {lex_name: [] for lex_name in lex_names}

    ds_max_lex = {lex_name: [] for lex_name in lex_names}
    for lex_name in lex_names:
        lex_values = [np.nanmean(sent_dat[lex_name]) for sent_dat in sent_max_data]
        ds_max_lex[lex_name] = lex_values
    # add a new field for number of words
    ds_max_lex['num_words'] = sent_max_num_words


    ds_min_lex = {lex_name: [] for lex_name in lex_names}
    for lex_name in lex_names:
        lex_values = [np.nanmean(sent_dat[lex_name]) for sent_dat in sent_min_data]
        ds_min_lex[lex_name] = lex_values
    # add a new field for number of words
    ds_min_lex['num_words'] = sent_min_num_words

    ds_rand_lex = {lex_name: [] for lex_name in lex_names}
    for lex_name in lex_names:
        lex_values = [np.nanmean(sent_dat[lex_name]) for sent_dat in sent_rand_data]
        ds_rand_lex[lex_name] = lex_values
    # add a new field for number of words
    ds_rand_lex['num_words'] = sent_rand_num_words

    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_max_parametric_lex_7_14_dec2024_num_words.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(ds_max_lex, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_min_parametric_lex_7_14_dec2024_num_words.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(ds_min_lex, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', 'Ds_rand_parametric_lex_7_14_dec2024_num_words.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(ds_rand_lex, f)


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
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL_7_14_dec2024.png')
    # save figure as pdf and png
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL_7_14_dec2024.eps')
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
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL_7_14_dec2024.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL_7_14_dec2024.eps')
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
    similiary_path=Path(ANALYZE_DIR,'similarity_dict_DsParametric_7_14_dec2024.pkl')
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
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL_7_14_dec2024.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}_FINAL_7_14_dec2024.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    #%% perform PCA on sets

    # %% compute the pca
    pca = PCA(n_components=0.95)
    model_pca_loads = []
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
        pca.fit(act_)
        act_pca = pca.transform(act_)
        act_min_pca = pca.transform(act_min)
        act_max_pca = pca.transform(act_max)
        act_rand_pca = pca.transform(act_rand)
        model_pca_loads.append(
            {'act': act_pca, 'act_min': act_min_pca, 'act_max': act_max_pca, 'act_rand': act_rand_pca,
             'exp_variance': pca.explained_variance_ratio_})

    # create a plot with the number of models
    # get model names from ext_obj.model_group_act

    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)
    # colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255),
    #          np.divide((55, 76, 128), 256)]
    model_names = ['Roberta', 'xlnet', 'bert', 'xlm', 'gpt2', 'albert', 'ctrl']

    for idx, model_loads in tqdm(enumerate(model_pca_loads)):
        True
        ax = plt.subplot(2, 4, idx + 1)
        ax.set_box_aspect(1)
        ax.set_title(ext_obj.model_spec[idx])
        l_all = model_loads['act'][:, :2]
        # ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
        ax.scatter(l_all[:, 0], l_all[:, 1], s=.5, c=grays)
        # ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
        l_min = model_loads['act_min'][:, :2]
        ax.scatter(l_min[:, 0], l_min[:, 1], s=2, c=colors[2],edgecolor='k')

        l_rand = model_loads['act_rand'][:, :2]
        ax.scatter(l_rand[:, 0], l_rand[:, 1], s=2, c=colors[1], edgecolor='k')

        l_max = model_loads['act_max'][:, :2]
        ax.scatter(l_max[:, 0], l_max[:, 1], s=2, c=colors[0],edgecolor='k')
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
        exp_varaince = 100 * sum(model_loads['exp_variance'][:2])
        ax.set_title(f"{ext_obj.model_spec[idx]}\n{exp_varaince:.2f}%", rotation=0,
                     fontsize=8)
    fig.show()


    save_loc = Path(ANALYZE_DIR, f'pca_loadings_Ds_min_rand_max_final_{extract_id}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(ANALYZE_DIR, f'pca_loadings_Ds_min_rand_max_final_{extract_id}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)
    #%% isomap
    from sklearn.manifold import Isomap
    X=np.asarray([x[0] for x in ext_obj.model_group_act[4]['activations']])
    embedding = Isomap(n_components=2,n_neighbors=10)
    X_transformed = embedding.fit_transform(X)

    reconstruction_error=embedding.reconstruction_error()
    x_min=X_transformed[ds_min_loc]
    x_max=X_transformed[ds_max_loc]
    x_rand=X_transformed[ds_rand_loc]
    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)
 # create an exais from .1,.1,.8,.8
    ax = plt.axes((.1, .1, .8, .8*8/11))
    ax.set_box_aspect(1)
    ax.set_title(ext_obj.model_spec[4])

    # ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=.5, c=grays)
    # ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
    ax.scatter(x_min[:, 0], x_min[:, 1], s=10, c=colors[3], edgecolor='k')
    ax.scatter(x_rand[:, 0], x_rand[:, 1], s=10, c=colors[2], edgecolor='k')
    ax.scatter(x_max[:, 0], x_max[:, 1], s=10, c=colors[1], edgecolor='k')
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
    fig.show()

    embedding.dist_matrix_
    # get the dist_matrix for x_min, x_rand, and x_max and plot them
    dist_min=embedding.dist_matrix_[ds_min_loc,:][:,ds_min_loc]
    dist_rand=embedding.dist_matrix_[ds_rand_loc,:][:,ds_rand_loc]
    dist_max=embedding.dist_matrix_[ds_max_loc,:][:,ds_max_loc]
    # plot the dist_min, dist_rand, and dist_max but make the range the same
    dis_min_condition=np.concatenate([dist_min.flatten(),dist_rand.flatten(),dist_max.flatten()])
    min_val=np.min(dis_min_condition)
    max_val=np.max(dis_min_condition)
    fig = plt.figure(figsize=(11, 8))
    ax = plt.subplot(1, 3, 1)
    im=ax.imshow(dist_min,cmap='viridis', vmax=max_val,vmin=min_val)
    ax.set_title('dist_min')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(1, 3, 2)
    im=ax.imshow(dist_rand,cmap='viridis', vmax=max_val,vmin=min_val)
    ax.set_title('dist_rand')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(1, 3, 3)
    im=ax.imshow(dist_max,cmap='viridis', vmax=max_val,vmin=min_val)
    ax.set_title('dist_max')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()
    #%% import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import LocallyLinearEmbedding

    X = np.asarray([x[0] for x in ext_obj.model_group_act[4]['activations']])
    embedding = LocallyLinearEmbedding(n_components=2)
    X_transformed = embedding.fit_transform(X)

    x_min=X_transformed[ds_min_loc]
    x_max=X_transformed[ds_max_loc]
    x_rand=X_transformed[ds_rand_loc]
    fig = plt.figure(figsize=(11, 8))
    grays = (.8, .8, .8, .5)
 # create an exais from .1,.1,.8,.8
    ax = plt.axes((.1, .1, .8, .8*8/11))
    ax.set_box_aspect(1)
    ax.set_title(ext_obj.model_spec[4])

    # ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=.5, c=grays)
    # ax.scatter(l_v[optim_set, 0].cpu(), l_v[optim_set, 1].cpu(), s=.7, c='k')
    ax.scatter(x_min[:, 0], x_min[:, 1], s=10, c=colors[3], edgecolor='k')
    ax.scatter(x_rand[:, 0], x_rand[:, 1], s=10, c=colors[2], edgecolor='k')
    ax.scatter(x_max[:, 0], x_max[:, 1], s=10, c=colors[1], edgecolor='k')
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
    fig.show()