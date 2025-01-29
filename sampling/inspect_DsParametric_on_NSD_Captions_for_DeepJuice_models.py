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


deepjuice_path='/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
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
import matplotlib
import scipy.io

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib
import scipy.io
from glob import glob
from scipy.stats import ks_2samp
from scipy.stats import shapiro, anderson, kstest, norm, probplot

matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 10,'font.weight':'bold'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False

import pickle
from glob import glob
if __name__ == '__main__':
    extract_mode='redux'
    n_samples=80
    extract_id = 'group=best_performing_pereira_1-dataset=NSD_benchmark_captions_clean_v3_textNoPeriod-activation-bench=None-ave=False'
    optim_id_min = f'coordinate_ascent_eh-obj=2-D_s-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    optim_id_max = f'coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    optim_id_rand = f'coordinate_ascent_eh-obj=D_s_rand-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'

    ext_obj=extract_pool[extract_id]()
    #deepjuice_identifier=f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    #ext_obj.identifier=deepjuice_identifier
    selected_models=['torchvision_alexnet_imagenet1k_v1',
                    'torchvision_regnet_x_800mf_imagenet1k_v2',
                     'openclip_vit_b_32_laion2b_e16',
                     'timm_swinv2_cr_tiny_ns_224',
                     'torchvision_efficientnet_b1_imagenet1k_v2',
                     'clip_rn50',
                     'timm_convnext_large_in22k',
                     ]

    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
    for model_ in tqdm(selected_models):
        save_file = f'{deepjuice_path}/nsd/{model_}*{extract_mode}.pkl'
        original_files = glob(save_file)
        # open the file
        with open(original_files[0], 'rb') as f:
            original = pickle.load(f)
        layer_id = original[0]
        act_ = original[1]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)


    optim_obj=optim_pool[optim_id_min]()
    optim_obj.N_S=1000
    optim_obj.extract_type='activation'
    optim_obj.activations = activations_list
    optim_obj.extractor_obj=ext_obj
    optim_obj.early_stopping=False
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                                 save_results=False)
    # read the excel that contains the selected sentences
    # %%  Load ds min and ds max data from NSD_caption_optimization
    (extract_short_hand, optim_short_hand_min) = make_shorthand(extract_id, optim_id_min)
    ds_min_path=f'{RESULTS_DIR}/results_{extract_short_hand}_{optim_short_hand_min}.pkl'
    with open(ds_min_path, 'rb') as f:
        results_ds_min = pickle.load(f)

    (extract_short_hand, optim_short_hand_max) = make_shorthand(extract_id, optim_id_max)
    ds_max_path = f'{RESULTS_DIR}/results_{extract_short_hand}_{optim_short_hand_max}.pkl'
    with open(ds_max_path, 'rb') as f:
        results_ds_max = pickle.load(f)

    (extract_short_hand, optim_short_rand) = make_shorthand(extract_id, optim_id_rand)
    ds_rand_path = f'{RESULTS_DIR}/results_{extract_short_hand}_{optim_short_rand}.pkl'
    with open(ds_rand_path, 'rb') as f:
        results_ds_rand = pickle.load(f)


    ds_min_loc = results_ds_min['optimized_S']
    ds_max_loc = results_ds_max['optimized_S']
    ds_rand_loc = results_ds_rand['optimized_S']
    #%%
    d_all_loc = list(
        set(np.arange(0, 1000)) - set(ds_min_loc) - set(ds_rand_loc) - set(ds_max_loc))
    # make sure d_id_leftout and ds_min_image_ids dont share any elements
    assert len(set(d_all_loc).intersection(set(ds_min_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_rand_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_max_loc))) == 0


    d_s_min, RDM_min = optim_obj.gpu_object_function_debug(ds_min_loc)
    d_s_rand, RDM_rand = optim_obj.gpu_object_function_debug(ds_rand_loc)
    d_s_max, RDM_max = optim_obj.gpu_object_function_debug(ds_max_loc)
    d_s_all, RDM_all = optim_obj.gpu_object_function_debug(d_all_loc)
    RDM_min=RDM_min.cpu()
    RDM_rand=RDM_rand.cpu()
    RDM_max = RDM_max.cpu()
    RDM_all = RDM_all.cpu()

    model_names_new_order=[0,4,2,1,3,6,5]
    model_names_new = [selected_models[i] for i in model_names_new_order]
    models_sh=['AlexNet','RegNet','ViT','Swin','EfficientNet','CLIP','ConvNext']
    model_sh_rotated=[models_sh[i] for i in model_names_new_order]
    # reorder the RDMs
    RDM_max = np.triu(RDM_max, k=1).T + np.triu(RDM_max, k=1)
    RDM_min = np.triu(RDM_min, k=1).T + np.triu(RDM_min, k=1)
    RDM_rand = np.triu(RDM_rand, k=1).T + np.triu(RDM_rand, k=1)
    RDM_all = np.triu(RDM_all, k=1).T + np.triu(RDM_all, k=1)

    RDM_max_new = RDM_max[model_names_new_order, :]
    RDM_max_new = RDM_max_new[:, model_names_new_order]
    RDM_min_new = RDM_min[model_names_new_order, :]
    RDM_min_new = RDM_min_new[:, model_names_new_order]
    RDM_rand_new = RDM_rand[model_names_new_order, :]
    RDM_rand_new = RDM_rand_new[:, model_names_new_order]
    RDM_all_new = RDM_all[model_names_new_order, :]
    RDM_all_new = RDM_all_new[:, model_names_new_order]
    #%%
    RDM_max = RDM_max_new
    RDM_min = RDM_min_new
    RDM_rand = RDM_rand_new
    RDM_all = RDM_all_new
    # create a dictionary with figure_3_data
    mask = np.triu(np.ones_like(RDM_max, dtype=bool))
    mask = np.where(mask, np.nan, 1)
    rdm_rand_vec = RDM_rand[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_max_vec = RDM_max[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_min_vec = RDM_min[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_all_vec = RDM_all[np.tril_indices(RDM_max.shape[0], k=-1)]
    model_pairs = []
    for i in range(len(model_names_new)):
        for j in range(i + 1, len(model_names_new)):
            model_pairs.append((model_names_new[i], model_names_new[j]))

    figure_3_data = {'RDM_max': RDM_max, 'RDM_min': RDM_min, 'RDM_rand': RDM_rand,'RDM_all':RDM_all, 'rdm_rand_vec': rdm_rand_vec,
                        'rdm_max_vec': rdm_max_vec, 'rdm_min_vec': rdm_min_vec,'rdm_all_vec':rdm_all_vec, 'model_pairs': model_pairs,
                        'model_names': model_names_new, 'mask': mask}

    # get the actuall RDMS
    X_Max_min_rand = []
    S_ids = [ds_max_loc, ds_min_loc, ds_rand_loc, d_all_loc]
    for idx, S_id in enumerate(S_ids):
        X_=[]
        for XY_corr in optim_obj.XY_corr_list:
            pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
            X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            X_.append(X_sample)
        X_Max_min_rand.append(X_)
    RDM_max_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_max_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_max_dict[model_name] = X_Max_min_rand[0][model_id]
    RDM_min_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_min_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_min_dict[model_name] = X_Max_min_rand[1][model_id]
    RDM_rand_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_rand_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_rand_dict[model_name] = X_Max_min_rand[2][model_id]
    RDM_all_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_all_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_all_dict[model_name] = X_Max_min_rand[3][model_id]

    (extract_short_hand, optim_short_hand_min) = make_shorthand(extract_id, optim_id_min)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'ds_data_Parametric_deepJuice_models_on_{extract_short_hand}_samples_{n_samples}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(figure_3_data, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_max_dict_parametric_deepJuice_models_on_{extract_short_hand}_samples_{n_samples}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_max_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_min_dict_parametric_deepJuice_models_on_{extract_short_hand}_samples_{n_samples}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_min_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_rand_dict_parametric_deepJuice_models_on_{extract_short_hand}_samples_{n_samples}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_rand_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_all_dict_parametric_deepJuice_models_on_{extract_short_hand}_samples_{n_samples}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_all_dict, f)

    #%%
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255)]
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio = 8 / 11
    ax = plt.axes((.2, .6, .08, .25 * pap_ratio))
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

    ax = plt.axes((.6, .73, .25, .25 * pap_ratio))
    im = ax.imshow(RDM_rand, cmap='viridis', vmax=RDM_max.max())
    # add values to image plot
    for i in range(RDM_rand.shape[0]):
        for j in range(RDM_rand.shape[1]):
            text = ax.text(j, i, f"{RDM_rand[i, j]:.2f}",
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_title('RDM_rand')
    # set ytick labels to ext_obj.model_spec
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(model_names_new, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(model_names_new, fontsize=6, rotation=90)

    ax = plt.axes((.6, .4, .25, .25 * pap_ratio))
    im = ax.imshow(RDM_max, cmap='viridis', vmax=RDM_max.max())
    # add values to image plot
    for i in range(RDM_max.shape[0]):
        for j in range(RDM_max.shape[1]):
            text = ax.text(j, i, f'{RDM_max[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(model_names_new, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(model_names_new, fontsize=6, rotation=90)

    ax.set_title('RDM_max')
    np.fill_diagonal(RDM_min, np.nan)
    ax = plt.axes((.6, .05, .25, .25 * pap_ratio))
    im = ax.imshow(RDM_min, cmap='viridis', vmax=RDM_max.max())
    # add values to image plot
    for i in range(RDM_min.shape[0]):
        for j in range(RDM_min.shape[1]):
            text = ax.text(j, i, f'{RDM_min[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(model_names_new, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(model_names_new, fontsize=6, rotation=90)

    ax.set_title('RDM_min')
    ax = plt.axes((.9, .05, .01, .25 * pap_ratio))
    plt.colorbar(im, cax=ax)

    # plot rdm vectors connecting points from rdom_rand to rdm max to rdm min
    # fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.1, .05, .15, .3 * pap_ratio))

    rdm_vec = np.vstack((rdm_min_vec, rdm_rand_vec, rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2, 3], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2, 3], rdm_vec[:, i], color=colors, s=10, marker='o', alpha=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['ds_min', 'ds_rand', 'ds_max'], fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim((0, 1.3))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 3.25))
    # ax.violinplot([0,1,2],rdm_vec.transpose(),showmeans=True,showextrema=False,showmedians=False)

    fig.show()

    # create a figure title
    save_path = Path(ANALYZE_DIR)
    (ext_sh,optim_sh)=make_shorthand(extract_id, optim_id_min)
    save_loc = Path(save_path.__str__(), f'ds_deepjuice_models_on_{extract_short_hand}_samples_{n_samples}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_deepjuice_models_on_{extract_short_hand}_samples_{n_samples}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)


    #%%
    X_Max = []
    S_id = ds_max_loc
    for XY_corr in optim_obj.XY_corr_list:
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        if X_sample.min() < 1e-6:
            print('zero value in X_sample')
            # print the index of the sample
            index_0=np.where(X_sample < 1e-8)
            pairs_0 = pairs[index_0]
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_Max.append(X_sample)

    X_Min = []
    S_id = ds_min_loc
    for XY_corr in optim_obj.XY_corr_list:
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)

        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        # see if any sample is almost zero
        if X_sample.min() < 1e-6:
            print('zero value in X_sample')
            # print the index of the sample
            index_0=np.where(X_sample < 1e-8)
            pairs_0 = pairs[index_0]
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

    x_max_ = []
    for a in X_Max:
        # get the upper diagonal part of a
        a_upper = a[np.triu_indices(a.shape[0], k=1)]
        x_max_.append(a_upper.squeeze())

    x_min_ = []
    for a in X_Min:
        # get the upper diagonal part of a
        a_upper = a[np.triu_indices(a.shape[0], k=1)]
        x_min_.append(a_upper.squeeze())

    x_rand_ = []
    for a in X_rand:
        # get the upper diagonal part of a
        a_upper = a[np.triu_indices(a.shape[0], k=1)]
        x_rand_.append(a_upper.squeeze())

    X_rands_many = []
    for k in tqdm(enumerate(range(500))):
        sent_random = list(np.random.choice(optim_obj.N_S, optim_obj.N_s))
        x_rand_many = []
        for XY_ in optim_obj.XY_corr_list:
            pairs = torch.combinations(torch.tensor(sent_random), with_replacement=False)
            X_sample = XY_[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            x_rand_many.append(X_sample)
        X_rands_many.append(x_rand_many)

    X_rands_many_vec = []
    for X_rand in X_rands_many:
        X_rands_many_vec.append([X[np.tril_indices(X.shape[0], k=-1)] for X in X_rand])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    for idx in range(len(x_min_)):
        x_max_mdl = np.asarray(x_max_[idx])
        x_min_mdl = np.asarray(x_min_[idx])
        x_ran_mdl = np.asarray(x_rand_[idx])
        x_rand_vec = np.stack([X_rand[idx] for X_rand in X_rands_many_vec])
        # compute the correlation between x_max and each row of x_rand_vec
        max_to_rand_sim = []
        min_to_rand_sim = []
        rand_to_rand_sim = []
        data_standardized = scaler.fit_transform(x_max_mdl.reshape(-1, 1)).flatten()
        stat, p = kstest(data_standardized, 'norm')
        print(f" MAX Kolmogorov-Smirnov Test Statistic: {stat}, p-value: {p}")
        data_standardized = scaler.fit_transform(x_min_mdl.reshape(-1, 1)).flatten()
        stat, p = kstest(data_standardized, 'norm')
        print(f" MIN Kolmogorov-Smirnov Test Statistic: {stat}, p-value: {p}")

        for x in tqdm(x_rand_vec):
            mw_stat, mw_p = mannwhitneyu(x_max_mdl, x)
            max_to_rand_sim.append([mw_stat, mw_p])
            mw_stat, mw_p = mannwhitneyu(x_ran_mdl, x)
            rand_to_rand_sim.append([mw_stat, mw_p])
            mw_stat, mw_p = mannwhitneyu(x_min_mdl, x)
            min_to_rand_sim.append([mw_stat, mw_p])
        # print the rario

        r_rand_to_rand = sum([x[1] > 0.05 for x in rand_to_rand_sim]) / len(rand_to_rand_sim)
        print(f"{selected_models[idx]} Rand to Rand ratio: {r_rand_to_rand}")
        r_min_to_rand = sum([x[1] > 0.05 for x in min_to_rand_sim]) / len(min_to_rand_sim)
        print(f"{selected_models[idx]} Min to Rand ratio: {r_min_to_rand}")
        r_max_to_rand = sum([x[1] > 0.05 for x in max_to_rand_sim]) / len(max_to_rand_sim)
        print(f"{selected_models[idx]} Max to Rand ratio: {r_max_to_rand}")
