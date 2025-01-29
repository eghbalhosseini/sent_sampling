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
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib
import scipy.io
from glob import glob
matplotlib.rcParams.update({'font.size': 10,'font.weight':'bold'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
from scipy.stats import ks_2samp
from scipy.stats import shapiro, anderson, kstest, norm, probplot
import seaborn
colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255)]
import pickle
from glob import glob
import time

if __name__ == '__main__':
    extract_mode='redux'
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    #%%  set up extraction
    ext_obj=extract_pool[extract_id]()
    deepjuice_identifier=f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    ext_obj.identifier=deepjuice_identifier
    selected_models=['torchvision_alexnet_imagenet1k_v1',
                    'torchvision_regnet_x_800mf_imagenet1k_v2',
                     'openclip_vit_b_32_laion2b_e16',
                     'timm_swinv2_cr_tiny_ns_224',
                     'torchvision_efficientnet_b1_imagenet1k_v2',
                     'clip_rn50',
                     'timm_convnext_large_in22k',
                     ]
    ext_obj.model_spec=selected_models
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

    #%%
    optimizer_id = "coordinate_ascent_eh-obj=2-D_s_kl_div-n_iter=50-n_samples=80-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    optimizer_obj=optim_pool[optimizer_id]()
    optimizer_obj.N_S=1000
    optimizer_obj.extract_type='activation'
    optimizer_obj.activations = activations_list
    optimizer_obj.extractor_obj=ext_obj
    optimizer_obj.early_stopping=False

    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                                 save_results=False)

    # %%
    kl_muliplier = 5.0
    kl_threshold = 0.06
    bins = 200
    epsilon = 1e-10

    optimizer_id_max = f"coordinate_ascent_eh-obj=D_s_kl_div-n_iter=50-n_samples=80-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"

    [ext_id, opt_id_max] = make_shorthand(deepjuice_identifier, optimizer_id_max)
    optim_file = os.path.join(RESULTS_DIR,
                              f'res_{ext_id}_{opt_id_max}_kl_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_norm.pkl')

    # check when the file was created


    if os.path.exists(optim_file):
        modification_time = os.path.getmtime(optim_file)
        modification_date = time.ctime(modification_time)
        print(f"The file was last modified on: {modification_date}")
        optim_max = load_obj(optim_file)
    else:
        print(f"File {optim_file} not found")

    # load optim_max
    optimizer_id_min = f"coordinate_ascent_eh-obj=2-D_s_kl_div-n_iter=50-n_samples=80-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    [ext_id, opt_id_min] = make_shorthand(deepjuice_identifier, optimizer_id_min)
    optim_file = os.path.join(RESULTS_DIR,
                              f'res_{ext_id}_{opt_id_min}_kl_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_norm.pkl')
    if os.path.exists(optim_file):
        modification_time = os.path.getmtime(optim_file)
        modification_date = time.ctime(modification_time)
        print(f"The file was last modified on: {modification_date}")
        optim_min = load_obj(optim_file)
    else:
        print(f"File {optim_file} not found")
#%%

    max_vals=[]
    min_vals=[]
    idx = np.triu_indices(optimizer_obj.XY_corr_list[0].shape[0], k=1)
    for i_m in range(len(optimizer_obj.XY_corr_list)):
        vals = optimizer_obj.XY_corr_list[i_m][idx]
        print(f'model: {ext_obj.model_spec[i_m]}, min: {vals.min()}, max: {vals.max()}')
        # round max to nearest 0.1 and drop the decimal part
        max_vals.append(torch.ceil(vals.max() * 10) / 10)
        min_vals.append(torch.floor(vals.min() * 10) / 10)
    # extract ev sentences
    # find location of ev sentences in sentences
    random_hist = []
    XY_corr_hist_list = []
    bins = 200
    epsilon = 1e-10
    for kk in tqdm(range(200)):
        S = np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False)
        # compute objective function for the random sample
        samples = torch.tensor(S, dtype=torch.long, device=optimizer_obj.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]].to(optimizer_obj.device) for XY_corr in
                          optimizer_obj.XY_corr_list]
        XY_corr_hist = [torch.histc(x_ref, bins=bins, min=0, max=2) for x_ref in XY_corr_sample]
        XY_corr_hist = [(hist_ref / torch.sum(hist_ref)) + epsilon for hist_ref in XY_corr_hist]
        XY_corr_hist = [p_smooth / p_smooth.sum() for p_smooth in XY_corr_hist]

        XY_corr_hist_list.append(torch.stack(XY_corr_hist))

    XY_corr_hist_mean=torch.stack(XY_corr_hist_list,dim=-1).mean(dim=-1)
    # normlaize along the last dimension to get a probability distribution per each row
    XY_corr_hist_mean = XY_corr_hist_mean / XY_corr_hist_mean.sum(dim=-1, keepdim=True)
    XY_corr_hist=torch.stack(XY_corr_hist,dim=0)


    kl_div_rnd = []
    for kk in tqdm(range(1000)):
        S = np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False)
        # compute objective function for the random sample
        samples = torch.tensor(S, dtype=torch.long, device=optimizer_obj.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]].to(optimizer_obj.device) for XY_corr in
                          optimizer_obj.XY_corr_list]
        XY_corr_hist = [torch.histc(x_ref, bins=bins, min=min_vals[id_m], max=max_vals[id_m]) for id_m,x_ref in enumerate(XY_corr_sample)]
        XY_corr_hist = [(hist_ref / torch.sum(hist_ref)) + epsilon for hist_ref in XY_corr_hist]
        XY_corr_hist = [p_smooth / p_smooth.sum() for p_smooth in XY_corr_hist]
        XY_corr_hist = torch.stack(XY_corr_hist, dim=0)
        kl_div_pm = (XY_corr_hist * (XY_corr_hist.log() - XY_corr_hist_mean.log())).mean(dim=-1)
        kl_div_rnd.append(kl_div_pm)

    kl_div_rnd=torch.stack(kl_div_rnd)
    kl_div_rnd_max=kl_div_rnd.max(dim=0)[0]

    optimizer_obj.XY_corr_hist_mean = XY_corr_hist_mean
    optimizer_obj.bins = bins
    optimizer_obj.kl_div_rnd_max=kl_muliplier*kl_div_rnd_max
    optimizer_obj.epsilon=1e-10
    optimizer_obj.kl_div_threshold=kl_threshold
    optimizer_obj.kl_div_muliplier=kl_muliplier
    optimizer_obj.corr_min_max=list(zip(min_vals,max_vals))

    #%

    #%% create a ds_rand condition
    optim_id_random=f'coordinate_ascent_eh-obj=D_s_rand_kl_div-n_iter=50-n_samples=80-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    (extract_short_hand, optim_short_rand) = make_shorthand(deepjuice_identifier, optim_id_random)
    ds_rand_path = f'{RESULTS_DIR}/res_{extract_short_hand}_{optim_short_rand}_{extract_mode}_kl_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_norm.pkl'
    # if path ds_rand_path exists, load it
    if Path(ds_rand_path).exists():
        with open(ds_rand_path, 'rb') as f:
            results_ds_rand = pickle.load(f)

    else:
        ds_rand = []
        RDM_rand = []
        sent_random_set=[]
        for k in tqdm(enumerate(range(1000))):
            sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s,replace=False))
            d_s_r, RDM_r = optimizer_obj.gpu_object_function_debug(sent_random)
            ds_rand.append(d_s_r)
            RDM_rand.append(RDM_r)
            sent_random_set.append(sent_random)
        # find ds_rand closest to mean
        ds_rand_set = np.argmin(np.abs(np.mean(ds_rand) - np.array(ds_rand)))
        ds_rand_loc = sent_random_set[ds_rand_set]
        results_ds_rand = dict(extractor_name=deepjuice_identifier,
                             model_spec=selected_models,
                             layer_spec=layers_list,
                             optimizatin_name=optim_id_random,
                             optimized_S=ds_rand_loc,
                             optimized_d=ds_rand[ds_rand_set])


        save_obj(results_ds_rand, ds_rand_path)
        
    #%% 
    ds_rand_loc = results_ds_rand['optimized_S']
    ds_min_loc=optim_min['optimized_S']
    ds_max_loc=optim_max['optimized_S']
    #%%
    (ext_sh, optim_sh) = make_shorthand(deepjuice_identifier, optimizer_id)
    d_all_loc = list(
        set(np.arange(0, 1000)) - set(ds_min_loc) - set(ds_rand_loc) - set(ds_max_loc))
    # make sure d_id_leftout and ds_min_image_ids dont share any elements
    assert len(set(d_all_loc).intersection(set(ds_min_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_rand_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_max_loc))) == 0




    d_s_min, RDM_min = optimizer_obj.gpu_object_function_debug(ds_min_loc)
    [ds_kl_min, _, kl_div_min]=optimizer_obj.gpu_object_function_ds_kl_div(ds_min_loc, debug=True)
    d_s_rand, RDM_rand = optimizer_obj.gpu_object_function_debug(ds_rand_loc)
    [ds_kl_rand, _, kl_div_rand]=optimizer_obj.gpu_object_function_ds_kl_div(ds_rand_loc, debug=True)
    d_s_max, RDM_max = optimizer_obj.gpu_object_function_debug(ds_max_loc)
    [ds_kl_max, _, kl_div_max]=optimizer_obj.gpu_object_function_ds_kl_div(ds_max_loc, debug=True)


    RDM_min=RDM_min.cpu()
    RDM_rand=RDM_rand.cpu()
    RDM_max = RDM_max.cpu()
    model_names = selected_models
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
    #%
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
    #%% create a set of random images
    ds_rand_set = []
    RDM_rand_set= []
    sent_rand_ids_set = []
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        sent_rand_ids_set.append(sent_random)
        d_s_r, RDM_r = optimizer_obj.gpu_object_function_debug(sent_random)
        ds_rand_set.append(d_s_r)
        RDM_rand_set.append(RDM_r)
    ds_rand_set = 2 - np.asarray(ds_rand_set)
    RDM_rand_set = [2 - x for x in RDM_rand_set]
    RDM_rand_set = [torch.triu(x,diagonal=1).T + torch.triu(x, diagonal=1) for x in RDM_rand_set]

    RDM_full_dict = {model_name: [] for model_name in optimizer_obj.extractor_obj.model_spec}
    for idx, XY_corr in enumerate(optimizer_obj.XY_corr_list):
        # get the upper diagonal of XY_corr by using torch.combinations
        model_name=optimizer_obj.extractor_obj.model_spec[idx]
        pairs = torch.combinations(torch.tensor(range(optimizer_obj.N_S)), with_replacement=False)
        # sort pairs
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
        # make squareform matrix
        #X_sample = squareform(X_sample)
        RDM_full_dict[model_name] = X_sample

    # save_path=Path(ANALYZE_DIR,'DsParametric', f'RDM_all_dict_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_sep2024.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #       pickle.dump(RDM_full_dict, f)


    #S_id = sent_random
    RDM_rand_dict = {model_name: [] for model_name in optimizer_obj.extractor_obj.model_spec}
    for idx, XY_corr in enumerate(optimizer_obj.XY_corr_list):
        X_samples=[]
        model_name=optimizer_obj.extractor_obj.model_spec[idx]
        for S_id in sent_rand_ids_set:
            pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
            X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
        # make squareform matrix
            X_sample = squareform(X_sample)
            X_samples.append(X_sample)
        RDM_rand_dict[model_name] = X_samples

    # save_path=Path(ANALYZE_DIR,'DsParametric', f'RDM_rand_dict_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_sep2024.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #       pickle.dump(RDM_rand_dict, f)

    #%% create and save the ds data for deepjuice
    figure_3_data = {'RDM_max': RDM_max, 'RDM_min': RDM_min, 'RDM_rand': RDM_rand, 'rdm_rand_vec': rdm_rand_vec,
                     'rdm_max_vec': rdm_max_vec, 'rdm_min_vec': rdm_min_vec, 'model_pairs': model_pairs,
                     'model_names': model_names_new, 'mask': mask}


    X_Max_min_rand = []
    S_ids = [ds_max_loc, ds_min_loc, ds_rand_loc, d_all_loc]
    for idx, S_id in enumerate(S_ids):
        X_ = []
        for XY_corr in optimizer_obj.XY_corr_list:
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
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'ds_data_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(figure_3_data, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_max_dict_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_max_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_min_dict_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_min_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric',
                     f'RDM_rand_dict_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_rand_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_all_dict_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_all_dict, f)

    #%%
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

    save_path = Path(ANALYZE_DIR)
    (ext_sh, optim_sh) = make_shorthand(deepjuice_identifier, optimizer_id)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}_norm.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}_norm.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)

    #%
    
    #%%
    X_Max = []
    S_id = ds_max_loc
    for XY_corr in optimizer_obj.XY_corr_list:
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
    for XY_corr in optimizer_obj.XY_corr_list:
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
    for XY_corr in optimizer_obj.XY_corr_list:
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
    ax_title = f'S_rdm,{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}'

    # save the figure
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)

    #%%
    # plot average distances between setnences for each model
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
    #
    # save a dictionary of x_min, x_rand and x_max
    model_names = selected_models
    similirity_dict = {'x_min': x_min_, 'x_rand': x_rand_, 'x_max': x_max_}
    similiary_path = Path(ANALYZE_DIR, 'similarity_dict_DsParametric.pkl')
    save_obj(similirity_dict, similiary_path.__str__())
    fig = plt.figure(figsize=(11, 8))
    for i in tqdm(range(len(X_Max))):
        ax = plt.subplot(2, 4, i + 1)
        # create a df with 2 columns, [x_min_[i],x_rand_[i],x_max_[i]]] and a second column with 'min','rand','max'
        df = pd.DataFrame(2 - np.vstack((x_min_[i], x_rand_[i], x_max_[i])).transpose(), columns=['min', 'rand', 'max'])
        # change the colors to match the colors in the previous plot
        # melt the df
        df = pd.melt(df)
        # Box plot with transparency
        seaborn.boxplot(
        df, x="variable", y="value", ax=ax, palette="Set2",
        whis=[0, 100], width=.6,
        boxprops=dict(alpha=0.5, color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black'),
        flierprops=dict(markerfacecolor='black', markeredgecolor='black', alpha=0.5)
        )
        seaborn.stripplot(df, x="variable", y="value", size=2, color=".6", ax=ax)

        # Add in points to show each observation

        # seaborn.swarmplot(x="variable", y="value", data=df,ax=ax,palette=colors)
        ax.set_title(f'{model_names[i]}', fontsize=8)
        ax.set_ylabel('Sentence alignment', fontsize=8)
        ax.set_xlabel('')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Ds_min', 'Ds_rand', 'Ds_max'], fontsize=8, rotation=90)
        # turn off spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #stat, p_value = mannwhitneyu(x_min_[i], x_rand_[i])
        #print(f"{model_names[i]} min to Rand Mann-Whitney U Test: statistic={stat}, p-value={p_value:.8f}")
        stat, p_value = mannwhitneyu(x_min_[i], x_max_[i])
        print(f"{model_names[i]} min to Max Mann-Whitney U Test: statistic={stat}, p-value={p_value:.8f}")
        #stat, p_value = mannwhitneyu(x_rand_[i], x_max_[i])
        #print(f"{model_names[i]} Rand to Max Mann-Whitney U Test: statistic={stat}, p-value={p_value:.8f}")
        # turn off ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
    plt.tight_layout()
    fig.show()
    # create a figure title
    ax_title = f'S_align_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}_norm'

    #ax_title = f'sent_alignment,{ext_sh},{optim_sh}'
    # save the figure
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    #%%
    X_rands_many = []
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        x_rand_many = []
        for XY_ in optimizer_obj.XY_corr_list:
            pairs = torch.combinations(torch.tensor(sent_random), with_replacement=False)
            X_sample = XY_[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            x_rand_many.append(X_sample)
        X_rands_many.append(x_rand_many)

    X_rands_many_vec = []
    for X_rand in X_rands_many:
        X_rands_many_vec.append([X[np.tril_indices(X.shape[0], k=-1)] for X in X_rand])
    # %
#%%
    for idx in range(len(X_Max)):
        x_max=np.asarray(x_max_[idx])
        x_min= np.asarray(x_min_[idx])
        x_rand_vec= np.stack([X_rand[idx] for X_rand in X_rands_many_vec])
        # compute the correlation between x_max and each row of x_rand_vec
        max_to_rand_coeff = []
        for x in x_rand_vec:
            max_to_rand_coeff.append(np.corrcoef(x_max,x)[0,1])
        max_to_rand_coeff = np.asarray(max_to_rand_coeff)
        # compute pairwise correlation between x_rand_vec rows
        rand_coeff = np.corrcoef(x_rand_vec)
        rand_coef_vec = rand_coeff[np.tril_indices(rand_coeff.shape[0], k=-1)]
        # check if max_to_rand_coeff and rand_coeff come from same distribution
        mw_stat, mw_p = mannwhitneyu(max_to_rand_coeff, rand_coef_vec)
        print(f"{model_names_new[idx]} Mann-Whitney U Test: statistic={mw_stat}, p-value={mw_p}")

        # do the same for min

        min_to_rand_coeff = []
        for x in x_rand_vec:
            min_to_rand_coeff.append(np.corrcoef(x_min,x)[0,1])
        min_to_rand_coeff = np.asarray(min_to_rand_coeff)
        # check if max_to_rand_coeff and rand_coeff come from same distribution
        mw_stat, mw_p = mannwhitneyu(min_to_rand_coeff, rand_coef_vec)
        print(f"{model_names_new[idx]} Mann-Whitney U Test: statistic={mw_stat}, p-value={mw_p}")

    #%% compute the correlation between x_max and x_rand
    X_rands_many = []
    for k in tqdm(enumerate(range(500))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        x_rand_many = []
        for XY_ in optimizer_obj.XY_corr_list:
            pairs = torch.combinations(torch.tensor(sent_random), with_replacement=False)
            X_sample = XY_[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            x_rand_many.append(X_sample)
        X_rands_many.append(x_rand_many)

    X_rands_many_vec = []
    for X_rand in X_rands_many:
        X_rands_many_vec.append([X[np.tril_indices(X.shape[0], k=-1)] for X in X_rand])
    #%
    from sklearn.preprocessing import StandardScaler
    import scipy.stats as stats

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
        print(f"{model_names[idx]} Rand to Rand ratio: {r_rand_to_rand}")
        r_min_to_rand = sum([x[1] > 0.05 for x in min_to_rand_sim]) / len(min_to_rand_sim)
        print(f"{model_names[idx]} Min to Rand ratio: {r_min_to_rand}")
        r_max_to_rand = sum([x[1] > 0.05 for x in max_to_rand_sim]) / len(max_to_rand_sim)
        print(f"{model_names[idx]} Max to Rand ratio: {r_max_to_rand}")



    #%

    jsd_range = []
    js_min_range = []
    js_max_range = []
    for kk in tqdm(range(500)):
        S = np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False)
        # compute objective function for the random sample
        ds_r, _, jsds = optimizer_obj.gpu_object_function_ds_grp_jsd(S, debug=True)
        _, _, jsds_min = optimizer_obj.gpu_object_function_ds_grp_jsd(ds_min_loc, debug=True)
        _, _, jsds_max = optimizer_obj.gpu_object_function_ds_grp_jsd(ds_max_loc, debug=True)
        js_min_range.append(jsds_min)
        js_max_range.append(jsds_max)
        jsd_range.append(jsds)

    jsd_range = np.stack(jsd_range)
    js_min_range = np.stack(js_min_range)
    js_max_range = np.stack(js_max_range)

    # create a figure with 7 panels and each one plot a histogram of jsd_rand columns
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio = 8 / 11
    for kk in range(7):
        ax = plt.axes((.1, .7 * (1 - kk / 7), .4, .06))
        modl_jsd_rand = jsd_range[:, kk]
        modl_min = js_min_range[:, kk]
        modl_max = js_max_range[:, kk]
        # find the max across all
        max_jsd = np.max([modl_jsd_rand.max(), modl_min.max(), modl_max.max()])
        # create edges from 0 to max_jsd
        edges = np.linspace(0, max_jsd, 50)
        # plot histograms
        ax.hist(modl_jsd_rand, bins=edges, color=colors[1], alpha=0.5, label='rand')
        ax.hist(modl_min, bins=edges, color=colors[0], alpha=0.5, label='min')
        ax.hist(modl_max, bins=edges, color=colors[2], alpha=0.5, label='max')
        # add model id
        ax.set_title(model_names[kk])
        # if the last one show the legend
        if kk == 6:
            ax.legend()

    fig.show()
    fig.savefig(
        os.path.join(ANALYZE_DIR,
                     f"S_jsd_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_norm.png"))
    # save as eps
    fig.savefig(
        os.path.join(ANALYZE_DIR,
                     f"S_jsd_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_norm.eps"))


