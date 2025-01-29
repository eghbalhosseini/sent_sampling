import os
import numpy as np
import sys
from pathlib import Path
import getpass
if getpass.getuser() == 'eghbalhosseini':
    SAMPLING_PARENT = '/Users/eghbalhosseini/MyCodes/sent_sampling'
    SAMPLING_DATA = '/Users/eghbalhosseini/MyCodes//fmri_DNN/ds_parametric/'
    sys.path.append('/Users/eghbalhosseini/MyCodes/DeepJuiceDev/')
    image_paths = '/Users/eghbalhosseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/Users/eghbalhosseini/MyData/DeepJuice/workspace/nsd/'
    benchmark_path = '/Users/eghbalhosseini/MyData/DeepJuice/nsd_data/'

elif getpass.getuser() == 'ehoseini':
    SAMPLING_PARENT = '/om/user/ehoseini/sent_sampling'
    SAMPLING_DATA = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'
    sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
    image_paths = '/om2/user/ehoseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/om2/user/ehoseini/MyData/neural_nlp_bench/activations/DeepJuice_DsParametricfMRI/'
    benchmark_path = '/om2/user/ehoseini/MyData/DeepJuice/nsd_data/'

deepjuice_path='/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
from scipy.stats import median_abs_deviation as mad
from benchmarks import NSDBenchmark, NSDSampleBenchmark
from deepjuice._backends.cupyfy import convert_to_tensor
import multiprocessing
#sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
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
matplotlib.rcParams.update({ 'font.size': 8,'font.weight':'bold'})
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
    deepjuice_identifier=f'group=deepjuice_brains-dataset=nsd-{extract_mode}-bench=None-ave=False'
    ext_obj.identifier=deepjuice_identifier
    selected_models=['subject_1',
                     'subject_2',
                     'subject_3',
                     'subject_4']

    # %%
    benchmark_ = NSDBenchmark(path_dir=benchmark_path)
    x_fmri = (benchmark_.response_data.to_numpy()).T
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    rois = roi_indices.keys()
    roi = 'OTC'
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices[roi].values()]

    ext_obj.model_spec=selected_models
    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
    for idx, model_ in enumerate(selected_models):
        layer_id = 'OTC'
        act_ = fmri_roi_sub_x[idx]
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
    model_deepJuice=f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'

    [ext_id, opt_id_max] = make_shorthand(model_deepJuice, optimizer_id_max)
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
    [ext_id, opt_id_min] = make_shorthand(model_deepJuice, optimizer_id_min)
    optim_file = os.path.join(RESULTS_DIR,
                              f'res_{ext_id}_{opt_id_min}_kl_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_norm.pkl')
    if os.path.exists(optim_file):
        modification_time = os.path.getmtime(optim_file)
        modification_date = time.ctime(modification_time)
        print(f"The file was last modified on: {modification_date}")
        optim_min = load_obj(optim_file)
    else:
        print(f"File {optim_file} not found")

    optim_id_random = f'coordinate_ascent_eh-obj=D_s_rand_kl_div-n_iter=50-n_samples=80-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    (extract_short_hand, optim_short_rand) = make_shorthand(model_deepJuice, optim_id_random)
    ds_rand_path = f'{RESULTS_DIR}/res_{extract_short_hand}_{optim_short_rand}_{extract_mode}_kl_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_norm.pkl'
    # if path ds_rand_path exists, load it
    if Path(ds_rand_path).exists():
        with open(ds_rand_path, 'rb') as f:
            results_ds_rand = pickle.load(f)

    ds_rand_loc = results_ds_rand['optimized_S']
    ds_min_loc=optim_min['optimized_S']
    ds_max_loc=optim_max['optimized_S']
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

    #%% create a ds_rand condition

        
    #%%
    d_all_loc = list(
        set(np.arange(0, 1000)) - set(ds_min_loc) - set(ds_rand_loc) - set(ds_max_loc))
    # make sure d_id_leftout and ds_min_image_ids dont share any elements
    assert len(set(d_all_loc).intersection(set(ds_min_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_rand_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_max_loc))) == 0

    d_s_min, RDM_min = optimizer_obj.gpu_object_function_debug(ds_min_loc)
    [ds_kl_min, _, kl_div_min] = optimizer_obj.gpu_object_function_ds_kl_div(ds_min_loc, debug=True)
    d_s_rand, RDM_rand = optimizer_obj.gpu_object_function_debug(ds_rand_loc)
    [ds_kl_rand, _, kl_div_rand] = optimizer_obj.gpu_object_function_ds_kl_div(ds_rand_loc, debug=True)
    d_s_max, RDM_max = optimizer_obj.gpu_object_function_debug(ds_max_loc)
    [ds_kl_max, _, kl_div_max] = optimizer_obj.gpu_object_function_ds_kl_div(ds_max_loc, debug=True)

    RDM_min = 2 - RDM_min.cpu()
    RDM_rand = 2 - RDM_rand.cpu()
    RDM_max = 2 - RDM_max.cpu()
    RDM_max = np.triu(RDM_max, k=1).T + np.triu(RDM_max, k=1)
    RDM_min = np.triu(RDM_min, k=1).T + np.triu(RDM_min, k=1)
    RDM_rand = np.triu(RDM_rand, k=1).T + np.triu(RDM_rand, k=1)
    rdm_rand_vec = RDM_rand[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_max_vec = RDM_max[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_min_vec = RDM_min[np.tril_indices(RDM_max.shape[0], k=-1)]
    RDM_Mix = np.triu(RDM_min) + np.tril(RDM_max)
    model_names_sh = ['sub_1', 'sub_2', 'sub_3', 'sub_4']
    #


    figure_3_data = {'RDM_max': RDM_max, 'RDM_min': RDM_min, 'RDM_rand': RDM_rand, 'rdm_rand_vec': rdm_rand_vec,
                        'rdm_max_vec': rdm_max_vec, 'rdm_min_vec': rdm_min_vec,
                        'model_names': selected_models}

    (ext_sh, optim_sh) = make_shorthand(model_deepJuice, optimizer_id)
    ax_title = f'ds_data,{ext_sh}_{optim_sh}_model_to_brain_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}'
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'{ax_title}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path.__str__(), 'wb') as f:
         pickle.dump(figure_3_data, f)

    #%%
    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8 / 11
    ax = plt.axes((.1, .1, .15, .3 * pap_ratio))
    rdm_vec = np.vstack((rdm_min_vec, rdm_rand_vec, rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2, 3], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5, zorder=1)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2, 3], rdm_vec[:, i], color=colors, s=10, marker='o', alpha=1, zorder=2, edgecolors='k',
                   linewidths=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'}, zorder=1)
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['min', 'rand', 'max'], fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim((.6, 2.0))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 3.25))

    # ax.violinplot([0,1,2],rdm_vec.transpose(),showmeans=True,showextrema=False,showmedians=False)
    ax = plt.axes((.4, .1, .3, .3 * pap_ratio))
    im = ax.imshow(RDM_Mix, cmap='viridis', vmin=.6, vmax=1.8)
    # add values to image plot
    for i in range(RDM_Mix.shape[0]):
        for j in range(RDM_Mix.shape[1]):
            text = ax.text(j, i, f'{RDM_Mix[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(model_names_sh, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(model_names_sh, fontsize=6, rotation=90)
    # set
    # turn of greed

    ax.grid(False)
    save_path = Path(ANALYZE_DIR)
    (ext_sh, optim_sh) = make_shorthand(model_deepJuice, optimizer_id)
    fig.show()

    ax_title = f'ds,{ext_sh}_{optim_sh}_model_to_brain_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}'
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                 facecolor='auto', edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                 edgecolor='auto', backend=None)
    
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
    # save_path = Path(ANALYZE_DIR)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    # fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    # #
    # RDM_min_dict = {model_name: [] for model_name in selected_models}
    # for idx, x in enumerate(X_Min):
    #     RDM_min_dict[selected_models[idx]] = x
    # RDM_max_dict = {model_name: [] for model_name in selected_models}
    # for idx, x in enumerate(X_Max):
    #     RDM_max_dict[selected_models[idx]] = x
    # RDM_rand_dict = {model_name: [] for model_name in selected_models}
    # for idx, x in enumerate(X_rand):
    #     RDM_rand_dict[selected_models[idx]] = x
    # # save the RDMs
    # save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_max_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #     pickle.dump(RDM_max_dict, f)
    # save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_min_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #     pickle.dump(RDM_min_dict, f)
    # save_path = Path(ANALYZE_DIR, 'DsParametric',
    #                  f'RDM_rand_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #     pickle.dump(RDM_rand_dict, f)
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
        #stat, p_value = mannwhitneyu(x_min_[i], x_max_[i])
        #print(f"{model_names[i]} min to Max Mann-Whitney U Test: statistic={stat}, p-value={p_value:.8f}")
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
    # save_path = Path(ANALYZE_DIR)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    # fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)


    #%% compare the ds_score for models
    extract_mode = 'redux'
    model_extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'

    model_ext_obj = extract_pool[model_extract_id]()
    model_deepjuice_identifier = f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    model_ext_obj.identifier = model_deepjuice_identifier
    model_selected_models = ['torchvision_alexnet_imagenet1k_v1',
                       'torchvision_regnet_x_800mf_imagenet1k_v2',
                       'openclip_vit_b_32_laion2b_e16',
                       'timm_swinv2_cr_tiny_ns_224',
                       'torchvision_efficientnet_b1_imagenet1k_v2',
                       'clip_rn50',
                       'timm_convnext_large_in22k',
                       ]
    model_ext_obj.model_spec = model_selected_models
    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
    for model_ in tqdm(model_selected_models):
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

    optimizer_id = "coordinate_ascent_eh-obj=2-D_s_kl_div-n_iter=50-n_samples=80-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    model_optimizer_obj = optim_pool[optimizer_id]()
    model_optimizer_obj.N_S = 1000
    model_optimizer_obj.extract_type = 'activation'
    model_optimizer_obj.activations = activations_list
    model_optimizer_obj.extractor_obj = model_ext_obj
    model_optimizer_obj.early_stopping = False

    model_optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                             save_results=False)


    model_d_s_min, model_RDM_min = model_optimizer_obj.gpu_object_function_debug(ds_min_loc)
    #[model_ds_kl_min, _, model_kl_div_min]=model_optimizer_obj.gpu_object_function_ds_kl_div(ds_min_loc, debug=True)
    model_d_s_rand, model_RDM_rand = model_optimizer_obj.gpu_object_function_debug(ds_rand_loc)
    #[model_ds_kl_rand, _, model_kl_div_rand]=model_optimizer_obj.gpu_object_function_ds_kl_div(ds_rand_loc, debug=True)
    model_d_s_max, model_RDM_max = model_optimizer_obj.gpu_object_function_debug(ds_max_loc)
    #[model_ds_kl_max, _, model_kl_div_max]=model_optimizer_obj.gpu_object_function_ds_kl_div(ds_max_loc, debug=True)
    model_ds_rand = []
    model_RDM_rand_set = []

    for k in tqdm(enumerate(range(1000))):
        sent_random = list(np.random.choice(model_optimizer_obj.N_S, model_optimizer_obj.N_s,replace=False))
        d_s_r, RDM_r = model_optimizer_obj.gpu_object_function_debug(sent_random)
        model_ds_rand.append(d_s_r)
        model_RDM_rand_set.append(RDM_r)

    model_RDM_min = 2 - model_RDM_min.cpu()
    model_RDM_rand = 2 - model_RDM_rand.cpu()
    model_RDM_max = 2 - model_RDM_max.cpu()
    model_RDM_rand_set = [2 - RDM.cpu() for RDM in model_RDM_rand_set]
    model_RDM_max = np.triu(model_RDM_max, k=1).T + np.triu(model_RDM_max, k=1)
    model_RDM_min = np.triu(model_RDM_min, k=1).T + np.triu(model_RDM_min, k=1)
    model_RDM_rand = np.triu(model_RDM_rand, k=1).T + np.triu(model_RDM_rand, k=1)
    model_RDM_rand_set=[np.triu(RDM, k=1).T + np.triu(RDM, k=1) for RDM in model_RDM_rand_set]
    rdm_rand_vec = model_RDM_rand[np.tril_indices(model_RDM_max.shape[0], k=-1)]
    rdm_max_vec = model_RDM_max[np.tril_indices(model_RDM_max.shape[0], k=-1)]
    rdm_min_vec = model_RDM_min[np.tril_indices(model_RDM_max.shape[0], k=-1)]
    model_rdm_rand_vec_set=[RDM[np.tril_indices(RDM.shape[0], k=-1)] for RDM in model_RDM_rand_set]
    RDM_Mix = np.triu(model_RDM_min) + np.tril(model_RDM_max)
    model_names_sh = ['AlexNet','RegNet','ViT','Swin','EfficientNet','clip','convnext']
    model_rdm_rand_vec_set=np.mean(np.stack(model_rdm_rand_vec_set),axis=0)

    colors_extra=np.vstack([np.asarray(colors),np.array([[0.625, 0.625, 0.625]])])
    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8 / 11
    ax = plt.axes((.1, .1, .15, .3 * pap_ratio))
    rdm_vec = np.vstack((rdm_min_vec, rdm_rand_vec, rdm_max_vec,model_rdm_rand_vec_set))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2, 3,4], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5, zorder=1)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2, 3,4], rdm_vec[:, i], color=colors_extra, s=10, marker='o', alpha=1, zorder=2, edgecolors='k',
                   linewidths=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'}, zorder=1)

    # ax.scatter(modle_rdm_vec_set_ave*0+4, modle_rdm_vec_set_ave, color=colors[1], s=10, marker='o', alpha=1, zorder=2, edgecolors='k',
    #            linewidths=.5)
    # ax.boxplot(np.expand_dims(modle_rdm_vec_set_ave,axis=1),positions=[4], vert=True, showfliers=False, showmeans=False,
    #            meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'}, zorder=1)

    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['min', 'rand', 'max','rand_model'], fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim((1.3, 2.0))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 4.25))

    # ax.violinplot([0,1,2],rdm_vec.transpose(),showmeans=True,showextrema=False,showmedians=False)
    ax = plt.axes((.4, .1, .3, .3 * pap_ratio))
    im = ax.imshow(RDM_Mix, cmap='viridis', vmin=1.4, vmax=1.9)
    # add values to image plot
    for i in range(RDM_Mix.shape[0]):
        for j in range(RDM_Mix.shape[1]):
            text = ax.text(j, i, f'{RDM_Mix[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(model_ext_obj.model_spec)))
    ax.set_yticklabels(model_names_sh, fontsize=6)
    ax.set_xticks(np.arange(len(model_ext_obj.model_spec)))
    ax.set_xticklabels(model_names_sh, fontsize=6, rotation=90)
    # set
    # turn of greed
    ax.grid(False)
    save_path = Path(ANALYZE_DIR)
    (ext_sh, optim_sh) = make_shorthand(extract_id, optimizer_id)
    fig.show()

    # ax_title = f'ds,{ext_sh}_{optim_sh}_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}'
    # save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
    #             facecolor='auto', edgecolor='auto', backend=None)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    # fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
    #             edgecolor='auto', backend=None)

    model_names_new_order = [0, 4, 2, 1, 3, 6, 5]
    model_names_new = [model_selected_models[i] for i in model_names_new_order]

    model_sh_rotated=[model_names_sh[i] for i in model_names_new_order]
    # reorder the RDMs
    model_RDM_max = np.triu(model_RDM_max, k=1).T + np.triu(model_RDM_max, k=1)
    model_RDM_min = np.triu(model_RDM_min, k=1).T + np.triu(model_RDM_min, k=1)
    model_RDM_rand = np.triu(model_RDM_rand, k=1).T + np.triu(model_RDM_rand, k=1)

    model_RDM_max_new = model_RDM_max[model_names_new_order, :]
    model_RDM_max_new = model_RDM_max_new[:, model_names_new_order]
    model_RDM_min_new = model_RDM_min[model_names_new_order, :]
    model_RDM_min_new = model_RDM_min_new[:, model_names_new_order]
    model_RDM_rand_new = model_RDM_rand[model_names_new_order, :]
    model_RDM_rand_new = model_RDM_rand_new[:, model_names_new_order]
    # do the same thing for model_RDM_rand_set
    model_RDM_rand_set_new = [x[model_names_new_order, :] for x in model_RDM_rand_set]
    model_RDM_rand_set_new = [x[:, model_names_new_order] for x in model_RDM_rand_set_new]
    mask = np.triu(np.ones_like(RDM_max, dtype=bool))
    mask = np.where(mask, np.nan, 1)
    rdm_rand_vec = model_RDM_rand_new[np.tril_indices(model_RDM_rand_new.shape[0], k=-1)]
    rdm_max_vec = model_RDM_max_new[np.tril_indices(model_RDM_max_new.shape[0], k=-1)]
    rdm_min_vec = model_RDM_min_new[np.tril_indices(model_RDM_min_new.shape[0], k=-1)]
    rdm_randvec_set = [x[np.tril_indices(model_RDM_rand_new.shape[0], k=-1)] for x in model_RDM_rand_set_new]

    # figure_3_data = {'RDM_max': model_RDM_max_new, 'RDM_min': model_RDM_min_new, 'RDM_rand': model_RDM_rand_new,'RDM_rand_set':model_RDM_rand_set_new,
    #                  'rdm_rand_vec': rdm_rand_vec,'rdm_max_vec': rdm_max_vec, 'rdm_min_vec': rdm_min_vec,'rdm_rand_vec_set':model_rdm_rand_vec_set,
    #                  'model_names': model_names_new}
    # (ext_sh, optim_sh) = make_shorthand(deepjuice_identifier, optimizer_id)
    # ax_title = f'ds_data_models,{ext_sh}_{optim_sh}_kl_div_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}'
    # save_path = Path(ANALYZE_DIR, 'DsParametric', f'{ax_title}.pkl')
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(save_path.__str__(), 'wb') as f:
    #     pickle.dump(figure_3_data, f)