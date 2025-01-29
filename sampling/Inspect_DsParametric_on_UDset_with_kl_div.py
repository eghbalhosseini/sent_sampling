import os
import pandas as pd
from tqdm import tqdm
from sent_sampling.utils.data_utils import RESULTS_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, pt_create_corr_rdm_short
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj, ANALYZE_DIR, LEX_PATH_SET
import torch
import numpy as np
from sent_sampling.utils import extract_pool, make_shorthand
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#args = parser.parse_args()
from scipy.stats import mannwhitneyu, ks_2samp
import os
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import torch
from sent_sampling.utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import pickle
import numpy as np
import time
from scipy.stats import ks_2samp
from scipy.stats import shapiro, anderson, kstest, norm, probplot
import seaborn
colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255)]
def find_best_match(target, string_list):
    # Function to count the number of word matches
    def word_match_count(str1, str2):
        set1 = set(str1.split())
        set2 = set(str2.split())
        return len(set1.intersection(set2))

    # Check if the target string is in the list
    if target in string_list:
        return target

    # If target is not found, find the string with the most word matches
    best_match = None
    best_match_count = -1

    for string in string_list:
        count = word_match_count(target, string)
        if count > best_match_count:
            best_match_count = count
            best_match = string

    return best_match


if __name__ == '__main__':
    kl_muliplier = 5.0
    kl_threshold = 0.05
    bins = 200
    epsilon = 1e-10
    n_samples=200
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optimizer_id = f"coordinate_ascent_eh-obj=D_s_kl_div-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    #optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples=225-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    extractor_obj = extract_pool[extract_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    low_resolution = False
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=False,
                                             save_results=False)

    # %%
    optimizer_id_max = f"coordinate_ascent_eh-obj=D_s_kl_div-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    [ext_id, opt_id_max] = make_shorthand(extract_id, optimizer_id_max)
    optim_file = os.path.join(RESULTS_DIR,f'res_{ext_id}_{opt_id_max}_kl_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_norm.pkl')
    # check when the file was created
    if os.path.exists(optim_file):
        modification_time = os.path.getmtime(optim_file)
        modification_date = time.ctime(modification_time)
        print(f"The file was last modified on: {modification_date}")
        optim_max = load_obj(optim_file)
    else:
        print(f"File {optim_file} not found")
    # load optim_max
    optimizer_id_min = f"coordinate_ascent_eh-obj=2-D_s_kl_div-n_iter=50-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    [ext_id, opt_id_min] = make_shorthand(extract_id, optimizer_id_min)
    optim_file = os.path.join(RESULTS_DIR,f'res_{ext_id}_{opt_id_min}_kl_thr_{kl_threshold}_mult_{kl_muliplier}_bins_{bins}_norm.pkl')
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
        print(f'model: {extractor_obj.model_spec[i_m]}, min: {vals.min()}, max: {vals.max()}')
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
    for kk in tqdm(range(100)):
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
    ds_min_loc=optim_min['optimized_S']
    ds_max_loc=optim_max['optimized_S']
    #ds_max_dict = pd.read_pickle('/rdma/vast-rdma/vast/evlab/ehoseini/sent_sampling/bash/slurm-37359693_final_list.pkl')
    #ds_min_dict = pd.read_pickle('/rdma/vast-rdma/vast/evlab/ehoseini/sent_sampling/bash/slurm-37359365_final_list.pkl')
    #ds_min_loc = [int(x['number']) for x in ds_min_dict]
    #ds_max_loc = [int(x['number']) for x in ds_max_dict]

    ds_rand_loc = np.random.choice(optimizer_obj.N_S, n_samples, replace=False)
    d_s_min, RDM_min = optimizer_obj.gpu_object_function_debug(ds_min_loc)
    [ds_kl_min, _, kl_div_min]=optimizer_obj.gpu_object_function_ds_kl_div(ds_min_loc, debug=True)
    d_s_rand, RDM_rand = optimizer_obj.gpu_object_function_debug(ds_rand_loc)
    [ds_kl_rand, _, kl_div_rand]=optimizer_obj.gpu_object_function_ds_kl_div(ds_rand_loc, debug=True)
    d_s_max, RDM_max = optimizer_obj.gpu_object_function_debug(ds_max_loc)
    [ds_kl_max, _, kl_div_max]=optimizer_obj.gpu_object_function_ds_kl_div(ds_max_loc, debug=True)


    RDM_min=RDM_min.cpu()
    RDM_rand=RDM_rand.cpu()
    RDM_max = RDM_max.cpu()
    model_names = optimizer_obj.extractor_obj.model_spec
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

    figure_3_data = {'RDM_max': RDM_max, 'RDM_min': RDM_min, 'RDM_rand': RDM_rand, 'rdm_rand_vec': rdm_rand_vec,
                     'rdm_max_vec': rdm_max_vec, 'rdm_min_vec': rdm_min_vec, 'model_pairs': model_pairs,
                     'model_names': model_names_new, 'mask': mask}
    #%
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
    ax.set_yticks(np.arange(len(extractor_obj.model_spec)))
    ax.set_yticklabels(model_names_new, fontsize=6)
    ax.set_xticks(np.arange(len(extractor_obj.model_spec)))
    ax.set_xticklabels(model_names_new, fontsize=6, rotation=90)

    ax = plt.axes((.6, .4, .25, .25 * pap_ratio))
    im = ax.imshow(RDM_max, cmap='viridis', vmax=RDM_max.max())
    # add values to image plot
    for i in range(RDM_max.shape[0]):
        for j in range(RDM_max.shape[1]):
            text = ax.text(j, i, f'{RDM_max[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(extractor_obj.model_spec)))
    ax.set_yticklabels(model_names_new, fontsize=6)
    ax.set_xticks(np.arange(len(extractor_obj.model_spec)))
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
    ax.set_yticks(np.arange(len(extractor_obj.model_spec)))
    ax.set_yticklabels(model_names_new, fontsize=6)
    ax.set_xticks(np.arange(len(extractor_obj.model_spec)))
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
    (ext_sh, optim_sh) = make_shorthand(extract_id, optimizer_id)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}_norm.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}_norm.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)

    #%
    act_dict=extractor_obj.model_group_act[0]
    sentences = [x[1] for x in act_dict['activations']]
    ds_min_sent=[sentences[x] for x in ds_min_loc]
    ds_max_sent=[sentences[x] for x in ds_max_loc]
    ds_rand_sent=[sentences[x] for x in ds_rand_loc]

    data_text = [x['text'] for x in extractor_obj.data_]
    data_textNoPeriod = []
    for x in data_text:
        if '.' in x[-1]:
            data_textNoPeriod.append(x[:-1])
        else:
            data_textNoPeriod.append(x)
    # drop the space in the end if it exist in data_textNoPeriod
    data_textNoPeriod = [x if x[-1] != ' ' else x[:-1] for x in data_textNoPeriod]

    ds_min_loc_in_dat = []
    for i, x in enumerate(ds_min_sent):
        if x not in data_textNoPeriod:
            x_new = find_best_match(x, data_textNoPeriod)
            # print the sentence that is not found in data_textNoPeriod
            print(f'{x} not found in data_textNoPeriod, best match is {x_new}')
            # find the index of x_new in data_textNoPeriod
            ds_min_loc_in_dat.append(data_textNoPeriod.index(x_new))
        else:
            ds_min_loc_in_dat.append(data_textNoPeriod.index(x))

    ds_max_loc_in_dat=[]
    #ds_max_loc_in_dat = [data_textNoPeriod.index(x) for x in ds_max_sent]
    for i, x in enumerate(ds_max_sent):
        if x not in data_textNoPeriod:
            x_new = find_best_match(x, data_textNoPeriod)
            # print the sentence that is not found in data_textNoPeriod
            print(f'{x} not found in data_textNoPeriod, best match is {x_new}')
            # find the index of x_new in data_textNoPeriod
            ds_max_loc_in_dat.append(data_textNoPeriod.index(x_new))
        else:
            ds_max_loc_in_dat.append(data_textNoPeriod.index(x))






    sent_max_data = [extractor_obj.data_[x] for x in ds_max_loc_in_dat]
    sent_min_data = [extractor_obj.data_[x] for x in ds_min_loc_in_dat]

    sent_all_data=extractor_obj.data_

    lex_names = [x['name'] for x in LEX_PATH_SET]
    sent_max_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_max_data]
    sent_min_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_min_data]
    sent_all_lex_values = [[np.nanmean(sent_dat[lex_name]) for lex_name in lex_names] for sent_dat in sent_all_data]
    # add num_words to the beginning of each list
    sent_max_num_words = [len(x['word_id']) for x in sent_max_data]
    sent_min_num_words = [len(x['word_id']) for x in sent_min_data]
    sent_all_num_words = [len(x['word_id']) for x in sent_all_data]

    # add sent_max_num_words to the beginning of each sent_max_lex_values
    sent_max_lex_values = np.concatenate(
        [np.asarray(sent_max_num_words).reshape(-1, 1), np.asarray(sent_max_lex_values)], axis=1)
    sent_min_lex_values = np.concatenate(
        [np.asarray(sent_min_num_words).reshape(-1, 1), np.asarray(sent_min_lex_values)], axis=1)
    sent_all_lex_values = np.concatenate(
        [np.asarray(sent_all_num_words).reshape(-1, 1), np.asarray(sent_all_lex_values)], axis=1)
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



    lex_names = ['num_words'] + lex_names
    assert len(lex_names) == sent_max_lex_values.shape[1]
    # create a figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    axes = axes.flatten()
    for i in range(len(lex_names)):
        # plot a histogram for sent_max_lex_values using seaborn.distplot on axes[i]
        seaborn.distplot(sent_all_lex_values[:, i], bins=50, label='Ds_all', norm_hist=True, hist=False, ax=axes[i],
                         kde_kws={"lw": 3, "color": np.divide([150, 150, 150], 255)})
        seaborn.distplot(sent_max_lex_values[:, i], bins=50, label='Ds_max', norm_hist=True, hist=False, ax=axes[i],
                         kde_kws={"lw": 3, "color": np.divide((255, 128, 0), 255)})
        seaborn.distplot(sent_min_lex_values[:, i], bins=50, label='Ds_min', norm_hist=True, hist=False, ax=axes[i],
                         kde_kws={"lw": 3, "color": np.divide((188, 80, 144), 255)})
        # put tick in the begining and end of x axis
        # axes[i].set_xticks([np.min(sent_all_lex_values[:, i]), np.max(sent_all_lex_values[:, i])])
        if i == (len(lex_names) - 1):
            axes[i].legend(loc='upper right')
        axes[i].set_ylabel(lex_names[i], fontsize=8)
        # remove top and right spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        # turn off yticks values
        axes[i].set_yticks([])

    plt.tight_layout()
    fig.show()
    ax_title = f'S_feat,{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}_norm'

    # add a suptitle to the figure
    fig.suptitle(ax_title, fontsize=10, y=.99)
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # save figure as pdf and png
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    fig.show()

    #%
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
        ax.set_ylabel(f'{extractor_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_max')
        # turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(X_Min)):
        ax = plt.subplot(3, 7, i + 1 + 14)
        im = ax.imshow(X_Min[i], cmap='viridis', vmax=X_Min[i].max())
        ax.set_ylabel(f'{extractor_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_min')
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(X_rand)):
        ax = plt.subplot(3, 7, i + 1)
        im = ax.imshow(X_rand[i], cmap='viridis', vmax=X_rand[i].max())
        ax.set_ylabel(f'{extractor_obj.model_spec[i]}', fontsize=6)
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

    #%
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
    model_names = [x['model_name'] for x in extractor_obj.model_group_act]
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
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)

    #% compute the correlation between x_max and x_rand
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
    model_names = optimizer_obj.extractor_obj.model_spec
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
                     f"kl_{ext_sh}_{optim_sh}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_norm.eps"))

    ##
    # X_Max = []
    # S_id = ds_min_loc
    # for XY_ in optimizer_obj.XY_corr_list:
    #     pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
    #     X_sample = XY_[pairs[:, 0], pairs[:, 1]].cpu().numpy()
    #     # make squareform matrix
    #     X_sample = squareform(X_sample)
    #     X_Max.append(X_sample)
    #
    # X_rands_many = []
    # for k in tqdm(enumerate(range(200))):
    #     sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
    #     x_rand_many = []
    #     for XY_ in optimizer_obj.XY_corr_list:
    #         pairs = torch.combinations(torch.tensor(sent_random), with_replacement=False)
    #         X_sample = XY_[pairs[:, 0], pairs[:, 1]].cpu().numpy()
    #         # make squareform matrix
    #         X_sample = squareform(X_sample)
    #         x_rand_many.append(X_sample)
    #     X_rands_many.append(x_rand_many)
    #
    #
    #
    # X_Max_vec = []
    # for X_max in X_Max:
    #     X_Max_vec.append(X_max[np.tril_indices(X_max.shape[0], k=-1)])
    #
    # for idx in range(len(X_Max_vec)):
    #     x_max = np.asarray(X_Max_vec[idx])
    #     x_rand_vec = np.stack([X_rand[idx] for X_rand in X_rands_many_vec])
    #     # compute the correlation between x_max and each row of x_rand_vec
    #     max_to_rand_coeff = []
    #     for x in x_rand_vec:
    #         max_to_rand_coeff.append(np.corrcoef(x_max, x)[0, 1])
    #     max_to_rand_coeff = np.asarray(max_to_rand_coeff)
    #     # compute pairwise correlation between x_rand_vec rows
    #     rand_coeff = np.corrcoef(x_rand_vec)
    #     rand_coef_vec = rand_coeff[np.tril_indices(rand_coeff.shape[0], k=-1)]
    #     # sample 200
    #     rand_coeff_sample = np.random.choice(rand_coef_vec, 200)
    #     # check if max_to_rand_coeff and rand_coeff come from same distribution
    #     mw_stat, mw_p = mannwhitneyu(max_to_rand_coeff, rand_coef_vec)
    #     print(f"{model_names[idx]} Mann-Whitney U Test: statistic={mw_stat}, p-value={mw_p}")
    #
    #
    #
    # X_Min_vec = []
    # for X_ in X_Min:
    #     X_Min_vec.append(X_[np.tril_indices(X_.shape[0], k=-1)])
    # for idx in range(len(X_Min_vec)):
    #     x_min = np.asarray(X_Min_vec[idx])
    #     x_rand_vec = np.stack([X_rand[idx] for X_rand in X_rands_many_vec])
    #     # compute the correlation between x_max and each row of x_rand_vec
    #     min_to_rand_coeff = []
    #     for x in x_rand_vec:
    #         min_to_rand_coeff.append(np.corrcoef(x_min, x)[0, 1])
    #     max_to_rand_coeff = np.asarray(max_to_rand_coeff)
    #     # compute pairwise correlation between x_rand_vec rows
    #     rand_coeff = np.corrcoef(x_rand_vec)
    #     rand_coef_vec = rand_coeff[np.tril_indices(rand_coeff.shape[0], k=-1)]
    #     # check if max_to_rand_coeff and rand_coeff come from same distribution
    #     mw_stat, mw_p = mannwhitneyu(max_to_rand_coeff, rand_coef_vec)
    #     print(f"{model_names[idx]} Mann-Whitney U Test: statistic={mw_stat}, p-value={mw_p}")
    #
    #

    #%%
