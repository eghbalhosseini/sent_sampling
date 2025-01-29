import glob
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sent_sampling.utils.data_utils import RESULTS_DIR,ANALYZE_DIR,LEX_PATH_SET
from sent_sampling.utils import extract_pool
import pickle
from neural_nlp.models import model_pool, model_layers
import fnmatch
import re
from sent_sampling.utils.extract_utils import model_extractor_parallel
from sent_sampling.utils.optim_utils import optim_pool
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import torch
import matplotlib
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
if __name__ == '__main__':
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=False'

    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    ext_obj()
    # extract ev sentences
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))
    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]
    sentences = [x[1] for x in ext_obj.model_group_act[0]['activations']]
    # find location of ev sentences in sentences
    ev_sentence_ids = []
    for ev_sent in ev_sentences:
        # remove the period
        ev_sent = ev_sent[:-1]
        ev_sentence_ids.append(sentences.index(ev_sent))

    #optim_file = os.path.join(RESULTS_DIR,
    #                          f"results_{ds_t['src']}_{ds_t['optim'].replace('-run_gpu=False', '')}.pkl")
    #res = load_obj(optim_file)
    optim_id='coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=200-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    optim_obj = optim_pool[optim_id]()

    optim_obj.load_extractor(ext_obj)

    low_resolution= False
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=False,
                                                 save_results=True)

    DS_max,RDM_max=optim_obj.gpu_object_function_debug(ev_sentence_ids)
    DS_max=2-DS_max
    optim_obj.s_init=None
    #S_opt_d, DS_opt_d = optim_obj()

    if  isinstance(RDM_max, torch.Tensor):
         RDM_max = RDM_max.cpu().numpy()
    RDM_max=2-RDM_max
    ds_rand = []
    RDM_rand = []
    sent_rand_ids = []
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optim_obj.N_S, optim_obj.N_s))
        sent_rand_ids.append(sent_random)
        d_s_r, RDM_r = optim_obj.gpu_object_function_debug(sent_random)
        ds_rand.append(d_s_r)
        RDM_rand.append(RDM_r)
    ds_rand = 2 - np.asarray(ds_rand)
    RDM_rand = [2 -x for x in  RDM_rand]
    # get model names
    model_names = optim_obj.extractor_obj.model_spec
    # create a new model order
    model_names_new_order=[0,2,5,3,1,6,4]
    model_names_new = [model_names[i] for i in model_names_new_order]
    # create a new RDM_max based on the new order
    # first make sure RDM_max is symmetric and if not copy upper triangle to lower triangle
    RDM_max = np.triu(RDM_max, k=1).T + np.triu(RDM_max, k=1)
    # do the same for RDM_rand
    RDM_rand = [torch.triu(x,diagonal=1).T + torch.triu(x, diagonal=1) for x in RDM_rand]

    optim_obj.XY_corr_list[0]

    # create a dictionary for max RDM based on model names
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

    save_path=Path(ANALYZE_DIR,'ANNSet1', 'RDM_all_dict_sep2024.pkl')
    with open(save_path.__str__(), 'wb') as f:
          pickle.dump(RDM_full_dict, f)

    RDM_max_new = RDM_max[model_names_new_order, :]
    RDM_max_new = RDM_max_new[:, model_names_new_order]
    # create a new RDM_rand based on the new order
    RDM_rand_new = [x[model_names_new_order, :] for x in RDM_rand]
    RDM_rand_new = [x[:, model_names_new_order] for x in RDM_rand_new]
    # take an
    # replace RDM_max with RDM_max_new
    RDM_max = RDM_max_new
    # replace RDM_rand with RDM_rand_new
    RDM_rand = RDM_rand_new


    # create a dictionary with figure_1_data
    mask = np.triu(np.ones_like(RDM_max, dtype=bool))
    mask = np.where(mask, np.nan, 1)
    RDM_rand_mean = torch.stack(RDM_rand).mean(0).cpu().numpy()
    RDM_rand_mean=RDM_rand_mean.T
    RDM_rand_mean = np.multiply(RDM_rand_mean, mask)
    rdm_rand_vec = RDM_rand_mean[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_max_vec = RDM_max[np.tril_indices(RDM_max.shape[0], k=-1)]

    # create a list of pairs of model names based on how they got vectorized
    model_pairs = []
    for i in range(len(model_names_new)):
        for j in range(i + 1, len(model_names_new)):
            model_pairs.append((model_names_new[i], model_names_new[j]))
    ANNSet1_ds = {'ds_rand': rdm_rand_vec, 'RDM_rand': RDM_rand_mean, 'ds_max': rdm_max_vec, 'RDM_max': RDM_max,'model_pairs': model_pairs, 'model_names': model_names_new,'rand_set_ids': sent_rand_ids}
    # get lexical features for the sentences
    lex_names = [x['name'] for x in LEX_PATH_SET]
    # create an emty dictionary of lex_names
    lex_dict = {lex_name: [] for lex_name in lex_names}
    # for each key in lex_dict, get the lexical feature for each sentence
    for lex_name in lex_dict.keys():
        lex_values = [np.nanmean(sent_dat[lex_name]) for sent_dat in ext_obj.data_]
        lex_dict[lex_name] = lex_values
    # add sentence, words, and word length to lex_dict
    lex_dict['text'] = [sent_dat['text'] for sent_dat in ext_obj.data_]
    lex_dict['sentence_length'] = [sent_dat['sentence_length'] for sent_dat in ext_obj.data_]
    lex_dict['word_string'] = [sent_dat['word_string'] for sent_dat in ext_obj.data_]
    # create one for ev sentences
    ANNSet1_lex = {lex_name: [] for lex_name in lex_dict.keys()}
    for lex_name in ANNSet1_lex.keys():
        # get the lex_full from the lex_dict
        lex_vals = lex_dict[lex_name]
        lex_values = [lex_vals[id] for id in ev_sentence_ids]
        ANNSet1_lex[lex_name] = lex_values

    assert([x==ANNSet1_lex['text'][idx] for idx,x in enumerate(list(ev_sentences.values))])
# create a dictionary for random sentences
    ANNSet1_lex_rand = {lex_name: [] for lex_name in lex_dict.keys()}
    # createa random sentence ids
    sent_random = list(np.random.choice(optim_obj.N_S, optim_obj.N_s))
    for lex_name in ANNSet1_lex_rand.keys():
        # get the lex_full from the lex_dict
        lex_vals = lex_dict[lex_name]
        lex_values=[]
        for sent_random in sent_rand_ids:
            lex_value = [lex_vals[id] for id in sent_random]
            lex_values.append(lex_value)
        ANNSet1_lex_rand[lex_name] = lex_values
    # get the actual RDMs for ev sentences and sent_random
    X_Max = []
    S_id = ev_sentence_ids
    for XY_corr in optim_obj.XY_corr_list:
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_Max.append(X_sample)
    X_rand=[]
    #S_id = sent_random
    for XY_corr in optim_obj.XY_corr_list:
        X_samples=[]
        for S_id in sent_rand_ids:
            pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
            X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
        # make squareform matrix
            X_sample = squareform(X_sample)
            X_samples.append(X_sample)
        X_rand.append(X_sample)
    # get the model names from optim_obj
    model_names = optim_obj.extractor_obj.model_spec
    # create a dictionary for max RDM based on model names
    RDM_max_dict = {model_name: [] for model_name in model_names}
    for model_name in RDM_max_dict.keys():
        # get the model id from the model names
        model_id = model_names.index(model_name)
        RDM_max_dict[model_name] = X_Max[model_id]
    # create a dictionary for random RDM based on model names
    RDM_rand_dict = {model_name: [] for model_name in model_names}
    for model_name in RDM_rand_dict.keys():
        # get the model id from the model names
        model_id = model_names.index(model_name)
        RDM_rand_dict[model_name] = X_rand[model_id]
    # save the ANNSet1_ds, ANNSet1_lex, ANNSet1_lex_rand, RDM_max_dict, RDM_rand_dict
    # save_path=Path(ANALYZE_DIR,'ANNSet1', 'ANNSet1_ds_sep2024.pkl')
    # # make sure it exists
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(save_path.__str__(), 'wb') as f:
    #      pickle.dump(ANNSet1_ds, f)
    # save_path=Path(ANALYZE_DIR,'ANNSet1', 'UD_lex_sep2024.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #      pickle.dump(lex_dict, f)
    # save_path=Path(ANALYZE_DIR,'ANNSet1', 'ANNSet1_lex_max_sep2024.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #      pickle.dump(ANNSet1_lex, f)
    # save_path=Path(ANALYZE_DIR,'ANNSet1', 'ANNSet1_lex_rand_set_sep2024.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #      pickle.dump(ANNSet1_lex_rand, f)
    # save_path=Path(ANALYZE_DIR,'ANNSet1', 'RDM_max_dict_sep2024.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #      pickle.dump(RDM_max_dict, f)
    # save_path=Path(ANALYZE_DIR,'ANNSet1', 'RDM_rand_dict_sep2024.pkl')
    # with open(save_path.__str__(), 'wb') as f:
    #      pickle.dump(RDM_rand_dict, f)

    save_path=Path(ANALYZE_DIR,'ANNSet1', 'ANNSet1_ds_sep2024.pkl')
    ANNSet1_ds=pd.read_pickle(save_path)
    save_path = Path(ANALYZE_DIR, 'ANNSet1', 'UD_lex_sep2024.pkl')
    lex_dict = pd.read_pickle(save_path)
    save_path = Path(ANALYZE_DIR, 'ANNSet1', 'ANNSet1_lex_max_sep2024.pkl')
    ANNSet1_lex = pd.read_pickle(save_path)
    save_path = Path(ANALYZE_DIR, 'ANNSet1', 'ANNSet1_lex_rand_set_sep2024.pkl')
    ANNSet1_lex_rand = pd.read_pickle(save_path)
    save_path = Path(ANALYZE_DIR, 'ANNSet1', 'RDM_max_dict_sep2024.pkl')
    RDM_max_dict = pd.read_pickle(save_path)
    save_path = Path(ANALYZE_DIR, 'ANNSet1', 'RDM_rand_dict_sep2024.pkl')
    RDM_rand_dict = pd.read_pickle(save_path)
    # recompute RDM_rand_set dict
    # read ANNset1_ds keys onto variables
    ds_rand = ANNSet1_ds['ds_rand']
    ds_max= ANNSet1_ds['ds_max']
    RDM_rand_mean = ANNSet1_ds['RDM_rand']
    RDM_max = ANNSet1_ds['RDM_max']
    model_pairs = ANNSet1_ds['model_pairs']
    model_names = ANNSet1_ds['model_names']

    ## reset defaults
    plt.rcdefaults()
    ## Set up LaTeX fonts
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 6,
    })
    y_lim=(.6,2)
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.2, .7, .08, .25))
    ax.scatter(.02 * np.random.normal(size=(np.asarray(len(ds_rand)))) + 0,np.asarray(ds_rand),
               color=(.6, .6, .6), s=2, alpha=.3)
    rand_mean = np.asarray(ds_rand).mean()
    ax.scatter(0, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
               label=f'random= {rand_mean:.4f}', edgecolor='k')
    ax.scatter(0, DS_max, color=np.divide((255, 128, 0), 255), s=50, label=f'Ds_max={DS_max:.4f}', edgecolor='k')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.4, 0.4))
    ax.set_ylim(y_lim)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1.1, .2), frameon=True)
    ax.set_ylabel(r'$D_s$')
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    mask = np.triu(np.ones_like(RDM_max, dtype=bool))
    # change True to nan and false to 1
    mask = np.where(mask, np.nan, 1)

    ax = plt.axes((.6, .73, .25, .25))
    #RDM_rand_mean = torch.stack(RDM_rand).mean(0).cpu().numpy()
    #RDM_rand_mean=RDM_rand_mean.T
    #RDM_rand_mean = np.multiply(RDM_rand_mean, mask)
    #im = ax.imshow(RDM_rand_mean, cmap='viridis', vmax=np.nanmax(RDM_max),vmin=0)

    im = ax.imshow(RDM_rand_mean, cmap='viridis', vmax=1.6, vmin=.6)
    # add values to image plot
    for i in range(RDM_rand_mean.shape[0]):
        for j in range(RDM_rand_mean.shape[1]):
            text = ax.text(j, i, f"{RDM_rand_mean[i, j]:.2f}",
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_title('RDM_rand')
    # set ytick labels to ext_obj.model_spec
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax = plt.axes((.6, .4, .25, .25))
    # transpose the RDM_max and set the upper triangle to nan
    # if RDM_max is on on torch move to cpu
    if isinstance(RDM_max, torch.Tensor):
        RDM_max = RDM_max.cpu().numpy()

    RDM_max = np.triu(RDM_max, k=1).T + np.triu(RDM_max, k=1)
    # change the diagonal to nan
    np.fill_diagonal(RDM_max, np.nan)
    # create an upper triangle mask
    # multiply the RDM_max with the mask
    RDM_max = RDM_max * mask

    # change the upper triangle to nan
    # add values to image plot
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white', 1.)
    #im = ax.imshow(RDM_max, cmap=cmap, vmin=0,vmax=np.nanmax(RDM_max))
    im = ax.imshow(RDM_max, cmap=cmap, vmin=.6, vmax=1.6)
    # add lower triangle of RDM_max to image plot
    for i in range(RDM_max.shape[0]):
        for j in range(RDM_max.shape[1]):
            text = ax.text(j,i, f'{RDM_max[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax.set_title('RDM_max')
    ax = plt.axes((.9, .05, .01, .25))
    plt.colorbar(im, cax=ax)

    rdm_rand_vec = RDM_rand_mean[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_max_vec = RDM_max[np.tril_indices(RDM_max.shape[0], k=-1)]

    # plot rdm vectors connecting points from rdom_rand to rdm max to rdm min
    # fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.1, .05, .1, .25))
    color_set = [ np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255)]
    rdm_vec = np.vstack(( rdm_rand_vec, rdm_max_vec))
    # plot one line per column in rdm_vec
    for i in range(rdm_vec.shape[1]):
        ax.plot([1, 2], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5, zorder=1)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2], rdm_vec[:, i], color=color_set, s=10, marker='o', alpha=.8, zorder=2)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels([ 'ds_rand', 'ds_max'], fontsize=8)
    ax.set_ylabel('Ds')
    ax.set_ylim(y_lim)
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 2.25))

    fig.show()
    save_path = Path(ANALYZE_DIR)


    # save_loc = Path(save_path.__str__(),  f'ANNSet1_Ds_{extract_id}_v2.png')
    # fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
    #             facecolor='auto',
    #             edgecolor='auto', backend=None)
    # save_loc = Path(save_path.__str__(),  f'ANNSet1_Ds_{extract_id}_v2.eps')
    # fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
    #             facecolor='white',
    #             edgecolor='white')


# do a comparison of RDM for max and rands
#     X_rands_many= []
#     for k in tqdm(enumerate(range(200))):
#         sent_random = list(np.random.choice(optim_obj.N_S, optim_obj.N_s))
#         x_rand_many = []
#         for XY_corr in optim_obj.XY_corr_list:
#             pairs = torch.combinations(torch.tensor(sent_random), with_replacement=False)
#             X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
#             # make squareform matrix
#             X_sample = squareform(X_sample)
#             x_rand_many.append(X_sample)
#         X_rands_many.append(x_rand_many)

    X_rands_many_vec = []
    for X_rand in X_rands_many:
        X_rands_many_vec.append([X[np.tril_indices(X.shape[0], k=-1)] for X in X_rand])

    X_Max_vec = []
    for X_max in X_Max:
        X_Max_vec.append(X_max[np.tril_indices(X_max.shape[0], k=-1)])

    for idx in range(len(X_Max_vec)):
        x_max=np.asarray(X_Max_vec[idx])
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

        # Kolmogorov-Smirnov Test
        # ks_stat, ks_p = ks_2samp(max_to_rand_coeff, rand_coef_vec)
        # print(f"{model_names_new[idx]} Kolmogorov-Smirnov Test: statistic={ks_stat}, p-value={ks_p}")


        # plot max_to_rand_coeff and rand_coeff
        # fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
        # ax = plt.axes((.1, .05, .1, .25))
        # ax.boxplot([max_to_rand_coeff, rand_coef_vec], vert=True, showfliers=False, showmeans=False,
        #            meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
        # ax.set_xticklabels(['max_to_rand', 'rand'], fontsize=8)
        # ax.set_ylabel('Correlation')
        # #ax.set_ylim((0, 1))
        # ax.set_title(f'Correlation of RDM for {model_names_new[idx]}')
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.set_xlim((.75, 2.25))
        # fig.show()

    from scipy.stats import kstest
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    for idx in range(len(X_Max_vec)):
        x_max_mdl = np.asarray(X_Max_vec[idx])

        x_rand_vec = np.stack([X_rand[idx] for X_rand in X_rands_many_vec])
        x_ran_mdl= x_rand_vec[0,:]
        x_rand_vec=x_rand_vec[1:,:]
        # compute the correlation between x_max and each row of x_rand_vec
        max_to_rand_sim = []
        rand_to_rand_sim = []
        data_standardized = scaler.fit_transform(x_max_mdl.reshape(-1, 1)).flatten()
        stat, p = kstest(data_standardized, 'norm')
        print(f" MAX Kolmogorov-Smirnov Test Statistic: {stat}, p-value: {p}")


        for x in tqdm(x_rand_vec):
            mw_stat, mw_p = mannwhitneyu(x_max_mdl, x)
            max_to_rand_sim.append([mw_stat, mw_p])
            mw_stat, mw_p = mannwhitneyu(x_ran_mdl, x)
            rand_to_rand_sim.append([mw_stat, mw_p])


        r_rand_to_rand = sum([x[1] > 0.05 for x in rand_to_rand_sim]) / len(rand_to_rand_sim)
        print(f"{model_names[idx]} Rand to Rand ratio: {r_rand_to_rand}")
        r_max_to_rand = sum([x[1] > 0.05 for x in max_to_rand_sim]) / len(max_to_rand_sim)
        print(f"{model_names[idx]} Max to Rand ratio: {r_max_to_rand}")


