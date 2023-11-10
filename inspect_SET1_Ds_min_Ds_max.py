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
if __name__ == '__main__':
    modelnames = ['roberta-base',  'xlnet-large-cased',  'bert-large-uncased','xlm-mlm-en-2048', 'gpt2-xl', 'albert-xxlarge-v2','ctrl']
    extract_id = [
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False']
    #optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
    #             'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True']

    optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
        ,'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True']
    #
    #optim_id=['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
    #            'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']
    # get group= from extract_id
    group_id = extract_id[0].split('-')[0].split('=')[1]
    dataset_id=extract_id[0].split('-')[1].split('=')[1]
    # get obj= from optim_id
    obj_id = ['Ds_max', '2-Ds_max']
    # get n_samples from each element in optim_id



    low_resolution = 'False'
    optim_files = []
    optim_results = []
    for ext in extract_id:
        for optim in optim_id:
            (ext_sh,optim_sh)=make_shorthand(ext, optim)
            #optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
            optim_file = Path(RESULTS_DIR, f"results_{ext_sh}_{optim_sh}.pkl")
            assert(optim_file.exists())

            optim_files.append(optim_file.__str__())
            optim_results.append(load_obj(optim_file.__str__()))


    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    ext_obj()

    optimizer_obj = optim_pool[optim_id[0]]()
    optimizer_obj.load_extractor(ext_obj)

    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=True,
                                                 save_results=False)
    # get n_samples from optimizer_obj
    n_samples = optimizer_obj.N_s
    # get low_dim from optimizer_obj
    is_low_dim = optimizer_obj.low_dim
    DS_max,RDM_max=optimizer_obj.gpu_object_function_debug(optim_results[0]['optimized_S'])
    DS_min,RDM_min = optimizer_obj.gpu_object_function_debug(optim_results[1]['optimized_S'])

    ds_rand=[]
    RDM_rand=[]
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        d_s_r,RDM_r= optimizer_obj.gpu_object_function_debug(sent_random)
        ds_rand.append(d_s_r)
        RDM_rand.append(RDM_r)

    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ax = plt.axes((.2, .7, .08, .25))
    ax.scatter(.02 * np.random.normal(size=(np.asarray(len(ds_rand)))) + 0,np.asarray(ds_rand),
               color=(.6, .6, .6), s=2, alpha=.3)
    rand_mean = np.asarray(ds_rand).mean()
    ax.scatter(0, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
               label=f'random= {rand_mean:.4f}', edgecolor='k')
    ax.scatter(0, DS_min, color=np.divide((188, 80, 144), 255), s=50, label=f'Ds_min={DS_min:.4f}', edgecolor='k')

    ax.scatter(0, DS_max, color=np.divide((255, 128, 0), 255), s=50, label=f'Ds_max={DS_max:.4f}', edgecolor='k')

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

    dataset=optim_results[0]['extractor_name']
    optim=optim_results[0]['optimizatin_name']
    ax_title = f'ds,{dataset},{optim}'
    ax.set_title(ax_title.replace(',', ',\n'),fontsize=8)

    ax=plt.axes((.6, .73, .25, .25))
    RDM_rand_mean=torch.stack(RDM_rand).mean(0).cpu().numpy()
    im=ax.imshow(RDM_rand_mean, cmap='viridis',vmax=RDM_max.cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_rand_mean.shape[0]):
        for j in range(RDM_rand_mean.shape[1]):
            text = ax.text(j, i, f"{RDM_rand_mean[i, j]:.2f}",
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_title('RDM_rand')
    # set ytick labels to ext_obj.model_spec
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec,fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6,rotation=90)

    ax=plt.axes((.6, .4, .25, .25))
    im=ax.imshow(RDM_max.cpu(), cmap='viridis',vmax=RDM_max.cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_max.shape[0]):
        for j in range(RDM_max.shape[1]):
            text = ax.text(j, i, f'{RDM_max[i, j]:.2f}',
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=6)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax.set_title('RDM_max')

    ax = plt.axes((.6, .05, .25, .25))
    im = ax.imshow(RDM_min.cpu(), cmap='viridis',vmax=RDM_max.cpu().numpy().max())
    # add values to image plot
    for i in range(RDM_min.shape[0]):
        for j in range(RDM_min.shape[1]):
            text = ax.text(j, i, f'{RDM_min[i, j]:.2f}',
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_yticks(np.arange(len(ext_obj.model_spec)))
    ax.set_yticklabels(ext_obj.model_spec, fontsize=8)
    ax.set_xticks(np.arange(len(ext_obj.model_spec)))
    ax.set_xticklabels(ext_obj.model_spec, fontsize=6, rotation=90)

    ax.set_title('RDM_min')
    ax = plt.axes((.9, .05, .01, .25))
    plt.colorbar(im, cax=ax)

    rdm_rand_vec=RDM_rand_mean[np.triu_indices(RDM_min.shape[0], k=1)]
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
    (ext_sh,optim_sh)=make_shorthand(extract_id[0], optim_id[0])
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'ds_{ext_sh}_{optim_sh}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)


    # plot model RDMS
    # for each matrix in optimizer_obj.XY_corr_list select rows and colums based on a list S
    # and plot the resulting matrix
    X_Max= []
    S_id = optim_results[0]['optimized_S']
    for XY_corr in optimizer_obj.XY_corr_list:

        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        # make squareform matrix
        X_sample=squareform(X_sample)
        X_Max.append(X_sample)

    X_Min = []
    S_id = optim_results[1]['optimized_S']
    for XY_corr in optimizer_obj.XY_corr_list:

        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_Min.append(X_sample)

    X_rand = []
    sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
    for XY_corr in optimizer_obj.XY_corr_list:
        S_id = sent_random
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_corr[pairs[:, 0], pairs[:, 1]]
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_rand.append(X_sample)

    # create a figure with 7 rows and 3 columns and plot x_samples in each row

    fig = plt.figure(figsize=(11, 8))
    for i in range(len(X_Max)):
        ax = plt.subplot(3, 7, i + 1+7)
        im = ax.imshow(X_Max[i], cmap='viridis',vmax=X_Max[i].max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}',fontsize=6)
        ax.set_title('Ds_max')
        # turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(X_Min)):
        ax = plt.subplot(3, 7, i + 1+14)
        im = ax.imshow(X_Min[i], cmap='viridis', vmax=X_Min[i].max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_min')
        ax.set_xticks([])
        ax.set_yticks([])


    for i in range(len(X_rand)):
        ax = plt.subplot(3, 7, i+1)
        im = ax.imshow(X_rand[i], cmap='viridis', vmax=X_rand[i].max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_rand')
        ax.set_xticks([])
        ax.set_yticks([])

    #ax = plt.axes((.95, .05, .01, .25))
    #plt.colorbar(im, cax=ax)

    fig.show()

    save_path = Path(ANALYZE_DIR)
    ax_1=ax_title.replace('ds_result','model_rdm')
    save_loc = Path(save_path.__str__(), f'RDMs_{ext_sh}_{optim_sh}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'RDMs_{ext_sh}_{optim_sh}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)

    S_id = optim_results[0]['optimized_S']
    select_sentences=[]
    select_activations=[]
    for model_act in ext_obj.model_group_act:
        select_sentences.append([model_act['activations'][s][1] for s in S_id])
        select_activations.append([model_act['activations'][s][0] for s in S_id])
    df_max = pd.DataFrame(np.asarray(select_sentences).transpose(), index=S_id, columns=ext_obj.model_spec)
    df_act_max= pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id, columns=ext_obj.model_spec)
    dataset = optim_results[0]['extractor_name']
    optim=optim_results[0]['optimizatin_name']
    (ext_sh, optim_sh) = make_shorthand(extract_id[0],optim_id[0])
    ax_title = f'sent,{ext_sh},{optim_sh}'
    df_max.to_csv(Path(ANALYZE_DIR, f'{ax_title}.csv'))

    S_id = optim_results[1]['optimized_S']
    select_activations = []
    select_sentences = []
    for model_act in ext_obj.model_group_act:
        select_sentences.append([model_act['activations'][s][1] for s in S_id])
        select_activations.append([model_act['activations'][s][0] for s in S_id])
    df_min = pd.DataFrame(np.asarray(select_sentences).transpose(), index=S_id, columns=ext_obj.model_spec)
    df_act_min = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id, columns=ext_obj.model_spec)
    dataset = optim_results[1]['extractor_name']
    optim=optim_results[1]['optimizatin_name']
    (ext_sh, optim_sh) = make_shorthand(extract_id[0], optim_id[1])
    ax_title = f'sent,{ext_sh},{optim_sh}'
    df_min.to_csv(Path(ANALYZE_DIR, f'{ax_title}.csv'))

    # create a random set that is closest to the mean of distrubtion
    ds_rand=[]
    RDM_rand=[]
    sent_randoms=[]
    for k in tqdm(enumerate(range(1000))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        d_s_r,RDM_r= optimizer_obj.gpu_object_function_debug(sent_random)
        ds_rand.append(d_s_r)
        RDM_rand.append(RDM_r)
        sent_randoms.append(sent_random)

    ds_rand_mean=np.mean(ds_rand)
    # find the index of ds_rand element that is closest to the mean of ds_rand
    ds_rand_mean_index = (np.abs(ds_rand - ds_rand_mean)).argmin()
    sent_randoms[ds_rand_mean_index]

    S_id = sent_randoms[ds_rand_mean_index]
    select_activations = []
    select_sentences = []
    for model_act in ext_obj.model_group_act:
        select_sentences.append([model_act['activations'][s][1] for s in S_id])
        select_activations.append([model_act['activations'][s][0] for s in S_id])
    df_rand = pd.DataFrame(np.asarray(select_sentences).transpose(), index=S_id, columns=ext_obj.model_spec)
    df_act_rand = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id, columns=ext_obj.model_spec)

    (ext_sh, optim_sh) = make_shorthand(extract_id[0], optim_id[1])
    # in optim_sh replace O=2-Ds with O=DRand
    optim_sh=optim_sh.replace('O=2-D_s','O=D_rand')
    ax_title = f'sent,{ext_sh},{optim_sh}'
    df_rand.to_csv(Path(ANALYZE_DIR, f'{ax_title}.csv'))

    # check if there an overlap between columns of df_max and df_min
    sent_max=df_max['roberta-base'].values
    sent_min=df_min['roberta-base'].values
    # find overlap between sent_max and sent_min
    overlap = np.intersect1d(sent_max, sent_min)
    print(f'overlap between max and min {overlap}')
    # plot a histogram for number of words in each sentence in sent_max and sent_min
    sentence_from_ext_obj=[]
    for sent_id,sentence in enumerate(ext_obj.data_):
        if u'\xa0' in sentence['text']:
            sentence['text'] = sentence['text'].replace(u'\xa0', u' ')
        words_from_text = sentence['text'].split(' ')
        if '.' in words_from_text[-1] and ext_obj.stim_type=='textNoPeriod':
            words_from_text[-1] = words_from_text[-1].rstrip('.')
        # drop empty string
        words_from_text = [x for x in words_from_text if x]
        word_ind = np.arange(len(words_from_text))
        sent_for_model=' '.join(words_from_text)
        sentence_from_ext_obj.append(sent_for_model)

    sent_max_loc=[sentence_from_ext_obj.index(x) for x in sent_max]
    sent_min_loc=[sentence_from_ext_obj.index(x) for x in sent_min]
    assert len(sent_max_loc)==len(sent_max)
    # get elements from ext_obj.data_ that are selected by sent_max_loc
    sent_max_data=[ext_obj.data_[x] for x in sent_max_loc]
    sent_min_data = [ext_obj.data_[x] for x in sent_min_loc]
    sent_all_data = [ext_obj.data_[x] for x in range(len(ext_obj.data_))]


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
        seaborn.distplot(sent_all_lex_values[:, i], bins=50, label='full', norm_hist=True, hist=False,ax=axes[i],kde_kws={"lw": 5, "color": 'k'})
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
    ax_title = f'sent_features,{group_id},{dataset_id},Ns={optimizer_obj.N_s},Low_dim={optimizer_obj.low_dim}'
    # add a suptitle to the figure
    fig.suptitle(ax_title, fontsize=10,y=.99)
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # save figure as pdf and png
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1)
    fig.show()
    # compare low dim to no low dim results
    #
    # optim_id_low_dim = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
    #             'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True']
    # #
    # optim_id_wo_low_dim=['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
    #             'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']
    # low_resolution = 'False'
    # optim_files = []
    # optim_results_low_dim = []
    # optim_results_wo_low_dim = []
    # for ext in extract_id:
    #     for optim in optim_id_low_dim:
    #         optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
    #         assert (optim_file.exists())
    #         optim_files.append(optim_file.__str__())
    #         optim_results_low_dim.append(load_obj(optim_file.__str__()))
    #     # get results without low dim
    #     for optim in optim_id_wo_low_dim:
    #         optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
    #         assert (optim_file.exists())
    #         optim_files.append(optim_file.__str__())
    #         optim_results_wo_low_dim.append(load_obj(optim_file.__str__()))
    #
    # ext_obj = extract_pool[extract_id[0]]()
    # ext_obj.load_dataset()
    # ext_obj()
    #
    # optimizer_obj_low_dim = optim_pool[optim_id_low_dim[0]]()
    # optimizer_obj_low_dim.load_extractor(ext_obj)
    # optimizer_obj_low_dim.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=True,
    #                                              save_results=False)
    # # make an optimizer object without low dim
    # optimizer_obj_wo_low_dim = optim_pool[optim_id_wo_low_dim[0]]()
    # optimizer_obj_wo_low_dim.load_extractor(ext_obj)
    # optimizer_obj_wo_low_dim.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=True,
    #                                                 save_results=False)
    #
    # DS_max_low_dim_low_dim, RDM_max_low_dim_low_dim = optimizer_obj_low_dim.gpu_object_function_debug(optim_results_low_dim[0]['optimized_S'])
    # DS_min_low_dim_low_dim, RDM_min_low_dim_low_dim = optimizer_obj_low_dim.gpu_object_function_debug(optim_results_low_dim[1]['optimized_S'])
    # # compute DS for no low dim
    # DS_max_wo_low_dim_wo_low_dim, RDM_Max_wo_low_dim_low_dim = optimizer_obj_wo_low_dim.gpu_object_function_debug(optim_results_wo_low_dim[0]['optimized_S'])
    # DS_min_wo_low_dim_wo_low_dim, RDM_min_low_dim_wo_low_dim = optimizer_obj_wo_low_dim.gpu_object_function_debug(optim_results_wo_low_dim[1]['optimized_S'])
    #
    # # compute Ds for low dim using no low dim
    # DS_max_low_dim_wo_low_dim, RDM_max_low_dim_wo_low_dim = optimizer_obj_wo_low_dim.gpu_object_function_debug(optim_results_low_dim[0]['optimized_S'])
    # DS_min_low_dim_wo_low_dim, RDM_min_low_dim_wo_low_dim = optimizer_obj_wo_low_dim.gpu_object_function_debug(optim_results_low_dim[1]['optimized_S'])
    # # compute Ds for no low dim using low dim
    # DS_max_wo_low_dim_low_dim, RDM_max_wo_low_dim_low_dim = optimizer_obj_low_dim.gpu_object_function_debug(optim_results_wo_low_dim[0]['optimized_S'])
    # DS_min_wo_low_dim_low_dim, RDM_min_wo_low_dim_low_dim = optimizer_obj_low_dim.gpu_object_function_debug(optim_results_wo_low_dim[1]['optimized_S'])
    #
    # ds_rand_low_dim = []
    # for k in tqdm(enumerate(range(200))):
    #     sent_random = list(np.random.choice(optimizer_obj_low_dim.N_S, optimizer_obj_low_dim.N_s))
    #     d_s_r, RDM_r = optimizer_obj_low_dim.gpu_object_function_debug(sent_random)
    #     ds_rand_low_dim.append(d_s_r)
    # # compute random ds for no low dim
    # ds_rand_wo_low_dim = []
    # for k in tqdm(enumerate(range(200))):
    #     sent_random = list(np.random.choice(optimizer_obj_wo_low_dim.N_S, optimizer_obj_wo_low_dim.N_s))
    #     d_s_r, RDM_r = optimizer_obj_wo_low_dim.gpu_object_function_debug(sent_random)
    #     ds_rand_wo_low_dim.append(d_s_r)
    #
    #
    #
    # # plot results
    # fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    # ax = plt.axes((.2, .7, .08, .25))
    # ax.scatter(.02 * np.random.normal(size=(np.asarray(len(ds_rand_low_dim)))) + 0, np.asarray(ds_rand_low_dim),
    #            color=(.6, .6, .6), s=2, alpha=.3)
    # rand_mean = np.asarray(ds_rand_low_dim).mean()
    # ax.scatter(0, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
    #            label=f'random= {rand_mean:.4f}', edgecolor='k')
    #
    #
    # ax.scatter(0, DS_min_low_dim_low_dim, color=np.divide((188, 80, 144), 255), s=50, label=f'Ds_min={DS_min_low_dim_low_dim:.4f}', edgecolor='k')
    # ax.scatter(0, DS_max_low_dim_low_dim, color=np.divide((255, 128, 0), 255), s=50, label=f'Ds_max={DS_max_low_dim_low_dim:.4f}', edgecolor='k')
    #
    # ax.scatter(0, DS_min_wo_low_dim_low_dim, color=np.divide((188, 80, 144), 255), s=50,edgecolor='w', label=f'Ds_min using full dim optimization samples={DS_min_wo_low_dim_low_dim:.4f}')
    # ax.scatter(0, DS_max_wo_low_dim_low_dim, color=np.divide((255, 128, 0), 255), s=50,edgecolor='w', label=f'Ds_max using full dim optimization samples={DS_max_wo_low_dim_low_dim:.4f}')
    #
    #
    # ax.scatter(1+.02 * np.random.normal(size=(np.asarray(len(ds_rand_wo_low_dim)))) + 0, np.asarray(ds_rand_wo_low_dim),
    #            color=(.6, .6, .6), s=2, alpha=.3)
    # rand_mean = np.asarray(ds_rand_wo_low_dim).mean()
    # ax.scatter(1, rand_mean, color=np.divide((55, 76, 128), 256), s=50,
    #            label=f'random= {rand_mean:.4f}',marker='s', edgecolor='k')
    # ax.scatter(1, DS_min_wo_low_dim_wo_low_dim,marker='s', color=np.divide((188, 80, 144), 255), s=50, label=f'Ds_min={DS_min_wo_low_dim_wo_low_dim:.4f}',
    #            edgecolor='k')
    # ax.scatter(1, DS_max_wo_low_dim_wo_low_dim,marker='s', color=np.divide((255, 128, 0), 255), s=50, label=f'Ds_max={DS_max_wo_low_dim_wo_low_dim:.4f}',
    #            edgecolor='k')
    #
    # ax.scatter(1, DS_min_low_dim_wo_low_dim, color=np.divide((188, 80, 144), 255), s=50, edgecolor='w',
    #            label=f'Ds_min using low dim optimization samples={DS_min_low_dim_wo_low_dim:.4f}',marker='s',)
    # ax.scatter(1, DS_max_low_dim_wo_low_dim, color=np.divide((255, 128, 0), 255), s=50, edgecolor='w',
    #            label=f'Ds_max using low dim optimization samples={DS_max_low_dim_wo_low_dim:.4f}',marker='s',)
    #
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_linewidth(1)
    # ax.spines['left'].set_linewidth(1)
    # ax.set_xlim((-.4, 1.4))
    # ax.set_ylim((0.0, 1.2))
    # ax.set_xticks([0,1])
    # ax.set_xticklabels(['low dim', 'full dim'],rotation=45)
    # ax.legend(bbox_to_anchor=(1.1, .2), frameon=True)
    # ax.set_ylabel(r'$D_s$')
    # # plot no low dim case
    #
    #
    # fig.show()
    # ax_title = f'ds_result,extractor={dataset}_low_dim_vs_no_low_dim_{optimizer_obj.N_s}'
    # save_path = Path(ANALYZE_DIR)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    # fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
    #             facecolor='auto',
    #             edgecolor='auto', backend=None)
    # save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    # fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
    #             facecolor='auto',
    #             edgecolor='auto', backend=None)
    # # looking at optimization

    # act_dict=optimizer_obj.activations[0]
    # act_ = [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
    # act = torch.tensor(act_, dtype=float, device=optimizer_obj.device, requires_grad=False)
    # act_pca, var_exp = low_dim_project(act)
    #
    # act_pca
    # # create a function that uses pca to project the activations to a lower dim
    #
    # pca = PCA(n_components=350)
    # pca.fit(act_)
    # act_pca_ = pca.transform(act_)
    # var_exp_ = pca.explained_variance_ratio_
    # # check whether act_pca and act_pca_ are the same
    # print(f'act_pca and act_pca_ are the same: {np.allclose(act_pca.cpu(), act_pca_)}')


    # compare the dimensionality of df_act_max

    # for each key in df_act_min and df_act_max, compute a pca over the rows and check the explained variance ratio
    dims_for_min = []
    dims_for_max = []
    for key in tqdm(df_act_min.keys()):
        a=np.stack(df_act_min[key].values)
        pca=PCA()
        pca.fit(a)
        dim_for_90_min=np.where(np.cumsum(pca.explained_variance_ratio_)<.9)[0][-1]/len(pca.explained_variance_ratio_)
        dims_for_min.append(dim_for_90_min)
        b=np.stack(df_act_max[key].values)
        pca=PCA()
        pca.fit(b)
        dim_for_90_max=np.where(np.cumsum(pca.explained_variance_ratio_)<.9)[0][-1]/len(pca.explained_variance_ratio_)
        dims_for_max.append(dim_for_90_max)
    # plot the results
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)

    ax = plt.axes((.2, .2, .35, .25))
    # plot a bar graph for dim_for_90_min and one for dim_for_90_max side by side
    ax.bar(np.arange(len(dims_for_min)), dims_for_min, width=.4, label='min')
    ax.bar(np.arange(len(dims_for_max)) + .4, dims_for_max, width=.4, label='max')
    ax.set_xticks(np.arange(len(dims_for_min)) + .2)
    ax.set_xticklabels(modelnames, rotation=90)
    ax.set_ylabel('fraction of variance explained')
    ax.set_title('PCA dimensionality for min and max activations')
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.show()

    ax_title = f'dimenstionalty_comp_ds_min_ds_max,low_dim={optimizer_obj.low_dim},{dataset}'
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)

    # check the projection of sampled activations on the PCA of the full activations
    optimizer_obj.load_extractor(ext_obj)
    proj_list_min=[]
    proj_list_max=[]
    for idx,act_dict in tqdm(enumerate(optimizer_obj.activations)):
        act_ = [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
        act_=np.stack(act_)
        pca=PCA()
        pca.fit(act_)
        a=np.stack(df_act_min[ext_obj.model_spec[idx]])
        # for each colum in a compute the projection on the pca components
        proj=np.matmul(a,pca.components_.T)
        proj_list_min.append(proj)
        #
        b=np.stack(df_act_max[ext_obj.model_spec[idx]])
        # for each colum in a compute the projection on the pca components
        proj=np.matmul(b,pca.components_.T)
        proj_list_max.append(proj)

    fig=plt.figure(figsize=(11,8),dpi=300,frameon=False)
    # create a grid of 10 columns and 7 rows
    gs=GridSpec(7,10)
    for idx,proj in enumerate(proj_list_min):
        # in each row plot a histogram of the projection values for each column in proj
        for idy in range(10):
            # select the column in proj
            a=proj[:,idy]
            # sort the values in a and plot a bar graph
            a=np.sort(a)
            ax=fig.add_subplot(gs[idx,idy])
            ax.scatter(np.arange(len(a)),a,color='b',s=1)
            # plot a line at 0
            #ax.axhline(0,color='k')
            b=proj_list_max[idx][:,idy]
            b=np.sort(b)
            ax.scatter(np.arange(len(b)),b,color='r',s=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            if idy==0:
                ax.set_ylabel(modelnames[idx],fontsize=8,rotation=0,horizontalalignment='right',verticalalignment='center',labelpad=10)
            if idx==0:
                ax.set_title(f'PC#{idy+1}',fontsize=8)
    ax_title = f'proj_on_pca_ds_min,low_dim={optimizer_obj.low_dim},{dataset}'
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',edgecolor='auto', backend=None)


        #act = torch.tensor(act_, dtype=float, device=optimizer_obj.device, requires_grad=False)
    # compute the DS scores for sentences that are selected
    Ds_rand_file_selected='sent,G=best_performing_pereira_1-D=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-B=None-AVE=False,O=D_rand-Nit=500-Ns=100-Nin=1-LD=False_eh_edit.csv'
    Ds_min_file_selected='sent,G=best_performing_pereira_1-D=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-B=None-AVE=False,O=2-D_s-Nit=500-Ns=100-Nin=1-LD=False_eh_edit.csv'
    Ds_max_file_selected='sent,G=best_performing_pereira_1-D=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-B=None-AVE=False,O=D_s-Nit=500-Ns=100-Nin=1-LD=False_eh_edit.csv'
    Ds_rand_file_selected=Path(ANALYZE_DIR,Ds_rand_file_selected)
    Ds_max_file_selected=Path(ANALYZE_DIR,Ds_max_file_selected)
    Ds_min_file_selected=Path(ANALYZE_DIR,Ds_min_file_selected)

    # read the Ds_rand_file_selected csv
    df_rand_selected=pd.read_csv(Ds_rand_file_selected)
    df_max_selected=pd.read_csv(Ds_max_file_selected)
    df_min_selected=pd.read_csv(Ds_min_file_selected)
    # select the sentences that are in the included in df_rand_selected
    df_rand_selected=df_rand_selected[df_rand_selected['Include']==1]
    df_max_selected=df_max_selected[df_max_selected['Include']==1]
    df_min_selected=df_min_selected[df_min_selected['Include']==1]

    # find the location of sentences in
    S_id_rand_selected=[int(x) for x in list(df_rand_selected['Unnamed: 0'])]
    S_id_max_selected=[int(x) for x in list(df_max_selected['Unnamed: 0'])]
    S_id_min_selected=[int(x) for x in list(df_min_selected['Unnamed: 0'])]
    assert len(S_id_rand_selected)==len(S_id_max_selected)==len(S_id_min_selected)==80
    DS_max_selected, RDM_max_selected = optimizer_obj.gpu_object_function_debug(S_id_max_selected)
    DS_min_selected, RDM_min_selected = optimizer_obj.gpu_object_function_debug(S_id_min_selected)
    DS_rand_selected, RDM_rand_selected = optimizer_obj.gpu_object_function_debug(S_id_rand_selected)

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

    fig.show()
    ax_title = f'sent_selected,{group_id},{dataset_id},Ns={optimizer_obj.N_s},Low_dim={optimizer_obj.low_dim}'
    # add a suptitle to the figure
    fig.suptitle(ax_title, fontsize=10, y=.99)
    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')

    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',edgecolor='auto', backend=None)

