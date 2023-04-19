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
from matplotlib.pyplot import GridSpec
import pandas as pd
from pathlib import Path
import torch
from utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib
import xarray as xr
import pandas as pd
if __name__ == '__main__':
    modelnames = ['roberta-base',  'xlnet-large-cased',  'bert-large-uncased','xlm-mlm-en-2048', 'gpt2-xl', 'albert-xxlarge-v2','ctrl']
    per_names= ['pereira2018_243sentences', 'pereira2018_384sentences']
    optim_runs=[75, 100,125,150,175]
    # create an itertool over per_names and optim_runs
    for per_name in per_names:
        for optim_run in optim_runs:
            extract_ids = [f'group=best_performing_pereira_1-dataset={per_name}_textNoPeriod-activation-bench=None-ave=False']
            optim_id = [f'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples={optim_run}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True',]
            # get obj= from optim_id
            obj_id = ['Ds_max', '2-Ds_max']
            # get n_samples from each element in optim_id
            low_resolution=False

            low_resolution = 'False'
            optim_files = []
            optim_results = []
            for ext in extract_ids:
                for optim in optim_id:
                    (ext_sh, optim_sh) = make_shorthand(ext, optim)
                    # optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
                    optim_file = Path(RESULTS_DIR, f"results_{ext}_{optim}.pkl")
                    assert (optim_file.exists())

                    optim_files.append(optim_file.__str__())
                    optim_results.append(load_obj(optim_file.__str__()))

            ext_obj = extract_pool[extract_ids[0]]()

            ext_obj()
            ext_obj.N_S = len(ext_obj.model_group_act[0]['activations'])

            optimizer_obj = optim_pool[optim_id[0]]()
            optimizer_obj.load_extractor(ext_obj)

            optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=True,
                                                     save_results=False)
            # get n_samples from optimizer_obj
            n_samples = optimizer_obj.N_s
            # get low_dim from optimizer_obj
            is_low_dim = optimizer_obj.low_dim
            DS_max, RDM_max = optimizer_obj.gpu_object_function_debug(optim_results[0]['optimized_S'])
            if  isinstance(RDM_max, torch.Tensor):
                RDM_max = RDM_max.cpu().numpy()

            ds_rand=[]
            RDM_rand=[]
            for k in tqdm(enumerate(range(200))):
                sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
                d_s_r,RDM_r= optimizer_obj.gpu_object_function_debug(sent_random)
                ds_rand.append(d_s_r)
                RDM_rand.append(RDM_r)

            fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
            ax = plt.axes((.2, .7, .08, .25))
            ax.scatter(.02 * np.random.normal(size=(np.asarray(len(ds_rand)))) + 0, np.asarray(ds_rand),
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
            ax.set_ylim((0.0, 1.2))
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.legend(bbox_to_anchor=(1.1, .2), frameon=True)
            ax.set_ylabel(r'$D_s$')
            ax.tick_params(direction='out', length=3, width=2, colors='k',
                           grid_color='k', grid_alpha=0.5)

            mask = np.triu(np.ones_like(RDM_max, dtype=np.bool))
            # change True to nan and false to 1
            mask = np.where(mask, np.nan, 1)

            ax = plt.axes((.6, .73, .25, .25))
            RDM_rand_mean = torch.stack(RDM_rand).mean(0).cpu().numpy()
            RDM_rand_mean = RDM_rand_mean.T
            RDM_rand_mean = np.multiply(RDM_rand_mean, mask)
            im = ax.imshow(RDM_rand_mean, cmap='viridis', vmax=np.nanmax(RDM_max), vmin=0)
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
            im = ax.imshow(RDM_max, cmap=cmap, vmin=0, vmax=np.nanmax(RDM_max))
            # add lower triangle of RDM_max to image plot
            for i in range(RDM_max.shape[0]):
                for j in range(RDM_max.shape[1]):
                    text = ax.text(j, i, f'{RDM_max[i, j]:.2f}',
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
            color_set = [np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255)]
            rdm_vec = np.vstack((rdm_rand_vec, rdm_max_vec))
            # plot one line per column in rdm_vec
            for i in range(rdm_vec.shape[1]):
                ax.plot([1, 2], rdm_vec[:, i], color='k', alpha=.3, linewidth=.5, zorder=1)
                # plot a scatter with each point color same as color_set
                ax.scatter([1, 2], rdm_vec[:, i], color=color_set, s=10, marker='o', alpha=.8, zorder=2)
            # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

            ax.boxplot(rdm_vec.transpose(), vert=True, showfliers=False, showmeans=False,
                       meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
            # set xtick labels to ds_min, ds_rand, ds_max
            ax.set_xticklabels(['ds_rand', 'ds_max'], fontsize=8)
            ax.set_ylabel('Ds')
            ax.set_ylim((0, 1.4))
            ax.set_title('Ds distribution')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim((.75, 2.25))

            #fig.show()


            save_path = Path(ANALYZE_DIR)


            save_loc = Path(save_path.__str__(),  f'{per_name}_Ds_{optimizer_obj.N_s}_{extract_ids[0]}.png')
            fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                        facecolor='auto',
                        edgecolor='auto', backend=None)
            save_loc = Path(save_path.__str__(),  f'{per_name}_Ds_{optimizer_obj.N_s}_{extract_ids[0]}.eps')
            fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                        facecolor='white',
                        edgecolor='white')
    #%%
    # load the Pereira data
    neural_nlp_loc='/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/'
    pereira_loc="/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Pereira2018_assembly.pkl"
    pereira_ds = pd.read_pickle(pereira_loc)
    # 243
    per_name=per_names[0]
    ext=f'group=best_performing_pereira_1-dataset={per_name}_textNoPeriod-activation-bench=None-ave=False'
    optim_run=100
    optim_max = f'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples={optim_run}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_min = f'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples={optim_run}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_243_max=load_obj(Path(RESULTS_DIR, f"results_{ext}_{optim_max}.pkl").__str__())['optimized_S']
    optim_243_min = load_obj(Path(RESULTS_DIR, f"results_{ext}_{optim_min}.pkl").__str__())['optimized_S']
    ext_obj = extract_pool[ext]()
    ext_obj()
    all_sentences=[x[1] for x in ext_obj.model_group_act[0]['activations']]
    ds_243max_sent = [ext_obj.model_group_act[0]['activations'][i][1] for i in sorted(optim_243_max)]
    ds_243min_sent = [ext_obj.model_group_act[0]['activations'][i][1] for i in sorted(optim_243_min)]
    # make a random set of sentences from the 243 sentences with the same size as optim_243_max
    optim_243_rand = np.random.choice(len(all_sentences), size=len(ds_243max_sent), replace=False)
    ds_243rand_sent = [ext_obj.model_group_act[0]['activations'][i][1] for i in sorted(optim_243_rand)]


    # 384
    per_name=per_names[1]
    ext=f'group=best_performing_pereira_1-dataset={per_name}_textNoPeriod-activation-bench=None-ave=False'
    optim_run=150
    optim_max = f'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples={optim_run}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_min = f'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples={optim_run}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_384_max=load_obj(Path(RESULTS_DIR, f"results_{ext}_{optim_max}.pkl").__str__())['optimized_S']
    optim_384_min = load_obj(Path(RESULTS_DIR, f"results_{ext}_{optim_min}.pkl").__str__())['optimized_S']
    ext_obj = extract_pool[ext]()
    ext_obj()
    all_sentences = [x[1] for x in ext_obj.model_group_act[0]['activations']]
    ds_384max_sent = [ext_obj.model_group_act[0]['activations'][i][1] for i in sorted(optim_384_max)]
    ds_384min_sent = [ext_obj.model_group_act[0]['activations'][i][1] for i in sorted(optim_384_min)]
    optim_384_rand = np.random.choice(len(all_sentences), size=len(ds_384max_sent), replace=False)
    ds_384rand_sent = [ext_obj.model_group_act[0]['activations'][i][1] for i in sorted(optim_384_rand)]

    # make a flat list from the list of lists
    ds_min_sent = [item for sublist in [ds_384min_sent,ds_243min_sent] for item in sublist]
    ds_max_sent = [item for sublist in [ds_384max_sent, ds_243max_sent] for item in sublist]
    ds_rand_sent = [item for sublist in [ds_384rand_sent, ds_243rand_sent] for item in sublist]
    # find location of ds_min_sentences in pereira_243.attrs['stimulus_set']
    ds_min_loc = [pereira_ds.attrs['stimulus_set']['stimulus_id'][np.where(pereira_ds.attrs['stimulus_set']['sentence'] == i)[0][0]] for i in ds_min_sent]
    ds_max_loc = [pereira_ds.attrs['stimulus_set']['stimulus_id'][np.where(pereira_ds.attrs['stimulus_set']['sentence'] == i)[0][0]] for i in ds_max_sent]
    ds_rand_loc = [pereira_ds.attrs['stimulus_set']['stimulus_id'][np.where(pereira_ds.attrs['stimulus_set']['sentence'] == i)[0][0]] for i in ds_rand_sent]

    # select part of pereira_ds that has the same stimulus id as ds_min_loc
    pereira_ds_min = pereira_ds[pereira_ds['stimulus_id'].isin(ds_min_loc)]
    pereira_ds_max = pereira_ds[pereira_ds['stimulus_id'].isin(ds_max_loc)]
    pereira_ds_rand = pereira_ds[pereira_ds['stimulus_id'].isin(ds_rand_loc)]
    # fix attributes of pereira_ds_min so that it only has ds_min_loc
    pereira_ds_min.attrs['stimulus_set'] = pereira_ds.attrs['stimulus_set'][pereira_ds.attrs['stimulus_set']['stimulus_id'].isin(ds_min_loc)]
    pereira_ds_max.attrs['stimulus_set'] = pereira_ds.attrs['stimulus_set'][pereira_ds.attrs['stimulus_set']['stimulus_id'].isin(ds_max_loc)]
    pereira_ds_rand.attrs['stimulus_set'] = pereira_ds.attrs['stimulus_set'][pereira_ds.attrs['stimulus_set']['stimulus_id'].isin(ds_rand_loc)]

    # save per_ds_min and per_ds_max
    import pickle
    with open(Path(neural_nlp_loc, f"pereira_ds_min.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_min, f)
    with open(Path(neural_nlp_loc, f"pereira_ds_max.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_max, f)
    with open(Path(neural_nlp_loc, f"pereira_ds_rand.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_rand, f)

    with open(Path(neural_nlp_loc, f"pereira_ds_full.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds, f)

    # create a second version of the data where there is a new coordinate for indicating which sentence was selected
    # this is to make it easier to plot the data
    pereira_ds_min2 = pereira_ds.copy()
    pereira_ds_max2 = pereira_ds.copy()
    pereira_ds_rand2 = pereira_ds.copy()

    # create a new coordinate where it is true when ds_min_loc is in the stimulus_id, use assign_coords to add this coordinate
    pereira_ds_min2 = pereira_ds_min2.assign_coords({'optim_sentence': (('presentation'), pereira_ds_min2['stimulus_id'].isin(ds_min_loc))})
    pereira_ds_max2 = pereira_ds_max2.assign_coords({'optim_sentence': (('presentation'), pereira_ds_max2['stimulus_id'].isin(ds_max_loc))})
    pereira_ds_rand2 = pereira_ds_rand2.assign_coords({'optim_sentence': (('presentation'), pereira_ds_rand2['stimulus_id'].isin(ds_rand_loc))})

    # save per_ds_min2 and per_ds_max2 and per_ds_rand2
    with open(Path(neural_nlp_loc, f"pereira_ds_min_v2.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_min2, f)
    with open(Path(neural_nlp_loc, f"pereira_ds_max_v2.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_max2, f)
    with open(Path(neural_nlp_loc, f"pereira_ds_rand_v2.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_rand2, f)

    # create a version 3 with only language atlas
    pereira_ds_min3 = pereira_ds_min2.copy()
    pereira_ds_max3 = pereira_ds_max2.copy()
    pereira_ds_rand3 = pereira_ds_rand2.copy()

    # select only language atlas in pereira_ds_min3 but keep atlas coordinate (this is done in the next step)
    pereira_ds_min3 = pereira_ds_min3.sel(atlas='language')
    pereira_ds_max3 = pereira_ds_max3.sel(atlas='language')
    pereira_ds_rand3 = pereira_ds_rand3.sel(atlas='language')
    # assign an atlas coordinate to pereira_ds_min3
    # make a list language with the same size as the nubmer of neuroids
    language = ['language' for i in range(pereira_ds_min3['neuroid'].size)]
    pereira_ds_min3 = pereira_ds_min3.assign_coords({'atlas': (('neuroid'), language)})
    pereira_ds_max3 = pereira_ds_max3.assign_coords({'atlas': (('neuroid'), language)})
    pereira_ds_rand3 = pereira_ds_rand3.assign_coords({'atlas': (('neuroid'), language)})



    # save per_ds_min3 and per_ds_max3 and per_ds_rand3
    with open(Path(neural_nlp_loc, f"pereira_ds_min_v3.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_min3, f)
    with open(Path(neural_nlp_loc, f"pereira_ds_max_v3.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_max3, f)
    with open(Path(neural_nlp_loc, f"pereira_ds_rand_v3.pkl").__str__(), 'wb') as f:
        pickle.dump(pereira_ds_rand3, f)














