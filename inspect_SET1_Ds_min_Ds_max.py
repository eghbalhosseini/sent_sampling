import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from utils import extract_pool
from utils.optim_utils import optim_pool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import torch
from scipy.spatial.distance import pdist, squareform
if __name__ == '__main__':
    extract_id = [
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textPeriod-activation-bench=None-ave=False']
    #optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
    #             'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True'             ]
    #
    optim_id=['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
                'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']
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

    optimizer_obj = optim_pool[optim_id[0]]()
    optimizer_obj.load_extractor(ext_obj)

    xy_dir = os.path.join(SAVE_DIR,
                          f"{optimizer_obj.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}-low_dim={optimizer_obj.low_dim}.pkl")
    if os.path.exists(xy_dir):
        print('loading precomputed correlation matrix ')
        D_precompute = load_obj(xy_dir)
        optimizer_obj.XY_corr_list = D_precompute
    else:
        print('precomputing correlation matrix ')
        optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=False,
                                                 save_results=True)

    DS_max,RDM_max=optimizer_obj.gpu_object_function_debug(optim_results[0]['optimized_S'])
    DS_min,RDM_min = optimizer_obj.gpu_object_function_debug(optim_results[1]['optimized_S'])

    ds_rand = []
    RDM_rand=[]
    for k in tqdm(enumerate(range(50))):
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
    ax_title = f'ds_result,extractor={dataset},optimizer={optim}'
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

    # add colorbar for RDM_max
    #ax = plt.axes((.9, .7, .01, .25))
    #plt.colorbar(im, cax=ax)


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
    #ax = plt.axes((.9, .4, .01, .25))
    #plt.colorbar(im, cax=ax)

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

    fig.show()


    save_path = Path(ANALYZE_DIR)
    save_loc = Path(save_path.__str__(), f'{ax_title}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_title}.eps')
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
        im = ax.imshow(X_Max[i], cmap='viridis',vmax=RDM_max.cpu().numpy().max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}',fontsize=6)
        ax.set_title('Ds_max')
        # turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(X_Min)):
        ax = plt.subplot(3, 7, i + 1+14)
        im = ax.imshow(X_Min[i], cmap='viridis', vmax=RDM_max.cpu().numpy().max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_min')
        ax.set_xticks([])
        ax.set_yticks([])


    for i in range(len(X_rand)):
        ax = plt.subplot(3, 7, i+1)
        im = ax.imshow(X_rand[i], cmap='viridis', vmax=RDM_max.cpu().numpy().max())
        ax.set_ylabel(f'{ext_obj.model_spec[i]}', fontsize=6)
        ax.set_title('Ds_rand')
        ax.set_xticks([])
        ax.set_yticks([])

    ax = plt.axes((.95, .05, .01, .25))
    plt.colorbar(im, cax=ax)

    fig.show()

    save_path = Path(ANALYZE_DIR)
    ax_1=ax_title.replace('ds_result','model_rdm')
    save_loc = Path(save_path.__str__(), f'{ax_1}.png')
    fig.savefig(save_loc.__str__(), format='png', metadata=None, bbox_inches=None, pad_inches=0.1, dpi=350,
                facecolor='auto',
                edgecolor='auto', backend=None)
    save_loc = Path(save_path.__str__(), f'{ax_1}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto', backend=None)

    S_id = optim_results[0]['optimized_S']
    select_activations=[]
    for model_act in ext_obj.model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])


    df = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id, columns=ext_obj.model_spec)
    ax1=ax_title.replace('ds_results', 'sentences')
    df.to_csv(Path(ANALYZE_DIR, f'{ax1}.csv'))

    S_id = optim_results[1]['optimized_S']
    select_activations = []
    for model_act in ext_obj.model_group_act:
        select_activations.append([model_act['activations'][s][1] for s in S_id])

    df = pd.DataFrame(np.asarray(select_activations).transpose(), index=S_id, columns=ext_obj.model_spec)
    ax1 = ax_title.replace('ds_result', 'sentences')
    df.to_csv(Path(ANALYZE_DIR, f'{ax1}.csv'))
