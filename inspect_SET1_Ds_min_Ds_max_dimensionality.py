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
    return pca_loads, var_explained[idx_cutoff]

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
    extract_id = [
        'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textPeriod-activation-bench=None-ave=False']
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

        ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
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

        ax.scatter(l_v[rot, 0].cpu(), l_v[rot, 1].cpu(), s=.5, c=line_cols)
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




