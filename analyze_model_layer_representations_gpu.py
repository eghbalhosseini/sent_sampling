from utils import extract_pool
from utils.optim_utils import optim_pool, pt_create_corr_rdm_short
import argparse
from utils.data_utils import RESULTS_DIR, save_obj, SAVE_DIR, ANALYZE_DIR
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')

args = parser.parse_args()

if __name__ == '__main__':
    extractor_id = args.extractor_id
    optimizer_id = args.optimizer_id
    print(extractor_id+'\n')
    print(optimizer_id+'\n')
    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    model_layers = extractor_obj.layer_name
    extractor_obj()
    mdl_name=str(np.unique(extractor_obj.model_spec).squeeze())
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    del extractor_obj
    # get score from pereira
    scores = pd.read_csv(os.path.join(SAVE_DIR, 'scoresscoresscores', 'scores-Pereira2018-encoding-normalized.csv'))
    score_layer = list(scores['layer'][scores['model'] == mdl_name])
    score_benchmark = list(scores['benchmark'][scores['model'] == mdl_name])
    score_score = np.asarray(scores['score'][scores['model'] == mdl_name])
    score_error = np.asarray(scores['error'][scores['model'] == mdl_name])
    # get activations
    layer_id_list = [int(x['layer']) for x in optimizer_obj.activations]
    # compute pca
    activation_list = []
    var_explained = []
    loadings = []
    components = []
    for idx, act_dict in tqdm(enumerate(optimizer_obj.activations)):
        act = torch.tensor(act_dict['activations'], dtype=float, device=optimizer_obj.device, requires_grad=False)
        # act must be in m sample * n feature shape ,
        u, s, v = torch.pca_lowrank(act, q=500)
        # keep 85% variance explained ,
        idx_85 = torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2) < .85
        cols = list(torch.where(idx_85)[0].cpu().numpy())
        act_50 = torch.matmul(act, v[:, :300])
        act_85 = torch.matmul(act, v[:, cols])
        activation_list.append(act_50)
        var_explained.append(
            torch.cumsum(torch.cat((torch.tensor([0], device=optimizer_obj.device), s ** 2)), dim=0) / torch.sum(s ** 2))
        # var_explained.append(torch.cumsum(s**2,dim=0)/torch.sum(s**2))
    var_explained = torch.stack(var_explained).cpu()
    # plot pca results
    num_colors = len(activation_list) + 1
    h0 = cm.get_cmap('viridis_r', num_colors)
    line_cols = np.flipud(h0(np.arange(num_colors) / num_colors))
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"{extractor_id}\n{mdl_name}\n", fontsize=16)
    ax = plt.subplot(2, 1, 1)
    ax.imshow(var_explained.cpu().numpy(), aspect='auto', interpolation='none')
    ax.set_yticks(np.arange(var_explained.shape[0]))
    ax.set_yticklabels([f" {model_layers[idx]}" for idx, x in enumerate(var_explained)])
    ax.set_xlabel('#PC')
    ax = plt.subplot(2, 1, 2)
    [ax.plot(x, color=line_cols[idx, :], linewidth=2, label=f" {model_layers[idx]}") for idx, x in
     enumerate(var_explained)]
    ax.legend()
    ax.set_xlabel('#PC')
    ax.set_ylabel('var explained')
    ax.set_title(f"{mdl_name}", fontsize=16)
    # save results
    plt.savefig(os.path.join(ANALYZE_DIR,f"{mdl_name}_layer_var_explained.png"), dpi=None, facecolor='w', edgecolor='w',
               orientation='landscape',transparent=True, bbox_inches=None, pad_inches=0.1,frameon=False)

    #
    act_list_norm = [(X - X.mean(axis=1, keepdim=True)) for X in activation_list]
    act_list_norm = [torch.nn.functional.normalize(X) for X in act_list_norm]
    #
    num_iter = 100
    total_sent = activation_list[0].shape[0]
    num_samples = 50
    layer_dist = []
    for idx in tqdm(range(len(activation_list))):
        pair_dist = []
        for idy in tqdm(range(len(activation_list)), position=1):
            sample_dist = []
            pair_list_norm = [act_list_norm[idx], act_list_norm[idy]]
            XY_corr_list = [torch.tensor(1, device=X.device, dtype=float) - torch.mm(X, torch.transpose(X, 1, 0)) for X
                            in
                            pair_list_norm]
            for sample_iter in range(num_iter):
                samples = torch.tensor(np.random.choice(total_sent, num_samples, replace=False), dtype=torch.long,
                                       device=act_list_norm[0].device)
                pairs = torch.combinations(samples, with_replacement=False)
                XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in XY_corr_list]
                XY_corr_sample_tensor = torch.stack(XY_corr_sample)
                XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
                if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
                    XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
                assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
                d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=samples.device)
                # n1 = d_mat.shape[1],
                # correction = n1 * n1 / (n1 * (n1 - 1) / 2),
                # d_val = correction * d_mat.mean(dim=(0, 1)),
                d_val = d_mat[0, 1]
                sample_dist.append(d_val)
            pair_dist.append(torch.stack(sample_dist))
        layer_dist.append(pair_dist)
    print("Done!")
    # plot and save the results
    # load pereria experiment settings
    pereira_settings = extract_pool[
        'group=best_performing_pereira_1-dataset=ud_sentences-activation-bench=None-ave=False']()
    try:
        model_loc = pereira_settings.model_spec.index(mdl_name)
        pereira_layer_id = pereira_settings.layer_spec[model_loc]
    except ValueError as e:
        pereira_layer_id = np.argmax(score_score)

    Pereira_dist = torch.mean(torch.stack(layer_dist[pereira_layer_id]), dim=1)
    dist_val, dist_idx = torch.sort(Pereira_dist)
    assert (dist_idx[0] == pereira_layer_id)
    cuts = np.linspace(dist_val.cpu().numpy().min(), dist_val.cpu().numpy().max(), 4, endpoint=False)

    # plot and save the results
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_axes((.1, .4, .5 * 1.5, .5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=.1, pad=0.5)
    im = ax.imshow(torch.stack([torch.stack([x.mean() for x in y]) for y in layer_dist]).cpu(), aspect='auto',
                   interpolation='none')
    ax.set_yticks(np.arange(var_explained.shape[0]))
    ax.set_xticks(np.arange(var_explained.shape[0]))
    ax.set_yticklabels([f" {model_layers[idx]}" for idx, x in enumerate(var_explained)])
    ax.set_xticklabels([f"{model_layers[idx]}" for idx, x in enumerate(var_explained)], rotation=90)

    cbar = fig.colorbar(im, cax=cax)
    ax = fig.add_axes((.1, .91, .5 * 1.37, .05))
    ax.errorbar(np.arange(len(layer_id_list)), score_score[layer_id_list], yerr=score_error[layer_id_list],
                markersize=10, marker='o', linewidth=0, elinewidth=2, label=f'score {score_benchmark[0]}')
    ax.set_ylabel('score')
    ax.set_xticks([])

    ax = fig.add_axes((.1, .05, .5 * 1.37, .22))
    ax.scatter(np.arange(dist_val.cpu().shape[0]), dist_val.cpu())
    ax.set_xlim((-1, dist_val.cpu().shape[0] + 1))
    ax.set_ylim((0 - .05, np.max(dist_val.cpu().numpy()) + .05))
    [ax.plot(plt.xlim(), [x, x], 'k--') for x in cuts],
    closest_points = [np.argmin(np.abs(dist_val.cpu() - x)) for x in cuts]
    [ax.scatter(x.cpu().numpy(), dist_val[int(x.cpu().numpy())].cpu().numpy(), 60, color=(0, 0, 0)) for x in
     closest_points]
    ax.set_xticks(tuple(np.arange(dist_val.cpu().shape[0])))
    # ax.set_xticklabels(dist_idx.cpu().numpy()),
    ax.set_xticklabels([model_layers[int(x)] for x in dist_idx.cpu().numpy()], rotation=90)
    [ax.plot([x.cpu().numpy(), x.cpu().numpy()], plt.ylim(), 'k-') for x in closest_points]
    ax.set_xlim((0 - .5, len(dist_idx) - .5))
    plt.savefig(os.path.join(ANALYZE_DIR,f"{mdl_name}_layerwise_similiarty_dist_vs_score.png"), dpi=None, facecolor='w', edgecolor='w',
       orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0.1,frameon=False)