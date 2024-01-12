import importlib
import utils
importlib.reload(utils)
from sent_sampling.utils import extract_pool,model_grps_config
import utils.optim_utils
importlib.reload(utils.optim_utils)
from sent_sampling.utils.optim_utils import optim, optim_pool, pt_create_corr_rdm_short
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj, ANALYZE_DIR
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
from tqdm import tqdm_notebook
import fnmatch
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable


extract_id=['group=gpt2-xl_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
            'group=ctrl_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
            'group=bert-large-uncased-whole-word-masking_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
            'group=albert-xxlarge-v2_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
            'group=roberta-base_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
            'group=xlm-mlm-en-2048_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
            'group=xlnet-large-cased_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False']

optim_id=['coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=50-n_init=2-run_gpu=True',
          'coordinate_ascent_eh-obj=D_s_var-n_iter=1000-n_samples=50-n_init=2-run_gpu=True']


optim_files=[]
optim_results=[]
for ext in extract_id:
    for optim in optim_id:
        optim_file=os.path.join(RESULTS_DIR,f"results_{ext}_{optim}.pkl")
        optim_files.append(optim_file)
        optim_results.append(load_obj(optim_file))


fig = plt.figure(figsize=[15,15])
ax = fig.add_axes([.1,.1,.4,.6])

cmap=cm.get_cmap('viridis_r')

alph_col=cmap(np.divide(range(len(optim_results)),len(optim_results)))
tick_l=[]
tick=[]
for idx, res in enumerate(optim_results):
    ax.barh(idx,res['optimized_d'],color=alph_col[[idx],:],label=res['optimizatin_name'])
    ext_obj=extract_pool[res['extractor_name']]()
    str_val="{:.5f}".format(res['optimized_d'])
    print(f"{str_val}")
    optim_type=re.search('obj=\w+-',res['optimizatin_name'])[0][0:-1]
    tick_l.append(f" {np.unique(ext_obj.model_spec)[0]}, {optim_type}, s: {len(res['optimized_S'])} \n {ext_obj.dataset}  ,  value:{str_val}")
    tick.append(idx)


#ax.set_xlabel(f"D_s \n\n  models:{ext_obj.model_spec}\nlayers:{ext_obj.layer_spec} averaging : {ext_obj.average_sentence}",fontsize=12)
ax.set_yticklabels(tick_l,fontsize=12)
ax.set_yticks(tick)
ax.set_title(res['optimizatin_name'],fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#ax.legend(bbox_to _anchor=(1.1, .85), frameon=True,fontsize=12)
ax.invert_yaxis()
#fig.savefig(os.path.join(Analysis_path,'DV_test_gamma_alpha_is_0.pdf'))
plt.savefig(os.path.join(ANALYZE_DIR,f"U01_SET2_optimization_results.png"), dpi=None, facecolor='w', edgecolor='w',
       orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0.1,frameon=False)

num_iter = 100
num_samples = 50

for idx, res in enumerate(optim_results):
    if idx > 0:
        ext_obj = extract_pool[res['extractor_name']]()
        mdl_name = np.unique(ext_obj.model_spec)[0]
        group = f'{mdl_name}_layers'
        extractor_id = f'group={group}-dataset={ext_obj.dataset}-{ext_obj.extract_type}-bench=None-ave={ext_obj.average_sentence}'
        extractor_obj = extract_pool[extractor_id]()
        extractor_obj.load_dataset()
        model_layers = extractor_obj.layer_name
        extractor_obj()
        mdl_name = str(np.unique(extractor_obj.model_spec).squeeze())
        optim_obj = optim_pool[res['optimizatin_name']]()
        optim_obj.load_extractor(extractor_obj)
        layer_id_list = [x['layer'] for x in optim_obj.activations]
        del extractor_obj
        activation_list = []
        var_explained = []
        loadings = []
        components = []
        pca_type = 'fixed'
        for idx, act_dict in tqdm(enumerate(optim_obj.activations)):
            act = torch.tensor(act_dict['activations'], dtype=float, device=optim_obj.device, requires_grad=False)
            # act must be in m sample * n feature shape ,
            u, s, v = torch.pca_lowrank(act, q=200)
            # keep 85% variance explained ,
            idx_85 = torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2) < .85
            cols = list(torch.where(idx_85)[0].cpu().numpy())
            if pca_type == 'fixed':
                act_pca = torch.matmul(act, v[:, :100])
            elif pca_type == 'equal_var':
                act_pca = torch.matmul(act, v[:, cols])

            activation_list.append(act_pca)
            var_explained.append(
                torch.cumsum(torch.cat((torch.tensor([0], device=optim_obj.device), s ** 2)), dim=0) / torch.sum(
                    s ** 2))
        # var_explained.append(torch.cumsum(s**2,dim=0)/torch.sum(s**2))
        var_explained = torch.stack(var_explained).cpu()

        total_sent = activation_list[0].shape[0]
        act_list_norm = [(X - X.mean(axis=1, keepdim=True)) for X in activation_list]
        act_list_norm = [torch.nn.functional.normalize(X) for X in act_list_norm]
        layer_dist = []
        for idx in tqdm_notebook(range(len(activation_list))):
            pair_dist = []
            for idy in tqdm_notebook(range(len(activation_list)), position=1):
                sample_dist = []
                pair_list_norm = [act_list_norm[idx], act_list_norm[idy]]
                XY_corr_list = [torch.tensor(1, device=X.device, dtype=float) - torch.mm(X, torch.transpose(X, 1, 0))
                                for X in
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

        optim_act_list_norm = [x[res['optimized_S'], :] for x in act_list_norm]
        layer_similarity = [pt_create_corr_rdm_short(x) for x in optim_act_list_norm]
        optim_pairs = torch.combinations(torch.tensor(np.arange(len(res['optimized_S']))), with_replacement=False)
        layer_optim_dist = []
        for idx in tqdm_notebook(range(len(activation_list))):
            pair_optim_dist = []
            for idy in tqdm_notebook(range(len(activation_list)), position=1):
                pair_similarity = [layer_similarity[idx], layer_similarity[idy]]
                XY_corr_sample = [XY_corr[optim_pairs[:, 0], optim_pairs[:, 1]] for XY_corr in pair_similarity]
                XY_corr_sample_tensor = torch.stack(XY_corr_sample)
                d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=XY_corr_sample_tensor.device)
                d_val = d_mat[0, 1].cpu()
                pair_optim_dist.append([d_val])
            layer_optim_dist.append(pair_optim_dist)
        print("Done!")
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
        Pereira_dist_optim = torch.tensor([torch.stack([x[0] for x in y]) for y in layer_optim_dist][pereira_layer_id])
        dist_val_optim, dist_idx_optim = torch.sort(Pereira_dist_optim)
        Pereira_ordered = Pereira_dist_optim[dist_idx]
        assert (dist_idx[0] == pereira_layer_id)

        fig = plt.figure(figsize=(8, 8 * 1.5))

        ax = fig.add_axes((.1, .4, .3 * 1.5, .3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=.05, pad=0.1)
        im = ax.imshow(torch.stack([torch.stack([x.mean() for x in y]) for y in layer_dist]).cpu(), aspect='auto',
                       interpolation='none')
        ax.set_yticks(np.arange(var_explained.shape[0]))
        ax.set_xticks(np.arange(var_explained.shape[0]))
        ax.set_yticklabels([f" {model_layers[idx]}" for idx, x in enumerate(var_explained)])
        ax.set_xticklabels([f"{model_layers[idx]}" for idx, x in enumerate(var_explained)], rotation=90)

        cbar = fig.colorbar(im, cax=cax)
        ax = fig.add_axes((.65, .4, .3 * 1.5, .3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=.05, pad=0.1)
        im = ax.imshow(torch.stack([torch.stack([x[0] for x in y]) for y in layer_optim_dist]).cpu(), aspect='auto',
                       interpolation='none')
        ax.set_yticks(np.arange(var_explained.shape[0]))
        ax.set_xticks(np.arange(var_explained.shape[0]))
        ax.set_yticklabels([])
        ax.set_xticklabels([], rotation=90)

        cbar = fig.colorbar(im, cax=cax)
        ax = fig.add_axes((.2, .1, .8, .12))
        ax.scatter(np.arange(dist_val.cpu().shape[0]), dist_val.cpu(), zorder=3)
        ax.scatter(np.arange(dist_val_optim.cpu().shape[0]), Pereira_ordered.cpu(), zorder=4)

        ax.set_xlim((-1, dist_val_optim.cpu().shape[0] + 1))
        ax.set_ylim((0 - .05, np.max(dist_val_optim.cpu().numpy()) + .05))
        closest_points = [int(np.where(dist_idx.cpu().numpy() == x)[0]) for x in res['layer_spec']]
        [ax.scatter(x, dist_val[x].cpu().numpy(), 50, color=(0, 0, 0), zorder=5) for x in closest_points]
        [ax.scatter(x, Pereira_ordered[x].cpu().numpy(), 50, color=(1, 0, 0), zorder=5) for x in closest_points]
        ax.set_xticks(tuple(np.arange(dist_val.cpu().shape[0])))
        ax.set_xticklabels(dist_idx.cpu().numpy())
        ax.set_xticklabels([model_layers[int(x)] for x in dist_idx.cpu().numpy()], rotation=90)
        [ax.plot([x, x], plt.ylim(), 'k-', zorder=2) for x in closest_points]

        ax.set_title(f"{res['extractor_name']} , \n {res['optimizatin_name']}", fontsize=12)

        plt.savefig(os.path.join(ANALYZE_DIR, f"{res['extractor_name']}_{res['optimizatin_name']}_RDM.png"), dpi=None,
                    facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True, bbox_inches=None, pad_inches=0.1, frameon=False)
