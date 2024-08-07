import sys
from sent_sampling.utils.optim_utils import optim_pool, optim,optim_configuration
import argparse
from sent_sampling.utils import extract_pool
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
from sent_sampling.utils import make_shorthand
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import xarray as xr
deepjuice_path='/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
from glob import glob
import pickle
import argparse
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
parser.add_argument('multiplier', type=float, default=1)
parser.add_argument('threshold', type=float, default=0.05)
args = parser.parse_args()

if __name__ == '__main__':
    optimizer_id = args.optimizer_id
    jsd_muliplier = float(args.multiplier)
    jsd_threshold = float(args.threshold)

    #sd_muliplier=5
    #jsd_threshold=0.05
    extract_mode='redux'
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    extractor_obj=extract_pool[extract_id]()
    deepjuice_identifier=f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    extractor_obj.identifier=deepjuice_identifier

    selected_models=['torchvision_alexnet_imagenet1k_v1',
                    'torchvision_regnet_x_800mf_imagenet1k_v2',
                     'openclip_vit_b_32_laion2b_e16',
                     'timm_swinv2_cr_tiny_ns_224',
                     'torchvision_efficientnet_b1_imagenet1k_v2',
                     'clip_rn50',
                     'timm_convnext_large_in22k']

    activations_list=[]
    layers_list=[]
    # for to deepjuice path and find model activation in the format
    for model_ in selected_models:
        save_file = f'{deepjuice_path}/nsd/{model_}*{extract_mode}.pkl'
        original_files = glob(save_file)
        # open the file
        with open(original_files[0], 'rb') as f:
            original = pickle.load(f)
        layer_id = original[0]
        act_=original[1]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)

    #optimizer_id = f"coordinate_ascent_eh-obj=D_s_jsd-n_iter=50-n_samples=80-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"

    optimizer_obj=optim_pool[optimizer_id]()
    optimizer_obj.N_S=1000
    optimizer_obj.extract_type='activation'
    optimizer_obj.activations = activations_list
    optimizer_obj.extractor_obj=extractor_obj
    optimizer_obj.early_stopping=False

    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                                 save_results=False)

    idx = np.triu_indices(optimizer_obj.XY_corr_list[0].shape[0], k=1)
    max_vals = []
    min_vals = []
    for i_m in range(len(optimizer_obj.XY_corr_list)):
        vals = optimizer_obj.XY_corr_list[i_m][idx]
        print(f'model: {extractor_obj.model_spec[i_m]}, min: {vals.min()}, max: {vals.max()}')
        # round max to nearest 0.1 and drop the decimal part
        max_vals.append(torch.ceil(vals.max() * 10) / 10)
        min_vals.append(torch.floor(vals.min() * 10) / 10)

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
        XY_corr_hist = [torch.histc(x_ref, bins=bins, min=min_vals[id_m], max=max_vals[id_m]) for id_m, x_ref in
                        enumerate(XY_corr_sample)]
        XY_corr_hist = [(hist_ref / torch.sum(hist_ref)) + epsilon for hist_ref in XY_corr_hist]
        XY_corr_hist = [p_smooth / p_smooth.sum() for p_smooth in XY_corr_hist]
        XY_corr_hist_list.append(torch.stack(XY_corr_hist))

    XY_corr_hist_mean = torch.stack(XY_corr_hist_list, dim=-1).mean(dim=-1)
    XY_corr_hist_mean = XY_corr_hist_mean / XY_corr_hist_mean.sum(dim=-1, keepdim=True)
    XY_corr_hist = torch.stack(XY_corr_hist, dim=0)

    jsd_rnd = []
    for kk in tqdm(range(1000)):
        S = np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False)
        # compute objective function for the random sample
        samples = torch.tensor(S, dtype=torch.long, device=optimizer_obj.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]].to(optimizer_obj.device) for XY_corr in
                          optimizer_obj.XY_corr_list]
        XY_corr_hist = [torch.histc(x_ref, bins=bins, min=min_vals[id_m], max=max_vals[id_m]) for id_m, x_ref in
                        enumerate(XY_corr_sample)]
        XY_corr_hist = [(hist_ref / torch.sum(hist_ref)) + epsilon for hist_ref in XY_corr_hist]
        XY_corr_hist = [p_smooth / p_smooth.sum() for p_smooth in XY_corr_hist]
        XY_corr_hist = torch.stack(XY_corr_hist, dim=0)
        p_smooth = XY_corr_hist
        q_smooth = XY_corr_hist_mean
        m = 0.5 * (p_smooth + q_smooth)
        kl_div_pm = (p_smooth * (p_smooth.log() - m.log())).mean(dim=-1)
        # Compute KL divergence between q and m
        kl_div_qm = (q_smooth * (q_smooth.log() - m.log())).mean(dim=-1)
        jsd = 0.5 * (kl_div_pm + kl_div_qm)

        jsd_rnd.append(jsd)

    jsd_rnd = torch.stack(jsd_rnd)
    jsd_rnd_max = jsd_rnd.max(dim=0)[0]

    optimizer_obj.XY_corr_hist_mean = XY_corr_hist_mean
    optimizer_obj.bins = bins
    optimizer_obj.jsd_rnd_max = jsd_muliplier * jsd_rnd_max
    optimizer_obj.epsilon = 1e-10
    optimizer_obj.jsd_threshold = jsd_threshold
    optimizer_obj.jsd_muliplier = jsd_muliplier
    optimizer_obj.corr_min_max = list(zip(min_vals, max_vals))




    (extract_short_hand, optim_short_hand) = make_shorthand(deepjuice_identifier, optimizer_id)
    optim_file = Path(RESULTS_DIR, f"results_{extract_short_hand}_{optim_short_hand}_{extract_mode}_jsd_thr_{optimizer_obj.jsd_threshold}_mult_{jsd_muliplier}_bins_{optimizer_obj.bins}_norm.pkl")

    if os.path.exists(optim_file):
        print(f"file {optim_file} exists, skipping optimization")
        optim_results = load_obj(optim_file)
        S_opt_d = optim_results['optimized_S']
        DS_opt_d = optim_results['optimized_d']
    else:
        S_opt_d, DS_opt_d = optimizer_obj()
        optim_results = dict(extractor_name=deepjuice_identifier,
                             model_spec=selected_models,
                             layer_spec=layers_list,
                             optimizatin_name=optimizer_id,
                             optimized_S=S_opt_d,
                             optimized_d=DS_opt_d,
                             bins=bins,
                             epsilon=epsilon,
                            jsd_threshold=optimizer_obj.jsd_threshold,
                            jsd_muliplier=optimizer_obj.jsd_muliplier,
                            corr_min_max=list(zip(min_vals, max_vals)),
                            jsd_max_mult=jsd_muliplier)
        # check of path is too long
        save_obj(optim_results, optim_file)



