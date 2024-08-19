import sys
from sent_sampling.utils.optim_utils import optim_pool, optim,optim_configuration
from sent_sampling.utils import extract_pool
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
from sent_sampling.utils import make_shorthand
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
deepjuice_path='/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
from benchmarks import NSDBenchmark, NSDSampleBenchmark
from deepjuice._backends.cupyfy import convert_to_tensor
import getpass
if getpass.getuser() == 'ehoseini':
    sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
    image_paths = '/om2/user/ehoseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/om2/user/ehoseini/MyData/neural_nlp_bench/activations/DeepJuice_DsParametricfMRI/'
    benchmark_path = '/om2/user/ehoseini/MyData/DeepJuice/nsd_data/'
else:
    sys.path.append('/Users/eghbalhosseini/MyCodes/DeepJuiceDev/')
    image_paths = '/Users/eghbalhosseini/MyData/DeepJuice/NSD_image_paths.pkl'
    deepjuice_ws_path = '/Users/eghbalhosseini/MyData/DeepJuice/workspace/nsd/'
    benchmark_path = '/Users/eghbalhosseini/MyData/DeepJuice/nsd_data/'

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
    kl_multiplier = float(args.multiplier)
    kl_threshold = float(args.threshold)

    extract_mode = 'redux'
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    extractor_obj = extract_pool[extract_id]()
    deepjuice_identifier = f'group=deepjuice_brains-dataset=nsd-{extract_mode}-bench=None-ave=False'
    extractor_obj.identifier = deepjuice_identifier

    selected_models = ['subject_1',
                       'subject_2',
                       'subject_3',
                       'subject_4']

    benchmark_ = NSDBenchmark(path_dir=benchmark_path)
    x_fmri = (benchmark_.response_data.to_numpy()).T
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    rois = roi_indices.keys()
    roi = 'OTC'

    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices[roi].values()]
    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
    for idx, model_ in enumerate(selected_models):
        layer_id = 'OTC'
        act_ = fmri_roi_sub_x[idx]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)


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
        # XY_corr_hist = [torch.histc(x_ref, bins=bins, min=0, max=2) for id_m, x_ref in
        #                enumerate(XY_corr_sample)]
        XY_corr_hist = [(hist_ref / torch.sum(hist_ref)) + epsilon for hist_ref in XY_corr_hist]
        XY_corr_hist = [p_smooth / p_smooth.sum() for p_smooth in XY_corr_hist]

        XY_corr_hist_list.append(torch.stack(XY_corr_hist))

    XY_corr_hist_mean = torch.stack(XY_corr_hist_list, dim=-1).mean(dim=-1)
    # normlaize along the last dimension to get a probability distribution per each row
    XY_corr_hist_mean = XY_corr_hist_mean / XY_corr_hist_mean.sum(dim=-1, keepdim=True)
    XY_corr_hist = torch.stack(XY_corr_hist, dim=0)

    kl_div_pm = (XY_corr_hist * (XY_corr_hist.log() - XY_corr_hist_mean.log())).mean(dim=-1)

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
    optimizer_obj.kl_div_rnd_max = kl_multiplier * kl_div_rnd_max
    optimizer_obj.epsilon = 1e-10
    optimizer_obj.kl_div_threshold = kl_threshold
    optimizer_obj.kl_div_muliplier = kl_multiplier
    optimizer_obj.corr_min_max = list(zip(min_vals, max_vals))


    (extract_short_hand, optim_short_hand) = make_shorthand(deepjuice_identifier, optimizer_id)
    optim_file = Path(RESULTS_DIR, f"res_{extract_short_hand}_{optim_short_hand}_kl_thr_{optimizer_obj.kl_div_threshold}_mult_{optimizer_obj.kl_div_muliplier}_bins_{optimizer_obj.bins}_norm.pkl")

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
                             kl_div_threshold=optimizer_obj.kl_div_threshold,
                             kl_div_muliplier=optimizer_obj.kl_div_muliplier,
                             corr_min_max=list(zip(min_vals, max_vals)),
                             kl_div_max_mult=kl_multiplier)
        # check of path is too long
        save_obj(optim_results, optim_file)



