import os
import pandas as pd
from tqdm import tqdm
from sent_sampling.utils.data_utils import RESULTS_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, pt_create_corr_rdm_short
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
import torch
from sent_sampling.utils import extract_pool, make_shorthand
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
parser.add_argument('multiplier', type=float, default=1)
parser.add_argument('threshold', type=float, default=0.05)

args = parser.parse_args()



if __name__ == '__main__':
    # load parser arguments
    optimizer_id = args.optimizer_id
    jsd_muliplier = float(args.multiplier)
    jsd_threshold = float(args.threshold)
    #optimizer_id = f"coordinate_ascent_eh-obj=D_s_jsd-n_iter=50-n_samples=225-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"

    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'

    extractor_obj = extract_pool[extract_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # extract ev sentences
    # find location of ev sentences in sentences

    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    low_resolution= False
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=True,save_results=False)

    idx = np.triu_indices(optimizer_obj.XY_corr_list[0].shape[0], k=1)
    max_vals=[]
    min_vals=[]
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

    XY_corr_hist_mean=torch.stack(XY_corr_hist_list,dim=-1).mean(dim=-1)
    XY_corr_hist_mean = XY_corr_hist_mean / XY_corr_hist_mean.sum(dim=-1, keepdim=True)
    XY_corr_hist=torch.stack(XY_corr_hist,dim=0)


    jsd_rnd = []
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
        p_smooth=XY_corr_hist
        q_smooth = XY_corr_hist_mean
        m = 0.5 * (p_smooth + q_smooth)
        kl_div_pm = (p_smooth * (p_smooth.log() - m.log())).mean(dim=-1)
        # Compute KL divergence between q and m
        kl_div_qm = (q_smooth * (q_smooth.log() - m.log())).mean(dim=-1)
        jsd = 0.5 * (kl_div_pm + kl_div_qm)

        jsd_rnd.append(jsd)

    jsd_rnd = torch.stack(jsd_rnd)
    jsd_rnd_max = jsd_rnd.max(dim=0)[0]

    # concatenate the list of XY_corr_sample_tensor along a new last dimension
    optimizer_obj.XY_corr_hist_mean = XY_corr_hist_mean
    optimizer_obj.bins = bins
    optimizer_obj.jsd_rnd_max = jsd_muliplier * jsd_rnd_max
    optimizer_obj.epsilon = 1e-10
    optimizer_obj.jsd_threshold = jsd_threshold
    optimizer_obj.jsd_muliplier = 1
    optimizer_obj.corr_min_max = list(zip(min_vals, max_vals))
    S_opt_d, DS_opt_d = optimizer_obj()

    optim_results = dict(extractor_name=extract_id,
                         model_spec=extractor_obj.model_spec,
                         layer_spec=extractor_obj.layer_spec,
                         data_type=extractor_obj.extract_type,
                         benchmark=extractor_obj.extract_benchmark,
                         average=extractor_obj.average_sentence,
                         optimizatin_name=optimizer_id,
                         optimized_S=S_opt_d,
                         optimized_d=DS_opt_d,
                         bins=bins,
                         epsilon=epsilon,
                        jsd_threshold=optimizer_obj.jsd_threshold,
                        jsd_muliplier=optimizer_obj.jsd_muliplier,
                        corr_min_max=list(zip(min_vals, max_vals)),
                        jsd_max_mult=jsd_muliplier)

    [ext_id,opt_id]=make_shorthand(extract_id,optimizer_id)

    optim_file = os.path.join(RESULTS_DIR,
                              f"results_{ext_id}_{opt_id}_jsd_thr_{optimizer_obj.jsd_threshold}_mult_max_{jsd_muliplier}_bins_{optimizer_obj.bins}_norm.pkl")
    # check of path is too long
    save_obj(optim_results, optim_file)





