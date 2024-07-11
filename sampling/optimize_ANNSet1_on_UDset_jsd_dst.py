import os
import pandas as pd
from tqdm import tqdm
from sent_sampling.utils.data_utils import RESULTS_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, pt_create_corr_rdm_short
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
import torch
import numpy as np
from sent_sampling.utils import extract_pool, make_shorthand
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#args = parser.parse_args()
from scipy.stats import mannwhitneyu, ks_2samp
import os
import torch
from tqdm import tqdm
import numpy as np

from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

if __name__ == '__main__':
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'

    extractor_obj = extract_pool[extract_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # extract ev sentences
    # find location of ev sentences in sentences

    optimizer_id = f"coordinate_ascent_eh-obj=2-D_s_jsd_dst-n_iter=50-n_samples=225-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    low_resolution= False
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=False,
                                                 save_results=False)

    jsd_range = []
    ds_rand = []
    XY_corr_sample_list = []
    for kk in tqdm(range(25)):
        S = np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False)
        # compute objective function for the random sample
        ds_r, _, jsds = optimizer_obj.gpu_object_function_ds_grp_jsd(S, debug=True,minus=False)
        jsd_range.append(jsds)
        ds_rand.append(ds_r)
        # compute XY_pairs
        samples = torch.tensor(S, dtype=torch.long, device=optimizer_obj.device)
        pairs_rand = torch.combinations(samples, with_replacement=False).to('cpu')
        XY_corr_sample_rand = [XY_corr[pairs_rand[:, 0], pairs_rand[:, 1]].to(optimizer_obj.device) for XY_corr in
                               optimizer_obj.XY_corr_list]
        XY_corr_sample_tensor_rand = torch.stack(XY_corr_sample_rand).to(optimizer_obj.device)
        XY_corr_sample_tensor_rand = torch.transpose(XY_corr_sample_tensor_rand, 1, 0)
        if XY_corr_sample_tensor_rand.shape[1] < XY_corr_sample_tensor_rand.shape[0]:
            XY_corr_sample_tensor_rand = torch.transpose(XY_corr_sample_tensor_rand, 1, 0)
        assert (XY_corr_sample_tensor_rand.shape[1] > XY_corr_sample_tensor_rand.shape[0])
        XY_corr_sample_list.append(XY_corr_sample_tensor_rand)
    # concatenate the list of XY_corr_sample_tensor along a new last dimension
    optimizer_obj.XY_corr_random_sample_list = torch.stack(XY_corr_sample_list, dim=-1)
    optimizer_obj.jsd_max=np.stack(jsd_range).max(axis=0)
    optimizer_obj.jsd_threshold=.2


    S_opt_d, DS_opt_d = optimizer_obj()

    optim_results = dict(extractor_name=extract_id,
                         model_spec=extractor_obj.model_spec,
                         layer_spec=extractor_obj.layer_spec,
                         data_type=extractor_obj.extract_type,
                         benchmark=extractor_obj.extract_benchmark,
                         average=extractor_obj.average_sentence,
                         optimizatin_name=optimizer_id,
                         optimized_S=S_opt_d,
                         optimized_d=DS_opt_d)

    [ext_id,opt_id]=make_shorthand(extract_id,optimizer_id)

    optim_file = os.path.join(RESULTS_DIR,
                              f"results_{ext_id}_{opt_id}_jsd_thr_{optimizer_obj.jsd_threshold}_mult_{optimizer_obj.jsd_muliplier}_norm.pkl")
    # check of path is too long
    save_obj(optim_results, optim_file)

    S_opt_d_jsd=optim_results['optimized_S']

    [ds_jsd, _, jsd_jsd] = optimizer_obj.gpu_object_function_ds_grp_jsd(S_opt_d_jsd, debug=True)

    jsd_range = []
    js_optim_range = []
    for kk in tqdm(range(1000)):
        S = np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False)
        # compute objective function for the random sample
        ds_r, _, jsds = optimizer_obj.gpu_object_function_ds_grp_jsd(S, debug=True)
        _, _, jsds_optim = optimizer_obj.gpu_object_function_ds_grp_jsd(S_opt_d_jsd, debug=True)
        js_optim_range.append(jsds_optim)
        jsd_range.append(jsds)

    jsd_range = np.stack(jsd_range)
    js_optim_range = np.stack(js_optim_range)

    colors = [np.divide((0, 157, 255), 255), np.divide((128, 128, 128), 256), np.divide((255, 98, 0), 255)]
    # create a figure with 7 panels and each one plot a histogram of jsd_rand columns
    model_names = optimizer_obj.extractor_obj.model_spec
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio = 8 / 11
    for kk in range(7):
        ax = plt.axes((.1, .7 * (1 - kk / 7), .4, .06))
        modl_jsd_rand = jsd_range[:, kk]
        modl_optim = js_optim_range[:, kk]
        # find the max across all
        max_jsd = np.max([modl_jsd_rand.max(), modl_optim.max()])
        # create edges from 0 to max_jsd
        edges = np.linspace(0, max_jsd, 50)
        # plot histograms
        ax.hist(modl_jsd_rand, bins=edges, color=colors[1], alpha=0.5, label='rand')
        ax.hist(modl_optim, bins=edges, color=colors[0], alpha=0.5, label='min')
        # add model id
        ax.set_title(model_names[kk])

        # plot a vertical line at jsd_min, jsd_max and jsd_rand
    # save figure
    fig.show()
    fig.savefig(
        os.path.join(RESULTS_DIR,
                     f"jsd_rand_vs_optim_{ext_id}_{opt_id}_jsd_thr_{optimizer_obj.jsd_threshold}_mult_{optimizer_obj.jsd_muliplier}_norm.png"))
    # save as eps
    fig.savefig(
        os.path.join(RESULTS_DIR,
                     f"jsd_rand_vs_optim_{ext_id}_{opt_id}_jsd_thr_{optimizer_obj.jsd_threshold}_mult_{optimizer_obj.jsd_muliplier}_norm.eps"))

    ##
    X_Max = []
    S_id = S_opt_d_jsd
    for XY_ in optimizer_obj.XY_corr_list:
        pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
        X_sample = XY_[pairs[:, 0], pairs[:, 1]].cpu().numpy()
        # make squareform matrix
        X_sample = squareform(X_sample)
        X_Max.append(X_sample)

    X_rands_many = []
    for k in tqdm(enumerate(range(200))):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        x_rand_many = []
        for XY_ in optimizer_obj.XY_corr_list:
            pairs = torch.combinations(torch.tensor(sent_random), with_replacement=False)
            X_sample = XY_[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            x_rand_many.append(X_sample)
        X_rands_many.append(x_rand_many)

    X_rands_many_vec = []
    for X_rand in X_rands_many:
        X_rands_many_vec.append([X[np.tril_indices(X.shape[0], k=-1)] for X in X_rand])

    X_Max_vec = []
    for X_max in X_Max:
        X_Max_vec.append(X_max[np.tril_indices(X_max.shape[0], k=-1)])

    for idx in range(len(X_Max_vec)):
        x_max = np.asarray(X_Max_vec[idx])
        x_rand_vec = np.stack([X_rand[idx] for X_rand in X_rands_many_vec])
        # compute the correlation between x_max and each row of x_rand_vec
        max_to_rand_coeff = []
        for x in x_rand_vec:
            max_to_rand_coeff.append(np.corrcoef(x_max, x)[0, 1])
        max_to_rand_coeff = np.asarray(max_to_rand_coeff)
        # compute pairwise correlation between x_rand_vec rows
        rand_coeff = np.corrcoef(x_rand_vec)
        rand_coef_vec = rand_coeff[np.tril_indices(rand_coeff.shape[0], k=-1)]
        # check if max_to_rand_coeff and rand_coeff come from same distribution
        mw_stat, mw_p = mannwhitneyu(max_to_rand_coeff, rand_coef_vec)
        print(f"{model_names[idx]} Mann-Whitney U Test: statistic={mw_stat}, p-value={mw_p}")





