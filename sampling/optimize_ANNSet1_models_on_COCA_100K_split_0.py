import sys
from sent_sampling.utils import extract_pool, make_shorthand
from sent_sampling.utils.optim_utils import optim_pool
import argparse
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
import os
import torch
from tqdm import tqdm
import numpy as np
#parser = argparse.ArgumentParser(description='extract activations and optimize')
#parser.add_argument('extractor_id', type=str,
#                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
#parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
#
#args = parser.parse_args()

if __name__ == '__main__':
    #extractor_id = args.extractor_id
    #optimizer_id = args.optimizer_id

    suffix = 'split_0'

    extractor_id = f'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_no_dup_100K_sample_1_{suffix}_textNoPeriod-activation-bench=None-ave=False'
    #optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=2-n_samples=200-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    #optimizer_id = f"coordinate_ascent_eh-obj=2-D_s_jsd-n_iter=2-n_samples=200-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    optimizer_id = f"coordinate_ascent_eh-obj=2-D_s_grp_jsd-n_iter=1-n_samples=200-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"

    [ext_id,opt_id]=make_shorthand(extractor_id,optimizer_id)
    # change activation to act
    #ext_id=ext_id.replace('activation','act')
    # change textNoPeriod to tNP
    #ext_id=ext_id.replace('textNoPeriod','tNP')
    print(extractor_id+'\n')
    print(optimizer_id+'\n')
    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    # load the corr rdm, its already computed
    low_resolution=False
    xy_pt = os.path.join(SAVE_DIR,
                         f"{optimizer_obj.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}-low_dim={optimizer_obj.low_dim}-pca_type={optimizer_obj.pca_type}-pca_var={optimizer_obj.pca_var}.pt")
    device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cpu')
    devices = [device1, device2]
    # use torch to load the file
    if os.path.exists(xy_pt):
        XY_corr = torch.load(xy_pt)
    # put the xy_corr element on device 0 if index is even else on device 1
    for idx in range(len(XY_corr)):
        XY_corr[idx] = XY_corr[idx].to(devices[idx % 2])

    optimizer_obj.XY_corr_list = XY_corr
    optimizer_obj.device=device1
    jsd_range = []
    for kk in tqdm(range(1000)):
        S = np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False)
        # compute objective function for the random sample
        _, _, jsds = optimizer_obj.gpu_object_function_ds_grp_jsd(S, debug=True)
        jsd_range.append(jsds)

    jsd_ave=np.stack(jsd_range).mean(axis=0)
    jsd_std=np.stack(jsd_range).std(axis=0)
    jsd_threshold=jsd_ave+1*jsd_std
    optimizer_obj.jsd_threshold=jsd_threshold
    optimizer_obj.jsd_muliplier=10
    S_opt_d, DS_opt_d = optimizer_obj()
    #[ds_,_,jsd_]=optimizer_obj.gpu_object_function_ds_grp_jsd(S_opt_d, debug=True)
    #2-optimizer_obj.gpu_object_function_debug(S_opt_d)[0]
    # save results
    optim_results = dict(extractor_name=extractor_id,
                         model_spec=extractor_obj.model_spec,
                         layer_spec=extractor_obj.layer_spec,
                         data_type=extractor_obj.extract_type,
                         benchmark=extractor_obj.extract_benchmark,
                         average=extractor_obj.average_sentence,
                         optimizatin_name=optimizer_id,
                         optimized_S=S_opt_d,
                         optimized_d=DS_opt_d)
    optim_file=os.path.join(RESULTS_DIR,f"results_{ext_id}_{opt_id}_jsd_{optimizer_obj.jsd_muliplier}.pkl")
    # check of path is too long
    save_obj(optim_results, optim_file)

    # load the results
    #optim_results=load_obj(optim_file)

    #S_opt_d=optim_results['optimized_S']
    #[ds_,_,jsd_]=optimizer_obj.gpu_object_function_ds_grp_jsd(S_opt_d, debug=True)
    optimizer_obj.gpu_object_function_ds_grp_jsd(S_opt_d)
# 2-optimizer_obj.gpu_object_function_debug(S_opt_d)[0]



