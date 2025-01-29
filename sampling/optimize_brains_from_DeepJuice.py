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
from scipy.stats import median_abs_deviation as mad
from benchmarks import NSDBenchmark, NSDSampleBenchmark
from deepjuice._backends.cupyfy import convert_to_tensor
import multiprocessing
import os
import argparse



parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extract_mode', type=str, default='original')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
args = parser.parse_args()

if __name__ == '__main__':
    optim_id = args.optimizer_id
    extract_mode = args.extract_mode
    extract_mode='redux'
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    ext_obj=extract_pool[extract_id]()
    deepjuice_identifier=f'group=deepjuice_brains-dataset=nsd-{extract_mode}-bench=None-ave=False'
    ext_obj.identifier=deepjuice_identifier

    selected_models=['subject_1',
                     'subject_2',
                     'subject_3',
                     'subject_4']
    #%%
    benchmark_ = NSDBenchmark(path_dir=benchmark_path)
    x_fmri = (benchmark_.response_data.to_numpy()).T
    roi_indices = benchmark_.get_roi_indices(row_number=True)
    rois = roi_indices.keys()
    roi = 'OTC'
    fmri_roi_sub_x = [x_fmri[:, indx] for indx in roi_indices[roi].values()]

    #%%
    activations_list=[]
    layers_list=[]
    # for to deepjuice path and find model activation in the format
    for idx, model_ in enumerate(selected_models):

        layer_id = 'OTC'
        act_=fmri_roi_sub_x[idx]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)



    optim_id='coordinate_ascent_eh-obj=D_s-n_iter=100-n_samples=80-n_init=1-low_dim=False-pca_var=0.95-pca_type=sklearn-run_gpu=True'
    #optim_id = 'coordinate_ascent_eh-obj=2-D_s_jsd-n_iter=2-n_samples=100-n_init=1-low_dim=False-pca_var=0.95-pca_type=sklearn-run_gpu=True'
    #optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=100-n_samples=100-n_init=1-low_dim=False-pca_var=0.95-pca_type=sklearn-run_gpu=True'

    optim_obj=optim_pool[optim_id]()
    optim_obj.N_S=1000
    optim_obj.extract_type='activation'
    optim_obj.activations = activations_list
    optim_obj.extractor_obj=ext_obj
    optim_obj.early_stopping=False

    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                                 save_results=False)


    #xy_list=load_obj(Path(deepjuice_path,'test_xy_corr_list.pkl').__str__())
    #optim_obj.XY_corr_list=xy_list
    # create a random sample of 80 sample between 0 and 999
    jsd_range=[]
    for kk in tqdm(range(1000)):
        True
        S = np.random.choice(1000, 80, replace=False)
        # compute objective function for the random sample
        _,_,jsds=optim_obj.gpu_object_function_ds_plus_jsd(S,debug=True)
        jsd_range.append(torch.stack(jsds).mean().cpu().numpy())
    jsd_rnd=np.mean(jsd_range)
    optim_obj.jsd_threshold=jsd_rnd
    optim_obj.jsd_muliplier=80


    S_opt_d, DS_opt_d = optim_obj()

    #_,_,jsd_optim=optim_obj.gpu_object_function_ds_plus_jsd(S_opt_d,debug=True)
    #jsd_o=torch.stack(jsd_optim).mean().cpu().numpy()
    # find the instance that jsd_o is larger than jsd_rnd
    #1-np.sum(jsd_o>np.stack(jsd_range))/len(jsd_range)
    #optim_obj.gpu_object_function_ds_plus_jsd(S, debug=True)
    #2-optim_obj.gpu_object_function_ds(S_opt_d)

    optim_results = dict(extractor_name=deepjuice_identifier,
                         model_spec=selected_models,
                         layer_spec=layers_list,
                         optimizatin_name=optim_id,
                         optimized_S=S_opt_d,
                         optimized_d=DS_opt_d)



    (extract_short_hand, optim_short_hand) = make_shorthand(deepjuice_identifier, optim_id)
    optim_file = Path(RESULTS_DIR, f"results_{extract_short_hand}_{optim_short_hand}_{extract_mode}_jsd_mult_{optim_obj.jsd_muliplier}.pkl")

    save_obj(optim_results, optim_file.__str__())