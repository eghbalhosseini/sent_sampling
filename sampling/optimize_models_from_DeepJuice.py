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
parser.add_argument('extract_mode', type=str, default='original')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
args = parser.parse_args()

if __name__ == '__main__':
    optim_id = args.optimizer_id
    extract_mode = args.extract_mode
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    ext_obj=extract_pool[extract_id]()
    deepjuice_identifier=f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    ext_obj.identifier=deepjuice_identifier

    selected_models=['torchvision_alexnet_imagenet1k_v1',
                    'torchvision_regnet_x_800mf_imagenet1k_v2',
                     'openclip_vit_b_32_laion2b_e16',
                     'timm_swinv2_cr_tiny_ns_224',
                     'torchvision_efficientnet_b1_imagenet1k_v2',
                     'clip_rn50',
                     'timm_convnext_large_in22k',
                     ]

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



    #optim_id='coordinate_ascent_eh-obj=D_s-n_iter=100-n_samples=100-n_init=1-low_dim=False-pca_var=0.95-pca_type=sklearn-run_gpu=True'

    optim_obj=optim_pool[optim_id]()
    optim_obj.N_S=1000
    optim_obj.extract_type='activation'
    optim_obj.activations = activations_list
    optim_obj.extractor_obj=ext_obj
    optim_obj.early_stopping=False

    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=False,
                                                 save_results=True)


    #xy_list=load_obj(Path(deepjuice_path,'test_xy_corr_list.pkl').__str__())
    #optim_obj.XY_corr_list=xy_list

    S_opt_d, DS_opt_d = optim_obj()
    optim_results = dict(extractor_name=deepjuice_identifier,
                         model_spec=selected_models,
                         layer_spec=layers_list,
                         optimizatin_name=optim_id,
                         optimized_S=S_opt_d,
                         optimized_d=DS_opt_d)



    (extract_short_hand, optim_short_hand) = make_shorthand(deepjuice_identifier, optim_id)
    optim_file = Path(RESULTS_DIR, f"results_{extract_short_hand}_{optim_short_hand}_{extract_mode}.pkl")

    save_obj(optim_results, optim_file.__str__())