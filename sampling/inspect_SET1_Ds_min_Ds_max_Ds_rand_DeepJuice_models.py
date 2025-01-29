import os
import numpy as np
import sys
from pathlib import Path
import getpass
if getpass.getuser() == 'eghbalhosseini':
    SAMPLING_PARENT = '/Users/eghbalhosseini/MyCodes/sent_sampling'
    SAMPLING_DATA = '/Users/eghbalhosseini/MyCodes//fmri_DNN/ds_parametric/'

elif getpass.getuser() == 'ehoseini':
    SAMPLING_PARENT = '/om/user/ehoseini/sent_sampling'
    SAMPLING_DATA = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'


deepjuice_path='/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
from matplotlib.pyplot import GridSpec
import pandas as pd
from pathlib import Path
import torch
from sent_sampling.utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib
import scipy.io
matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 10,'font.weight':'bold'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False

import pickle
from glob import glob
if __name__ == '__main__':
    extract_mode='redux'
    n_samples=80
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id_min = f'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_id_max = f'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'

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

    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
    for model_ in tqdm(selected_models):
        save_file = f'{deepjuice_path}/nsd/{model_}*{extract_mode}.pkl'
        original_files = glob(save_file)
        # open the file
        with open(original_files[0], 'rb') as f:
            original = pickle.load(f)
        layer_id = original[0]
        act_ = original[1]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)


    optim_obj=optim_pool[optim_id_min]()
    optim_obj.N_S=1000
    optim_obj.extract_type='activation'
    optim_obj.activations = activations_list
    optim_obj.extractor_obj=ext_obj
    optim_obj.early_stopping=False
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=False, preload=True,
                                                 save_results=False)
    # read the excel that contains the selected sentences
    # %%  Load ds min and ds max data
    (extract_short_hand, optim_short_hand_min) = make_shorthand(deepjuice_identifier, optim_id_min)
    ds_min_path=f'{RESULTS_DIR}/results_{extract_short_hand}_{optim_short_hand_min}_{extract_mode}.pkl'
    with open(ds_min_path, 'rb') as f:
        results_ds_min = pickle.load(f)

    (extract_short_hand, optim_short_hand_max) = make_shorthand(deepjuice_identifier, optim_id_max)
    ds_max_path = f'{RESULTS_DIR}/results_{extract_short_hand}_{optim_short_hand_max}_{extract_mode}.pkl'
    with open(ds_max_path, 'rb') as f:
        results_ds_max = pickle.load(f)

    ds_min_loc = results_ds_min['optimized_S']
    ds_max_loc = results_ds_max['optimized_S']
    #%% create a ds_rand condition
    optim_id_random=f'coordinate_ascent_eh-obj=D_s_rand-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    (extract_short_hand, optim_short_rand) = make_shorthand(deepjuice_identifier, optim_id_random)
    ds_rand_path = f'{RESULTS_DIR}/results_{extract_short_hand}_{optim_short_rand}_{extract_mode}.pkl'
    # if path ds_rand_path exists, load it
    if Path(ds_rand_path).exists():
        with open(ds_rand_path, 'rb') as f:
            results_ds_rand = pickle.load(f)

    else:
        ds_rand = []
        RDM_rand = []
        sent_random_set=[]
        for k in tqdm(enumerate(range(1000))):
            sent_random = list(np.random.choice(optim_obj.N_S, optim_obj.N_s))
            d_s_r, RDM_r = optim_obj.gpu_object_function_debug(sent_random)
            ds_rand.append(d_s_r)
            RDM_rand.append(RDM_r)
            sent_random_set.append(sent_random)
        # find ds_rand closest to mean
        ds_rand_set = np.argmin(np.abs(np.mean(ds_rand) - np.array(ds_rand)))
        ds_rand_loc = sent_random_set[ds_rand_set]
        results_ds_rand = dict(extractor_name=deepjuice_identifier,
                             model_spec=selected_models,
                             layer_spec=layers_list,
                             optimizatin_name=optim_id_random,
                             optimized_S=ds_rand_loc,
                             optimized_d=ds_rand[ds_rand_set])

        optim_file = Path(RESULTS_DIR, f"results_{extract_short_hand}_{optim_short_rand}_{extract_mode}.pkl")
        save_obj(results_ds_rand, optim_file.__str__())
    ds_rand_loc = results_ds_rand['optimized_S']
    #%%
    d_all_loc = list(
        set(np.arange(0, 1000)) - set(ds_min_loc) - set(ds_rand_loc) - set(ds_max_loc))
    # make sure d_id_leftout and ds_min_image_ids dont share any elements
    assert len(set(d_all_loc).intersection(set(ds_min_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_rand_loc))) == 0
    assert len(set(d_all_loc).intersection(set(ds_max_loc))) == 0


    d_s_min, RDM_min = optim_obj.gpu_object_function_debug(ds_min_loc)
    d_s_rand, RDM_rand = optim_obj.gpu_object_function_debug(ds_rand_loc)
    d_s_max, RDM_max = optim_obj.gpu_object_function_debug(ds_max_loc)
    d_s_all, RDM_all = optim_obj.gpu_object_function_debug(d_all_loc)
    RDM_min=RDM_min.cpu()
    RDM_rand=RDM_rand.cpu()
    RDM_max = RDM_max.cpu()
    RDM_all = RDM_all.cpu()

    model_names_new_order=[0,4,2,1,3,6,5]
    model_names_new = [selected_models[i] for i in model_names_new_order]
    models_sh=['AlexNet','RegNet','ViT','Swin','EfficientNet','CLIP','ConvNext']
    model_sh_rotated=[models_sh[i] for i in model_names_new_order]
    # reorder the RDMs
    RDM_max = np.triu(RDM_max, k=1).T + np.triu(RDM_max, k=1)
    RDM_min = np.triu(RDM_min, k=1).T + np.triu(RDM_min, k=1)
    RDM_rand = np.triu(RDM_rand, k=1).T + np.triu(RDM_rand, k=1)
    RDM_all = np.triu(RDM_all, k=1).T + np.triu(RDM_all, k=1)

    RDM_max_new = RDM_max[model_names_new_order, :]
    RDM_max_new = RDM_max_new[:, model_names_new_order]
    RDM_min_new = RDM_min[model_names_new_order, :]
    RDM_min_new = RDM_min_new[:, model_names_new_order]
    RDM_rand_new = RDM_rand[model_names_new_order, :]
    RDM_rand_new = RDM_rand_new[:, model_names_new_order]
    RDM_all_new = RDM_all[model_names_new_order, :]
    RDM_all_new = RDM_all_new[:, model_names_new_order]
    #%%
    RDM_max = RDM_max_new
    RDM_min = RDM_min_new
    RDM_rand = RDM_rand_new
    RDM_all = RDM_all_new
    # create a dictionary with figure_3_data
    mask = np.triu(np.ones_like(RDM_max, dtype=bool))
    mask = np.where(mask, np.nan, 1)
    rdm_rand_vec = RDM_rand[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_max_vec = RDM_max[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_min_vec = RDM_min[np.tril_indices(RDM_max.shape[0], k=-1)]
    rdm_all_vec = RDM_all[np.tril_indices(RDM_max.shape[0], k=-1)]
    model_pairs = []
    for i in range(len(model_names_new)):
        for j in range(i + 1, len(model_names_new)):
            model_pairs.append((model_names_new[i], model_names_new[j]))

    figure_3_data = {'RDM_max': RDM_max, 'RDM_min': RDM_min, 'RDM_rand': RDM_rand,'RDM_all':RDM_all, 'rdm_rand_vec': rdm_rand_vec,
                        'rdm_max_vec': rdm_max_vec, 'rdm_min_vec': rdm_min_vec,'rdm_all_vec':rdm_all_vec, 'model_pairs': model_pairs,
                        'model_names': model_names_new, 'mask': mask}

    # get the actuall RDMS
    X_Max_min_rand = []
    S_ids = [ds_max_loc, ds_min_loc, ds_rand_loc, d_all_loc]
    for idx, S_id in enumerate(S_ids):
        X_=[]
        for XY_corr in optim_obj.XY_corr_list:
            pairs = torch.combinations(torch.tensor(S_id), with_replacement=False)
            X_sample = XY_corr[pairs[:, 0], pairs[:, 1]].cpu().numpy()
            # make squareform matrix
            X_sample = squareform(X_sample)
            X_.append(X_sample)
        X_Max_min_rand.append(X_)
    RDM_max_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_max_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_max_dict[model_name] = X_Max_min_rand[0][model_id]
    RDM_min_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_min_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_min_dict[model_name] = X_Max_min_rand[1][model_id]
    RDM_rand_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_rand_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_rand_dict[model_name] = X_Max_min_rand[2][model_id]
    RDM_all_dict = {model_name: [] for model_name in selected_models}
    for model_name in RDM_all_dict.keys():
        # get the model id from the model names
        model_id = selected_models.index(model_name)
        RDM_all_dict[model_name] = X_Max_min_rand[3][model_id]
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'ds_data_Parametric_deepJuice_n_{n_samples}_{extract_mode}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(figure_3_data, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_max_dict_parametric_deepJuice_n_{n_samples}_{extract_mode}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_max_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_min_dict_parametric_deepJuice_n_{n_samples}_{extract_mode}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_min_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_rand_dict_parametric_deepJuice_n_{n_samples}_{extract_mode}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_rand_dict, f)
    save_path = Path(ANALYZE_DIR, 'DsParametric', f'RDM_all_dict_parametric_deepJuice_n_{n_samples}_{extract_mode}.pkl')
    with open(save_path.__str__(), 'wb') as f:
        pickle.dump(RDM_all_dict, f)

