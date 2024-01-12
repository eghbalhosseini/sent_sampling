import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool
import argparse
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
from sent_sampling.utils.optim_utils import optim, Distance
import os
import torch
from tqdm import tqdm
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')

args = parser.parse_args()



if __name__ == '__main__':
    #extractor_id = args.extractor_id
    #optimizer_id = args.optimizer_id
    extract_id=f'group=best_performing_pereira_1-dataset=ud_sentences_U01_AnnSET1_ordered_for_RDM-activation-bench=None-ave=False'
    extractor_obj = extract_pool[extract_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=100-n_init=1-run_gpu=True"
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=False, save_results=False)
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
    optim_file = os.path.join(RESULTS_DIR, f"results_{extract_id}_{optimizer_id}.pkl")
    save_obj(optim_results, optim_file)

    ds_rand = []
    ds_optim = []
    for k in tqdm([50, 75, 100, 125, 150, 175 ]):
        optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples={k}-n_init=1-run_gpu=True"
        extract_id = f'group=best_performing_pereira_1-dataset=ud_sentences_U01_AnnSET1_ordered_for_RDM-activation-bench=None-ave=False'
        optim_file = os.path.join(RESULTS_DIR, f"results_{extract_id}_{optimizer_id}.pkl")
        optim_res = load_obj(optim_file)
        ds_r = []
        for kk in range(100):
            sent_random = list(np.random.choice(optimizer_obj.N_S, k, replace=False))
            ds_r.append(optimizer_obj.gpu_object_function(sent_random))
        ds_optim.append(optimizer_obj.gpu_object_function(optim_res['optimized_S']))
        ds_rand.append(ds_r)
    ds_pereira = optimizer_obj.gpu_object_function(
        list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_S, replace=False)))


    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(26, 14), dpi=300, frameon=False)

    cmap = cm.get_cmap('viridis_r')
    ax = plt.axes((.1, .1, .4, .4))
    cmap = cm.get_cmap('viridis_r')

    tick_l = []
    tick = []
    idx = 0
    for idx, sample in enumerate(([50, 75, 100, 125, 150, 175 ])):
        Ds_rand = np.asarray(ds_rand[idx])
        ax.scatter(.1 * np.random.normal(size=(np.asarray(Ds_rand).shape)) + idx, np.asarray(Ds_rand),
                   color=(.2, .2, .2), s=2, alpha=.5)
        ax.scatter(idx, np.asarray(Ds_rand).mean(), color=(0, 0, 0), s=50, label='random')

        ax.scatter(idx, ds_optim[idx], color=(1, 0, 0), s=50, label=f'optimized')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    # ax.set_xlim((0,6))
    # ax.set_ylim((0,1))
    ax.scatter(6, ds_pereira, color=(0, 0, 1), s=100, label='full set')
    ax.set_xticks(list(range(len([50, 75, 100, 125, 150, 175 ,200]))))
    ax.set_xticklabels([50, 75, 100, 125, 150, 175 ,200], rotation=0)
    # ax.tick_params(direction='out', length=3, width=2, colors='k',
    #                grid_color='k', grid_alpha=0.5)
    ax.set_ylabel(r'$D_s$')
    # ax.legend(bbox_to_anchor=(.1, .5), frameon=True)
    ax.set_title('Ds values for Pereira sentences')
    fig.show()
    # a=load_obj('/om/user/ehoseini/MyData/sent_sampling/results/act_ev_AnnSet1.pkl')
    # a.keys()
    # ##
    # act_list=[]
    # for idx, x in enumerate(a['model_names']):
    #     True
    #     act_=a['model_acts'][idx]
    #     act_list.append(dict(activations=act_,model_names=x))

    #extractor_name = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False'

    #xtract_pool.keys()
    #optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=50-n_init=1-run_gpu=True"
    #optimizer_obj = optim_pool[optimizer_id]()
    #optim_obj=optim(n_init=1,n_iter=300,N_s=100,objective_function=Distance,early_stopping=False)
    #optim_obj.optim_algorithm=optimizer_obj.optim_algorithm
    #optim_obj.activations=act_list
    #optim_obj.extract_type='activation'
    #optim_obj.N_S=200
    #optim_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=False, save_results=False)
    #ptimizer_obj.activations= act_list
    #optimizer_obj.N_S= 200

    #optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=False, save_results=False)



    #S_opt_d, DS_opt_d = optimizer_obj()
    # save results


