import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
import os
import numpy as np
import torch
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=-n_samples=100-n_init=1')

args = parser.parse_args()

if __name__ == '__main__':
    #extractor_id = args.extractor_id
    #optimizer_id = args.optimizer_id
    extractor_id = f'group=best_performing_pereira_1-dataset=pereira2018-243sentences-activation-bench=None-ave=False'
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=50-n_init=1-run_gpu=True"
    low_resolution='False'
    low_dim='False'
    print(extractor_id+'\n')
    print(optimizer_id+'\n')
    # extract data
    extractor_obj = extract_pool[extractor_id]()

    extractor_obj()
    extractor_obj.N_S=243
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)

    # load the corr rdm, its already computed
    #xy_dir = os.path.join(SAVE_DIR, f"{extractor_id}_XY_corr_list-low_res=True.pkl")
    #if os.path.exists(xy_dir):
    #    xy_list=load_obj(xy_dir)

    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False,cpu_dump=True,preload=False,save_results=True)


    S_opt_d, DS_opt_d = optimizer_obj()
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
    optim_file=os.path.join(RESULTS_DIR,f"results_{extractor_id}_{optimizer_id}.pkl")
    save_obj(optim_results, optim_file)

    ##
    ds_rand=[]
    ds_optim=[]
    for k in tqdm([50,75,100,125,150,175,200,225]):
        optimizer_id=f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples={k}-n_init=1-run_gpu=True"
        optim_file = os.path.join(RESULTS_DIR, f"results_{extractor_id}_{optimizer_id}.pkl")
        optim_res=load_obj(optim_file)
        ds_r=[]
        for kk in range(200):
            sent_random = list(np.random.choice(optimizer_obj.N_S, k,replace=False))
            ds_r.append(optimizer_obj.gpu_object_function(sent_random))
        ds_optim.append(optimizer_obj.gpu_object_function(optim_res['optimized_S']))
        ds_rand.append(ds_r)
    ds_pereira=optimizer_obj.gpu_object_function(list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_S,replace=False)))
    import matplotlib
    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(26, 14), dpi=100, frameon=False)

    cmap = cm.get_cmap('viridis_r')
    ax = plt.axes((.1, .1, .4, .4))
    cmap = cm.get_cmap('viridis_r')

    tick_l = []
    tick = []
    idx = 0
    for idx, sample in enumerate(([50,75,100,125,150,175,200,225])):
        Ds_rand = np.asarray(ds_rand[idx])
        ax.scatter(.1 * np.random.normal(size=(np.asarray(Ds_rand).shape)) + idx, np.asarray(Ds_rand),
                   color=(.2, .2, .2), s=2, alpha=.5)
        ax.scatter(idx, np.asarray(Ds_rand).mean(), color=(0, 0, 0), s=50,label='random')

        ax.scatter(idx, ds_optim[idx], color=(1, 0, 0), s=50, label=f'optimized')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    # ax.set_xlim((0,6))
    # ax.set_ylim((0,1))
    ax.scatter(8,ds_pereira,color=(0, 0, 1), s=100,label='full set')
    ax.set_xticks(list(range(len([50,75,100,125,150,175,200,225,243]))))
    ax.set_xticklabels([50,75,100,125,150,175,200,225,243], rotation=0)
    # ax.tick_params(direction='out', length=3, width=2, colors='k',
    #                grid_color='k', grid_alpha=0.5)
    ax.set_ylabel(r'$D_s$')
    #ax.legend(bbox_to_anchor=(.1, .5), frameon=True)
    ax.set_title('Ds values for Pereira sentences')
    fig.show()
    #
    fig.savefig(os.path.join(RESULTS_DIR, f'Ds_for_{extractor_id}.png'),
                dpi=300, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(RESULTS_DIR, f'Ds_for_{extractor_id}.pdf'),
                format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1)

