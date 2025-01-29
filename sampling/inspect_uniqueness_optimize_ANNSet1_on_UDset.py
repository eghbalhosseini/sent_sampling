import os
import pandas as pd
from tqdm import tqdm
from sent_sampling.utils.data_utils import RESULTS_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, pt_create_corr_rdm_short
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj, ANALYZE_DIR
import torch
from sent_sampling.utils import extract_pool, make_shorthand
import numpy as np
from glob import glob
from pathlib import Path
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # load parser arguments

    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples=75-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    extract_id = "group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False"
    extractor_obj = extract_pool[extract_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # extract ev sentences
    # find location of ev sentences in sentences

    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    low_resolution= False
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=False,save_results=False)
    #S = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False))
    #optimizer_obj.s_init = S
    #optimizer_obj.gpu_object_function_debug(S)
    #S_opt_d, DS_opt_d = optimizer_obj()


    all_same_S=[]
    all_d_optimized=[]
    # find all the files with _run_%d.pkl using glob
    for num_samples in tqdm([25,50,75]):
        optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples={num_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
        optim_run = []
        [ext_id, opt_id] = make_shorthand(extract_id, optimizer_id)
        file_path=Path(os.path.join(RESULTS_DIR,f"results_{ext_id}_{opt_id}_run_*.pkl"))
        files=glob(file_path.__str__())

        for run_id in range(len(files)):
            optim_file = files[run_id]
            # check of path is too long
            optim_run.append(load_obj(optim_file))

        S_optimized=[x['optimized_S'] for x in optim_run]
        d_optimized=[x['optimized_d'] for x in optim_run]

    # for every pair of S_optimized find how many are the same with the rest of the S_optimized
        same_S=[]
        for i in range(len(S_optimized)):
            same_S.append([len(set(S_optimized[i]).intersection(set(x))) for x in S_optimized])
        all_same_S.append(same_S)
        all_d_optimized.append(d_optimized)


    # make a plot where on the x axis there are num_samples, on the y axis there is a scatter plot of d_optimized, and show the mean
    fig=plt.figure()
    ax=fig.add_subplot(121)
    x_values=[25,50,75]
    for i in range(len(all_d_optimized)):
        ax.scatter([x_values[i]]*len(all_d_optimized[i]),all_d_optimized[i],c='b',alpha=0.5,label='run' if i==0 else None)
    ax.plot(x_values,[np.mean(x) for x in all_d_optimized],c='r',label='mean',linewidth=2)
    # plot std as error bars
    ax.errorbar(x_values,[np.mean(x) for x in all_d_optimized],yerr=[np.std(x) for x in all_d_optimized],c='r',label='std',capsize=5)
    ax.set_xlabel(f'size of optimized set \n( n out of {optimizer_obj.N_S}) samples')
    ax.set_ylabel('Optimized distance across models')
    ax.set_ylim([1.12,1.155])
    ax.set_xticks(x_values)
    # show legend
    ax.legend()
    # do a second suplot where you show the number of same S
    ax=fig.add_subplot(122)
    all_sm=[]
    for i in range(len(all_same_S)):
        sam_s=np.asarray(all_same_S[i])
        np.triu_indices(sam_s.shape[1],k=1)
        sam_s=sam_s[np.triu_indices(sam_s.shape[1],k=1)]/x_values[i]
        ax.scatter([x_values[i]]*len(sam_s),sam_s,c='b',alpha=0.5,label='run')
        all_sm.append(sam_s)
    ax.plot(x_values,[np.mean(x) for x in all_sm],c='r',label='mean',linewidth=2)
    # plot std as error bars and show caps
    ax.errorbar(x_values,[np.mean(x) for x in all_sm],yerr=[np.std(x) for x in all_sm],c='r',label='std',capsize=5)
    ax.set_xlabel('size of optimized set')
    ax.set_xticks(x_values)
    ax.set_ylabel('ratio samples that are same across runs')
    plt.tight_layout()
    fig.show()
    # save the figure
    sample_str='_'.join([str(x) for x in x_values])
    fig.savefig(os.path.join(ANALYZE_DIR,f"uniqueness_of_sampling_{ext_id}_{opt_id}_samples={sample_str}.png"))





