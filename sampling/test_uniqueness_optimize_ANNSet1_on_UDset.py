import os
import pandas as pd
from tqdm import tqdm
from sent_sampling.utils.data_utils import RESULTS_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, pt_create_corr_rdm_short
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
import torch
from sent_sampling.utils import extract_pool, make_shorthand
import numpy as np
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('run_id', type=str, default='0')



if __name__ == '__main__':
    # load parser arguments
    args = parser.parse_args()
    run_id = str(args.run_id)
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=50-n_samples=25-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    extractor_obj = extract_pool[extract_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # extract ev sentences
    # find location of ev sentences in sentences

    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    low_resolution= False
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=False,save_results=False)
    S = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s, replace=False))
    optimizer_obj.s_init = S
    optimizer_obj.gpu_object_function_debug(S)
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
                         s_init=optimizer_obj.s_init,
                         )
    [ext_id, opt_id] = make_shorthand(extract_id, optimizer_id)

    optim_file = os.path.join(RESULTS_DIR,f"results_{ext_id}_{opt_id}_run_{run_id}.pkl")
    # check of path is too long
    save_obj(optim_results, optim_file)






