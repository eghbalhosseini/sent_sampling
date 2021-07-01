import utils
from utils import extract_pool,model_grps_config
from utils.optim_utils import optim_pool, optim_group
from utils.data_utils import RESULTS_DIR, save_obj
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
import os
import torch
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
parser.add_argument('dataset', type=str, default='ud_sentences_filter_v3_sample')
args = parser.parse_args()

if __name__ == '__main__':
    #extract_name='gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layer_compare_v1'

    optim_id = args.optimizer_id
    print(optim_id + '\n')
    dataset_id = args.dataset
    print(dataset_id + '\n')


    extract_name = 'gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layers'

    extract_id = [f'group=gpt2-xl_layers-dataset={dataset_id}-activation-bench=None-ave=False',
                  f'group=ctrl_layers-dataset={dataset_id}-activation-bench=None-ave=False',
                  f'group=bert-large-uncased-whole-word-masking_layers-dataset={dataset_id}-activation-bench=None-ave=False',
                  f'group=gpt2_layers-dataset={dataset_id}-activation-bench=None-ave=False',
                  f'group=openaigpt_layers-dataset={dataset_id}-activation-bench=None-ave=False',
                 f'group=lm_1b_layers-dataset={dataset_id}-activation-bench=None-ave=False']


    # extract data
    optimizer_obj = optim_pool[optim_id]()
    optim_group_obj = optim_group(n_init=optimizer_obj.n_init,
                                  extract_group_name=extract_name,
                                  ext_group_ids=extract_id,
                                  n_iter=optimizer_obj.n_iter,
                                  N_s=optimizer_obj.N_s,
                                  objective_function=optimizer_obj.objective_function,
                                  optim_algorithm=optimizer_obj.optim_algorithm,
                                  run_gpu=optimizer_obj.run_gpu)
    # extract and constrcut low dim reprensetation
    if os.path.exists(os.path.join(SAVE_DIR,f"{optim_group_obj.extract_group_name}_XY_corr_list.pkl")):
        D_precompute=load_obj(os.path.join(SAVE_DIR, f"{optim_group_obj.extract_group_name}_XY_corr_list.pkl"))
        optim_group_obj.grp_XY_corr_list=D_precompute['grp_XY_corr_list']
        optim_group_obj.N_S=D_precompute['N_S']
    else:
        optim_group_obj.load_extr_grp_and_corr_rdm_in_low_dim()
    # optimize
    S_opt_d, DS_opt_d = optim_group_obj()
    # save results
    optim_results = dict(extractor_grp_name=extract_id,
                        optimizatin_name=optim_id,
                        optimized_S=S_opt_d,
                        optimized_d=DS_opt_d)
    optim_file=os.path.join(RESULTS_DIR,f"results_{extract_name}_{dataset_id}_{optim_id}_low_dim_gpu.pkl")
    save_obj(optim_results, optim_file)