from utils import extract_pool
from utils.optim_utils import optim_pool, optim_group
import argparse
from utils.data_utils import RESULTS_DIR, save_obj
import os
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
args = parser.parse_args()

if __name__ == '__main__':
    extract_name='albert_roberta_layer_compare_v1'
    extract_id = ['group=albert-xxlarge-v2_layer_compare_v1-dataset=coca_spok_filter_punct_10K_sample_1-activation-bench=None-ave=False',
        'group=roberta-base_layer_compare_v1-dataset=coca_spok_filter_punct_10K_sample_1-activation-bench=None-ave=False']
    optim_id = args.optimizer_id
    print(optim_id + '\n')
    #optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=50-n_init=2-run_gpu=True'
    # extract data
    optimizer_obj = optim_pool[optim_id]()
    optim_group_obj = optim_group(n_init=optimizer_obj.n_init,
                                  ext_group_ids=extract_id,
                                  n_iter=optimizer_obj.n_iter,
                                  N_s=optimizer_obj.N_s,
                                  objective_function=optimizer_obj.objective_function,
                                  optim_algorithm=optimizer_obj.optim_algorithm,
                                  run_gpu=optimizer_obj.run_gpu)
    # extract and constrcut low dim reprensetation
    optim_group_obj.load_extr_grp_and_corr_rdm_in_low_dim()
    # optimize
    S_opt_d, DS_opt_d = optim_group_obj()
    # save results
    optim_results = dict(extractor_grp_name=extract_id,
                         optimizatin_name=optim_id,
                         optimized_S=S_opt_d,
                         optimized_d=DS_opt_d)
    optim_file=os.path.join(RESULTS_DIR,f"results_{extract_name}_{optim_id}_low_dim.pkl")
    save_obj(optim_results, optim_file)