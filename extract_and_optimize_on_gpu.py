import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR, load_obj
import os
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')

args = parser.parse_args()

if __name__ == '__main__':
    extractor_id = args.extractor_id
    optimizer_id = args.optimizer_id

    #optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=125-n_init=1-low_dim=False-run_gpu=True"
    #optimizer_id = f"coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=125-n_init=1-low_dim=True-run_gpu=False"
    #extractor_id = f'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textPeriod-activation-bench=None-ave=False'
    low_resolution='False'
    print(extractor_id+'\n')
    print(optimizer_id+'\n')
    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    optimizer_obj.early_stopping=False
    xy_dir=os.path.join(SAVE_DIR, f"{optimizer_obj.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}-low_dim={optimizer_obj.low_dim}.pkl")
    if os.path.exists(xy_dir):
        print('loading precomputed correlation matrix ')
        D_precompute=load_obj(xy_dir)
        optimizer_obj.XY_corr_list=D_precompute
    else:
        print('precomputing correlation matrix ')
        optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution,cpu_dump=True,preload=False,save_results=True)
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
