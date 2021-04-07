from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj
import os

if __name__ == '__main__':
    extractor_ids = ['group=gpt2-xl_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
                     'group=ctrl_layer_compare_v1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False']
    optimizer_id = 'coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=50-n_init=2-run_gpu=True'

    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    optimizer_obj.precompute_corr_rdm_on_gpu(low_dim=True,low_resolution=True)
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
    optim_file=os.path.join(RESULTS_DIR,f"results_{extractor_id}_{optimizer_id}_low_dim.pkl")
    save_obj(optim_results, optim_file)