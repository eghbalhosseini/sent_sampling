import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR, load_obj
import os
from pathlib import Path
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')

args = parser.parse_args()

def make_shorthand(extractor_id, optimizer_id):
    # create a short hand version for the file name
    # get group= from extract_id
    group = extractor_id.split('-')[0].split('=')[1]
    # get dataset= from extract_id
    dataset = extractor_id.split('-')[1].split('=')[1]
    # get activation from extract_id
    activation = extractor_id.split('-')[2]
    # get bench from extract_id
    bench = extractor_id.split('-')[3].split('=')[1]
    # get ave from extract_id
    ave = extractor_id.split('-')[4].split('=')[1]
    # get coord from optim_id
    coord = optimizer_id.split('-')[0]
    # make an auxilary name by removing coord from optimizer_id
    aux_name = '-'.join(optimizer_id.split('-')[1:])
    # get obj from aux_name by finding -n_iter and taking the first part
    obj = aux_name.split('-n_iter')[0].split('=')[1]
    # get n_iter from aux_name
    n_iter = aux_name.split('-n_iter')[1].split('-')[0].split('=')[1]
    # get n_samples from aux_name
    n_samples = aux_name.split('-n_samples')[1].split('-')[0].split('=')[1]
    # get n_init from aux_name
    n_init = aux_name.split('-n_init')[1].split('-')[0].split('=')[1]
    # get low_dim from aux_name
    low_dim = aux_name.split('-low_dim')[1].split('-')[0].split('=')[1]
    # get pca_var from aux_name
    pca_var = aux_name.split('-pca_var')[1].split('-')[0].split('=')[1]
    # get pca_type from aux_name
    pca_type = aux_name.split('-pca_type')[1].split('-')[0].split('=')[1]
    # get run_gpu from aux_name
    run_gpu = aux_name.split('-run_gpu')[1].split('-')[0].split('=')[1]
    # create a short hand name from the above
    optim_short_hand = f"[{coord}]-[O={obj}]-[Nit={n_iter}]-[Ns={n_samples}]-[Nin={n_init}]-[LD={low_dim}]-[V={pca_var}]-[T={pca_type}]-[GPU={run_gpu}]"
    optim_short_hand = optim_short_hand.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    # extract shorthand
    extract_short_hand = f"[G={group}]-[D={dataset}]-[{activation}]-[B={bench}]-[AVE={ave}]"
    extract_short_hand = extract_short_hand.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    return (extract_short_hand, optim_short_hand)

if __name__ == '__main__':
    extractor_id = args.extractor_id
    optimizer_id = args.optimizer_id

    #optimizer_id = f"coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=125-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
    #optimizer_id = f"coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=125-n_init=1-low_dim=True-run_gpu=False"
    #extractor_id = f'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textPeriod-activation-bench=None-ave=False'
    low_resolution='False'
    print(extractor_id+'\n')
    print(optimizer_id+'\n')
    (extract_short_hand, optim_short_hand) = make_shorthand(extractor_id, optimizer_id)
    optim_file = Path(RESULTS_DIR, f"results_{extract_short_hand}_{optim_short_hand}.pkl")
    assert len(optim_file.name) < 255 , "file name too long"
    print(optim_file.name + '\n')
    # make sure the size of optim file is less than max character length allowed by linux
    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    optimizer_obj.early_stopping=False
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=True, preload=True,
                                                 save_results=False)
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

    save_obj(optim_results, optim_file.__str__())
