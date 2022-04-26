from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj
import os
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
args = parser.parse_args()

if __name__ == '__main__':
    extractor_id = args.extractor_id

    print(extractor_id+'\n')
    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()