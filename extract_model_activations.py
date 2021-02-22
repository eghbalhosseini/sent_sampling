from utils import extract_pool
from utils.extract_utils import model_extractor
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj
import os
from utils.data_utils import SENTENCE_CONFIG
parser = argparse.ArgumentParser(description='extract activations from a model')
parser.add_argument('model_name', type=str,
                    default='bert-base-uncased')
parser.add_argument('dataset', type=str, default='ud_sentences_filter_v3_sample')

args = parser.parse_args()

if __name__ == '__main__':
    model_id = args.model_name
    dataset_id = args.dataset
    print(model_id+'\n')
    print(dataset_id+'\n')
    # extract data
    datafile=[x['file_loc'] for x in SENTENCE_CONFIG if x['name']==dataset_id][0]
    extractor_obj = model_extractor(dataset=dataset_id, datafile=datafile, model_spec=model_id,average_sentence=True)
    extractor_obj.load_dataset()
    extractor_obj()