from sent_sampling.utils import extract_pool
from sent_sampling.utils.extract_utils import model_extractor
from sent_sampling.utils.optim_utils import optim_pool
import argparse
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj
import os
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from collections import namedtuple

def get_args():
    parser = argparse.ArgumentParser(description='extract activations from a model')
    parser.add_argument('model_name', type=str,
                        default='bert-base-uncased')
    parser.add_argument('dataset', type=str, default='ud_sentences_filter_v3_sample')
    args = parser.parse_args()
    return args

def mock_get_args():
    mock_args = namedtuple('debug', ['model_name', 'dataset'])
    new_args = mock_args('gpt2-xl', 'ud_sentencez_token_filter_v3_sample')
    return new_args

debug=False

if __name__ == '__main__':
    if debug:
        args = mock_get_args()
    else:
        args = get_args()

    model_id = args.model_name
    dataset_id = args.dataset
    print(model_id+'\n')
    print(dataset_id+'\n')
    # extract data
    datafile=[x['file_loc'] for x in SENTENCE_CONFIG if x['name']==dataset_id][0]
    extractor_obj = model_extractor(dataset=dataset_id,stim_type='textPeriod', datafile=datafile, model_spec=model_id,average_sentence=False)
    extractor_obj.load_dataset()
    extractor_obj(overwrite=False)