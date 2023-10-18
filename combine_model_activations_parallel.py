import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.extract_utils import model_extractor_parallel
import argparse
from utils.data_utils import SENTENCE_CONFIG
parser = argparse.ArgumentParser(description='extract activations from a model')
parser.add_argument('model_name', type=str,
                    default='bert-base-uncased')
parser.add_argument('dataset', type=str, default='ud_sentences_filter_v3_sample')
parser.add_argument('stim_type', type=str, default='textPeriod')
parser.add_argument('ave_mode', type=str, default='False')


args = parser.parse_args()

if __name__ == '__main__':
    model_id = args.model_name
    dataset_id = args.dataset
    ave_mode=args.ave_mode
    stim_type = args.stim_type
    print(model_id+'\n')
    print(dataset_id+'\n')
    print(ave_mode + '\n')
    # extract data
    datafile=[x['file_loc'] for x in SENTENCE_CONFIG if x['name']==dataset_id][0]
    extractor_obj = model_extractor_parallel(dataset=dataset_id, datafile=datafile, model_spec=model_id,average_sentence=ave_mode,stim_type=stim_type)
    extractor_obj.load_dataset()
    extractor_obj.total_runs
    extractor_obj.combine_runs(overwrite=False)