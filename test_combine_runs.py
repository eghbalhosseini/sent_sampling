import importlib
#import utils
#importlib.reload(utils)
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from neural_nlp.models import model_pool

import utils.extract_utils
importlib.reload(utils.extract_utils)
from sent_sampling.utils.extract_utils import model_extractor_parallel

dataset='ud_sentencez_token_filter_v3'
datafile=[x['file_loc'] for x in SENTENCE_CONFIG if x['name']==dataset][0]
model_name='openaigpt'
test=model_extractor_parallel(dataset=dataset,datafile=datafile,model_spec=model_name,average_sentence='None')

test.load_dataset()
test.combine_runs()
