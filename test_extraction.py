import importlib
#import utils
#importlib.reload(utils)
from utils.data_utils import SENTENCE_CONFIG
from neural_nlp.models import model_pool





import utils.extract_utils
importlib.reload(utils.extract_utils)
from utils.extract_utils import model_extractor

dataset='ud_sentences_filter_v3_sample'
datafile=[x['file_loc'] for x in SENTENCE_CONFIG if x['name']==dataset][0]
model_name='bert-base-uncased'
test=model_extractor(dataset=dataset,datafile=datafile,model_spec=model_name)

test.load_dataset()
test()
