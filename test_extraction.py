import importlib
#import utils
#importlib.reload(utils)
from utils.data_utils import SENTENCE_CONFIG
from neural_nlp.models import model_pool





import utils.extract_utils
importlib.reload(utils.extract_utils)
from utils.extract_utils import model_extractor

dataset='coca_spok_filter_punct_sample'
datafile=[x['file_loc'] for x in SENTENCE_CONFIG if x['name']==dataset][0]
model_name='gpt2'
test=model_extractor(dataset=dataset,datafile=datafile,model_spec=model_name,average_sentence=True)

test.load_dataset()
test()
