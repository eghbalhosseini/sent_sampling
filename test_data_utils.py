from neural_nlp.benchmarks.neural import read_words
from neural_nlp.models import model_pool, model_layers
from neural_nlp import FixedLayer

from utils.data_utils import ud_sentences_filter_set, ud_sentences_filter_sample_set

test=model_pool['gpt2']
layers=model_layers['gpt2']
candidate=FixedLayer(test,layers[1],prerun=None)
model_activations=read_words(candidate, ud_sentences_filter_sample_set, copy_columns=['stimulus_id'], average_sentence=False) #

