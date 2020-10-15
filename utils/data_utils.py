import pickle
from tqdm import tqdm
from neural_nlp.stimuli import StimulusSet
import getpass
import os
if getpass.getuser()=='eghbalhosseini':
    UD_PARENT='/Users/eghbalhosseini/MyData/Universal Dependencies 2.6/'
    CACHING_DIR='/Users/eghbalhosseini/.result_caching/neural_nlp.score'
elif getpass.getuser()=='ehoseini':
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
    CACHING_DIR = '/om/user/ehoseini/.result_caching/neural_nlp.score'
else:
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
print(UD_PARENT)
def save_obj(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_obj(filename_):
    print(filename_)
    with open(filename_, 'rb') as f:
        return pickle.load(f)

def construct_stimuli_set(stimuli_data, stimuli_data_name):
    sentence_words, word_nums, sentenceID = [], [], []
    for sent_ind, sentence in tqdm(enumerate(stimuli_data)):
        for word_ind, word in enumerate(sentence['word_FORM']):
            if word == '\n':
                continue
            word = word.rstrip('\n')
            sentence_words.append(word)
            word_nums.append(word_ind)
            sentenceID.append(sent_ind)
    word_number = list(range(len(sentence_words)))

    zipped_lst = list(zip(sentenceID, word_number, sentence_words))
    sentence_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word'])
    sentence_set.name = stimuli_data_name
    return sentence_set

BENCHMARK_CONFIG=dict(file_loc=CACHING_DIR)

SENTENCE_CONFIG = [
    dict(name='ud_sentences', file_loc=os.path.join(UD_PARENT,'ud_sentence_data.pkl')),
    dict(name='ud_sentences_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter.pkl')),
    dict(name='ud_sentences_filter_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter_sample.pkl')),
    dict(name='ud_sentences_token_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter.pkl')),
    dict(name='ud_sentences_token_filter_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter_sample.pkl'))]
