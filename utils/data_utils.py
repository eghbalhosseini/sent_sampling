import pickle
from tqdm import tqdm
from neural_nlp.stimuli import StimulusSet
import getpass
import os
import numpy as np
if getpass.getuser()=='eghbalhosseini':
    UD_PARENT = '/Users/eghbalhosseini/MyData/Universal Dependencies 2.6/'
    CACHING_DIR = '/Users/eghbalhosseini/.result_caching/neural_nlp.score'
    SAVE_DIR = '/Users/eghbalhosseini/MyData/sent_sampling/'
elif getpass.getuser()=='ehoseini':
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
    CACHING_DIR = '/om/user/ehoseini/.result_caching/neural_nlp.score'
    SAVE_DIR = '/om/user/ehoseini/MyData/sent_sampling/'
else:
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
print(UD_PARENT)
def save_obj(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_obj(filename_):
    print('loading '+ filename_)
    with open(filename_, 'rb') as f:
        return pickle.load(f)

def construct_stimuli_set(stimuli_data, stimuli_data_name):
    all_sentence_set=[]
    seq = np.floor(np.linspace(0, len(stimuli_data), num=10))
    seq_pair=np.vstack((seq[0:-1], seq[1:]))
    seq_pair=seq_pair.astype(np.int).transpose()
    num_row=seq_pair.shape[0]
    for row in range(num_row):
        sentence_words, word_nums, sentenceID = [], [], []
        for id, sent_id in tqdm(enumerate(range(seq_pair[row,0],seq_pair[row,1]))):
            sentence= stimuli_data[sent_id]
            for word_ind, word in enumerate(sentence['word_FORM']):
                if word == '\n':
                    continue
                if word =='.' and word_ind==len(sentence['word_FORM'])-1:
                    continue

                word = word.rstrip('\n')
                sentence_words.append(word)
                word_nums.append(word_ind)
                sentenceID.append(sent_id)
        word_number = list(range(len(sentence_words)))
        zipped_lst = list(zip(sentenceID, word_number, sentence_words))
        sentence_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word'])
        sentence_set.name = stimuli_data_name+'_group_'+str(row)
        all_sentence_set.append(sentence_set)
    return all_sentence_set

BENCHMARK_CONFIG=dict(file_loc=CACHING_DIR)

SENTENCE_CONFIG = [
    dict(name='ud_sentences', file_loc=os.path.join(UD_PARENT,'ud_sentence_data.pkl')),
    dict(name='ud_sentences_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter.pkl')),
    dict(name='ud_sentences_filter_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter_sample.pkl')),
    dict(name='ud_sentences_token_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter.pkl')),
    dict(name='ud_sentences_token_filter_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter_sample.pkl'))]
