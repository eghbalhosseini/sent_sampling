import pickle
from tqdm import tqdm
from neural_nlp.stimuli import StimulusSet
import getpass
import os
import numpy as np
import pandas as pd
if getpass.getuser()=='eghbalhosseini':
    UD_PARENT = '/Users/eghbalhosseini/MyData/Universal Dependencies 2.6/'
    BENCHMARK_DIR = '/Users/eghbalhosseini/.result_caching/neural_nlp.score/'
    SAVE_DIR = '/Users/eghbalhosseini/MyData/sent_sampling/'
    RESULTS_DIR = '/Users/eghbalhosseini/MyData/sent_sampling/results/'
elif getpass.getuser()=='ehoseini':
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
    BENCHMARK_DIR = '/om/user/ehoseini/.result_caching/neural_nlp.score/'
    SAVE_DIR = '/om/user/ehoseini/MyData/sent_sampling/'
    RESULTS_DIR='/om/user/ehoseini/MyData/sent_sampling/results/'
elif getpass.getuser() == 'alexso':
    UD_PARENT = '/om/user/alexso/MyData/Universal Dependencies 2.6/'
    BENCHMARK_DIR = '/om/user/alexso/.result_caching/neural_nlp.score'
    SAVE_DIR = '/om/user/alexso/MyData/sent_sampling/'
    RESULTS_DIR = '/om/user/alexso/MyData/sent_sampling/results/'
else:
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'

UD_PATH = UD_PARENT+'/ud-treebanks-v2.6/'
GOOGLE10L_1T = UD_PARENT+'/Google10L-1T/'


LEX_PATH_SET = [
    {'name': 'AgeOfAcquisition', 'tag': 'AoA_ratings_Kuperman_et_al_BRM.xlsx', 'word_form': 'word_LEMMA',
     'word_column': 'Word',
     'metric_column': 'Rating.Mean',
     'url': 'http://crr.ugent.be/archives/806',
     'read_instruction': lambda x: pd.read_excel(x)},

    {'name': 'Concreteness', 'tag': 'Concreteness_ratings_Brysbaert_et_al_BRM.xlsx', 'word_form': 'word_LEMMA',
     'word_column': 'Word', 'metric_column': 'Conc.M',
     'url': 'http://crr.ugent.be/archives/1330',
     'read_instruction': lambda x: pd.read_excel(x)},

    {'name': 'Prevalence', 'tag': 'English_Word_Prevalences.xlsx', 'word_form': 'word_LEMMA', 'word_column': 'Word',
     'metric_column': 'Prevalence',
     'url': 'https://osf.io/nbu9e/',
     'read_instruction': lambda x: pd.read_excel(x)},

    {'name': 'Arousal', 'tag': 'Ratings_Warriner_et_al.csv', 'word_form': 'word_LEMMA', 'word_column': 'Word',
     'metric_column': 'A.Mean.Sum',
     'url': 'http://crr.ugent.be/archives/1003',
     'read_instruction': lambda x: pd.read_csv(x)},

    {'name': 'Valence', 'tag': 'Ratings_Warriner_et_al.csv', 'word_form': 'word_LEMMA', 'word_column': 'Word',
     'metric_column': 'V.Mean.Sum',
     'url': 'http://crr.ugent.be/archives/1003',
     'read_instruction': lambda x: pd.read_csv(x)},

    {'name': 'Ambiguity', 'tag': 'SUBTLEX-US frequency list with PoS and Zipf information.xlsx',
     'word_form': 'word_LEMMA',
     'word_column': 'Word', 'metric_column': 'Percentage_dom_PoS',
     'url': 'https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus',
     'read_instruction': lambda x: pd.read_excel(x)},

    {'name': 'LexFreq', 'tag': 'SUBTLEX-US frequency list with PoS and Zipf information.xlsx',
     'word_form': 'word_LEMMA',
     'word_column': 'Word', 'metric_column': 'Lg10WF',
     'url': 'https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus',
     'read_instruction': lambda x: pd.read_excel(x)},

    {'name': 'surprisal_3', 'tag': os.path.join(GOOGLE10L_1T, 'surprisal-ENGLISH-3.txt'), 'word_form': 'word_LEMMA',
     'word_column': 'word', 'metric_column': 'surprisal',
     'url': 'http://colala.berkeley.edu/data/PiantadosiTilyGibson2011/Google10L-1T/',
     'read_instruction': lambda x: pd.read_csv(x, sep='\t', skiprows=7)}
]


# reference : https://universaldependencies.org/u/overview/tokenization.html
# reference : https://universaldependencies.org/format.html

UD_ENGLISH_PATH_SET = [{'name': 'UD_English-EWT', 'tag': 'en_ewt-ud', 'group': ['train', 'test', 'dev']},
                       {'name': 'UD_English-ESL', 'tag': 'en_esl-ud', 'group': ['train', 'test', 'dev']},
                       {'name': 'UD_English-GUM', 'tag': 'en_gum-ud', 'group': ['train', 'test', 'dev']},
                       {'name': 'UD_English-PUD', 'tag': 'en_pud-ud', 'group': ['test']},
                       {'name': 'UD_English-LinES', 'tag': 'en_lines-ud', 'group': ['train', 'test', 'dev']},
                       {'name': 'UD_English-GUMReddit', 'tag': 'en_gumreddit-ud', 'group': ['train', 'test', 'dev']}]
def uppercount(str_in):
    count=0
    for i in str_in:
        if(i.isupper()):
            count=count+1
    return count

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

BENCHMARK_CONFIG=dict(file_loc=BENCHMARK_DIR)

SENTENCE_CONFIG = [
    dict(name='ud_sentences', file_loc=os.path.join(UD_PARENT,'ud_sentence_data.pkl')),
    dict(name='ud_sentences_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter.pkl')),
    dict(name='ud_sentences_filter_v2',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter_v2.pkl')),
    dict(name='ud_sentences_filter_v3',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter_v3_no_dup.pkl')),
    dict(name='ud_sentences_filter_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter_sample.pkl')),
    dict(name='ud_sentences_token_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter.pkl')),
    dict(name='ud_sentences_token_filter_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter_sample.pkl'))]
