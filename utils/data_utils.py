import pickle
from tqdm import tqdm
from neural_nlp.stimuli import StimulusSet

def save_obj(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_obj( filename_):
    with open(filename_, 'rb') as f:
       return  pickle.load(f)

def construct_stimuli_set(stimuli_data,stimuli_data_name):
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


UD_SENTENCE_PKL='/Users/eghbalhosseini/MyData/Universal Dependencies 2.6/ud_sentence_data.pkl'
ud_sentence_data=load_obj(UD_SENTENCE_PKL)
ud_sentence_set=construct_stimuli_set(ud_sentence_data,'ud_sentences')


UD_SENTENCE_FILTER_PKL='/Users/eghbalhosseini/MyData/Universal Dependencies 2.6/ud_sentence_data_filter.pkl'
filter_sentence_data=load_obj(UD_SENTENCE_FILTER_PKL)
ud_sentences_filter_set=construct_stimuli_set(filter_sentence_data,'ud_sentences_filter')

UD_SENTENCE_SAMPLE_PKL='/Users/eghbalhosseini/MyData/Universal Dependencies 2.6/ud_sentence_data_filter_sample.pkl'
sample_filter_sentence_data=load_obj(UD_SENTENCE_SAMPLE_PKL)
ud_sentences_filter_sample_set=construct_stimuli_set(sample_filter_sentence_data,'ud_sentences_filter_sample')