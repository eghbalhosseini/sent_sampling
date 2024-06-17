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
    COCA_CORPUS_DIR = '/Users/eghbalhosseini/MyData/COCA_corpus/parsed/'
    COCA_PREPROCESSED_DIR = '/Users/eghbalhosseini/MyData/COCA_corpus/preprocessed/'
    AUTO_ENCODER_DIR='/Users/eghbalhosseini/MyData/sent_sampling/auto_encoder/'
    ANALYZE_DIR = '/Users/eghbalhosseini/MyData/sent_sampling/analysis/'
    DSPARAMETRIC_DIR='/Users/eghbalhosseini/MyData//fmri_DNN/ds_parametric/'
elif getpass.getuser()=='ehoseini':
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
    COCA_CORPUS_DIR = '/nese/mit/group/evlab/u/ehoseini/MyData/COCA_corpus/parsed/'
    COCA_PREPROCESSED_DIR = '/nese/mit/group/evlab/u/ehoseini/MyData//COCA_corpus/preprocessed/'
    BENCHMARK_DIR = '/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.score/'
    SAVE_DIR = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/'
    RESULTS_DIR='/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/results/'
    ANALYZE_DIR = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/analysis/'
    AUTO_ENCODER_DIR = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/auto_encoder/'
    DSPARAMETRIC_DIR = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'
else:
    UD_PARENT = '/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
    COCA_CORPUS_DIR = '/om/user/ehoseini/MyData/COCA_corpus/parsed/'


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
        pickle.dump(di_, f, protocol=4)

def load_obj(filename_,silent=False):
    if not silent:
        print('loading '+ filename_)
    with open(filename_, 'rb') as f:
        return pickle.load(f)

def construct_stimuli_set(stimuli_data, stimuli_data_name):
    all_sentence_set=[]
    seq = np.floor(np.linspace(0, len(stimuli_data), num=10))
    seq_pair=np.vstack((seq[0:-1], seq[1:]))
    seq_pair=seq_pair.astype(int).transpose()
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

def construct_stimuli_set_from_text(stimuli_data, stimuli_data_name,drop_period=False,splits=20):
    all_sentence_set=[]
    seq = np.floor(np.linspace(0, len(stimuli_data), num=splits))
    seq_pair=np.vstack((seq[0:-1], seq[1:]))
    seq_pair=seq_pair.astype(int).transpose()
    num_row=seq_pair.shape[0]
    for row in range(num_row):
        sentence_words, word_nums, sentenceID = [], [], []
        for id, sent_id in tqdm(enumerate(range(seq_pair[row,0],seq_pair[row,1]))):
            sentence= stimuli_data[sent_id]
            if u'\xa0' in sentence['text']:
                sentence['text']=sentence['text'].replace(u'\xa0', u' ')
            words_from_text=sentence['text'].split(' ')
            if '.' in words_from_text[-1] and drop_period:
                words_from_text[-1]=words_from_text[-1].rstrip('.')
            # drop empty string
            words_from_text=[x for x in words_from_text if x]
            word_ind = np.arange(len(words_from_text))
            sent_ind = np.repeat(sent_id, len(words_from_text))
            sentence_words.append(words_from_text)
            word_nums.append(word_ind)
            sentenceID.append(sent_ind)
        sentence_words_flat = [item for sublist in sentence_words for item in sublist]
        sentenceID_flat=[item for sublist in sentenceID for item in sublist]
        word_number_flat = list(range(len(sentence_words_flat)))
        zipped_lst = list(zip(sentenceID_flat, word_number_flat, sentence_words_flat))
        sentence_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word'])

        sentence_set.name = f'{stimuli_data_name}_from_text_period_{drop_period}_group_{row}'
        all_sentence_set.append(sentence_set)
    return all_sentence_set


def construct_stimuli_set_no_grouping(stimuli_data, stimuli_data_name):
    all_sentence_set=[]
    seq = np.floor(np.linspace(0, len(stimuli_data), num=2))
    seq_pair=np.vstack((seq[0:-1], seq[1:]))
    seq_pair=seq_pair.astype(int).transpose()
    num_row=seq_pair.shape[0]
    for row in range(num_row):
        sentence_words, word_nums, sentenceID = [], [], []
        for id, sent_id in tqdm(enumerate(range(seq_pair[row,0],seq_pair[row,1]))):
            sentence= stimuli_data[sent_id]
            for word_ind, word in enumerate(sentence['text'].split()):
                word = word.rstrip('\n')
                sentence_words.append(word)
                word_nums.append(word_ind)
                sentenceID.append(sent_id)
        word_number = list(range(len(sentence_words)))
        zipped_lst = list(zip(sentenceID, word_number, sentence_words))
        sentence_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word'])

        sentence_set.name = stimuli_data_name+'_no_group'
        all_sentence_set.append(sentence_set)
    return all_sentence_set


def construct_stimuli_set_from_pd(stimuli_pd, stimuli_data_name='null',drop_period=False,splits=200):
    all_sentence_set=[]
    # create a column that is the sentence_number and goes from 0 to len(stimuli_pd.groupby('sent_id'))
    new_col=np.zeros((len(stimuli_pd),))
    # make new_col have the same index as stimuli_pd
    new_col=pd.Series(new_col,index=stimuli_pd.index)
    for idx, group in tqdm(enumerate(stimuli_pd.groupby('sent_id'))):
        new_col[group[1].index]=int(idx)
    # combine new_col and stimuli_pd
    stimuli_pd['sentence_number']=new_col
    seq = np.floor(np.linspace(0, stimuli_pd.sentence_number.max()+1, num=splits))
    seq_pair=np.vstack((seq[0:-1], seq[1:]))
    seq_pair=seq_pair.astype(int).transpose()
    num_row=seq_pair.shape[0]
    for row in tqdm(range(num_row)):
        sentence_words, word_nums, sentenceID = [], [], []
        stimuli_chunck=stimuli_pd[np.logical_and(np.asarray(stimuli_pd.sentence_number>=seq_pair[row,0]),
                                  np.asarray(stimuli_pd.sentence_number< seq_pair[row,1]))]

        sentenceID=list(stimuli_chunck.sent_id)
        if drop_period:
            # group word form by the sentence number and look at the last word and remove the period
            sentence_words=[]
            for idx, group in stimuli_chunck.groupby('sentence_number'):
                if group.word_form.iloc[-1]=='.':
                    group.word_form.iloc[-1]=group.word_form.iloc[-1].rstrip('.')
                sentence_words.append(group.word_form)
            sentence_words = list(pd.concat(sentence_words))
        else:
            sentence_words = list(stimuli_chunck.word_form)
        word_number=list(range(len(stimuli_chunck)))
        sentenceNum=list(stimuli_chunck.sentence_number)
        zipped_lst = list(zip(sentenceID, word_number, sentence_words,sentenceNum))
        sentence_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word','sentence_number'])
        sentence_set.name = f'{stimuli_data_name}_from_text_period_{drop_period}_group_{row}'
        all_sentence_set.append(sentence_set)
    return all_sentence_set


BENCHMARK_CONFIG=dict(file_loc=BENCHMARK_DIR)

SENTENCE_CONFIG = [
    dict(name='ud_sentences', file_loc=os.path.join(UD_PARENT,'ud_sentence_data.pkl')),
    dict(name='ud_sentences_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter.pkl')),
    dict(name='ud_sentences_filter_v2',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter_v2.pkl')),
    dict(name='ud_sentences_filter_v3',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_filter_v3_no_dup.pkl')),
    dict(name='ud_sentences_filter_v3_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter_sample_v3_no_dup.pkl')),
    dict(name='ud_sentences_filter_sample', file_loc=os.path.join(UD_PARENT, 'ud_sentence_data_filter_sample.pkl')),
    dict(name='ud_sentences_token_filter_v3',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter_v3_no_dup.pkl')),
    dict(name='ud_sentencez_token_filter_v3',file_loc=os.path.join(UD_PARENT,'ud_sentencez_data_token_filter_v3_no_dup.pkl')),
    dict(name='ud_sentencez_token_filter_v3_minus_ev_sentences',file_loc=os.path.join(UD_PARENT,'ud_sentencez_data_token_filter_v3_no_dup_minus_ev_sentences.pkl')),
    dict(name='ud_sentencez_token_filter_v3_minus_ev_sentences_len_7_14',file_loc=os.path.join(UD_PARENT,'ud_sentencez_data_token_filter_v3_no_dup_minus_ev_sentences_len_7_14.pkl')),
    dict(name='ud_sentencez_ds_random_100_edited',file_loc=os.path.join(UD_PARENT,'ud_sentencez_ds_random_100_edited.pkl')),
    dict(name='ud_sentencez_ds_max_100_edited',file_loc=os.path.join(UD_PARENT,'ud_sentencez_ds_max_100_edited.pkl')),
    dict(name='ud_sentencez_ds_min_100_edited',file_loc=os.path.join(UD_PARENT,'ud_sentencez_ds_min_100_edited.pkl')),
    dict(name='ud_sentencez_ds_random_100_edited_selected',file_loc=os.path.join(UD_PARENT, 'ud_sentencez_ds_random_100_edited_selected.pkl')),
    dict(name='ud_sentencez_ds_max_100_edited_selected',file_loc=os.path.join(UD_PARENT, 'ud_sentencez_ds_max_100_edited_selected.pkl')),
    dict(name='ud_sentencez_ds_min_100_edited_selected',file_loc=os.path.join(UD_PARENT, 'ud_sentencez_ds_min_100_edited_selected.pkl')),
    dict(name='ud_sentencez_token_filter_v3_sample',file_loc=os.path.join(UD_PARENT,'ud_sentencez_data_token_filter_sample_v3_no_dup.pkl')),
    dict(name='ud_sentences_token_filter',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter.pkl')),
    dict(name='ud_sentences_token_filter_sample',file_loc=os.path.join(UD_PARENT,'ud_sentence_data_token_filter_sample.pkl')),
    dict(name='coca_spok_filter_punct_sample',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_sample.pkl')),
    dict(name='coca_spok_filter_punct_10K_sample_1',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_10K_sample_1.pkl')),
    dict(name='coca_spok_filter_punct_10K_sample_2',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_10K_sample_2.pkl')),
    dict(name='coca_spok_filter_punct_10K_sample_3',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_10K_sample_3.pkl')),
    dict(name='coca_spok_filter_punct_10K_sample_4',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_10K_sample_4.pkl')),
    dict(name='coca_spok_filter_punct_10K_sample_5',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_10K_sample_5.pkl')),
    dict(name='coca_spok_filter_punct_50K',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_50K.pkl')), # 50000 sentences
    dict(name='coca_spok_filter_punct_50K_sylb',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_50K_sylb.pkl')), # 19852 sentences
    dict(name='coca_spok_filter_punct_50K_sylb_2to4sec',file_loc=os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_50K_sylb_2to4sec.pkl')), # 11651 sentences
    dict(name='coca_spok_filter_punct_10K_sample_ev_editsSep12',file_loc='/om/user/ehoseini/MyData/sent_sampling/results//coca_spok_filter_punct_10K_sample_ev_editsSep12.pkl'),
    dict(name='coca_spok_filter_punct_10K_sample_ev_editsOct16',file_loc='/om/user/ehoseini/MyData/sent_sampling/results//coca_spok_filter_punct_10K_sample_ev_editsOct16.pkl'),
    dict(name='ud_sentences_U01_SET1_paraphrase',file_loc=os.path.join(RESULTS_DIR,'ud_sentences_U01_SET1_paraphrase.pkl')),
    dict(name='ud_sentences_U01_AnnSET1_ordered_for_RDM',
         file_loc=os.path.join(RESULTS_DIR, 'ud_sentences_U01_AnnSET1_ordered_for_RDM.pkl')),
    dict(name='coca_preprocessed_all',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all.pkl')),
    dict(name='coca_preprocessed_all_clean',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_1',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_1.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_2',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_2.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_3',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_3.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_4',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_4.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_5',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_5.pkl')),
    dict(name='coca_preprocessed_all_clean_no_dup_100K_sample_1',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_no_dup_100K_sample_1.pkl')),
    dict(name='coca_preprocessed_all_clean_no_dup_100K_sample_1_split_0',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_no_dup_100K_sample_1_split_0.pkl')),
    dict(name='coca_preprocessed_all_clean_no_dup_100K_sample_2',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_no_dup_100K_sample_2.pkl')),
    dict(name='coca_preprocessed_all_clean_no_dup_100K_sample_3',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_no_dup_100K_sample_3.pkl')),
    dict(name='coca_preprocessed_all_clean_no_dup_100K_sample_4',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_no_dup_100K_sample_4.pkl')),
    dict(name='coca_preprocessed_all_clean_no_dup_100K_sample_5',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_no_dup_100K_sample_5.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_1_estim_ds_min',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_1_estim_ds_min.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10000.pkl')),
    dict(name='coca_preprocessed_all_clean_100K_sample_1_2_ds_max_est_n_10K',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_1_2_ds_max_est_n_10000.pkl')),
    dict(name='neural_ctrl_stim',file_loc=os.path.join(AUTO_ENCODER_DIR,'beta-control-neural_stimset_D-S_light_freq_extract.pkl')),
    dict(name='ds_parametric',file_loc=os.path.join(DSPARAMETRIC_DIR,'ds_parametric_extract.pkl')),
    dict(name='pereira2018_243sentences',file_loc=os.path.join(COCA_PREPROCESSED_DIR,'')),
    dict(name='pereira2018_384sentences',file_loc=os.path.join(COCA_PREPROCESSED_DIR,''))]

