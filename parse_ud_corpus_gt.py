import os
from tqdm import tqdm
import pyconll
import pandas as pd
import pickle
import numpy as np
import copy
from utils.data_utils import UD_PARENT, UD_ENGLISH_PATH_SET, UD_PATH

if __name__ == '__main__':

    file_set = []
    for ud_set in UD_ENGLISH_PATH_SET:
        for group in ud_set['group']:
            file_set.append(os.path.join(UD_PATH, ud_set['name'], ud_set['tag'] + '-' + group + '.conllu'))

    # collect all sentences
    sentence_data = []
    for file in file_set:
        ud_data = pyconll.load_from_file(file)
        for i, sentence in tqdm(enumerate(ud_data)):
            dat_ = {'text': sentence.text,
                    'text_key': sentence.TEXT_KEY,
                    'sentence_length': sentence.__len__(),
                    'meta': str(sentence._meta),
                    'id': sentence.id,
                    # see https://universaldependencies.org/format.html
                    'word_id': [x.id for x in sentence],  # FORM: Word form or punctuation symbol.
                    'word_FORM': [x.form for x in sentence],  # FORM: Word form or punctuation symbol.
                    'word_LEMMA': [x.lemma for x in sentence],  # LEMMA: Lemma or stem of word form.
                    'word_FEATS': [x.feats for x in sentence],
                    # FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
                    'word_XPOS': [x.xpos for x in sentence],
                    # XPOS: Language-specific part-of-speech tag; underscore if not available.
                    'word_UPOS': [x.upos for x in sentence],  # UPOS: Universal part-of-speech tag.
                    'word_string': [x.form for x in sentence]}
            sentence_data.append(dat_)

    all_lemma = []
    all_words = []
    for index, sent in tqdm(enumerate(sentence_data)):
        [all_lemma.append(x.lower()) for x in sent['word_LEMMA'] if type(x) == str]
        [all_words.append(x.lower()) for x in sent['word_string'] if type(x) == str]
    unique_lemma = list(set(all_lemma))
    unique_words = list(set(all_words))
    Unique_set = {'word_LEMMA': unique_lemma, 'word_string': unique_words}
    lemma_ind_dict = dict((k, i) for i, k in enumerate(unique_lemma))
    word_ind_dict = dict((k, i) for i, k in enumerate(unique_words))

    all_word_ind = []
    all_lemma_ind = []
    for index, sent in tqdm(enumerate(sentence_data)):
        word_ind = [word_ind_dict[x.lower()] for x in sent['word_string'] if type(x) == str]
        lemma_ind = [lemma_ind_dict[x.lower()] for x in sent['word_LEMMA'] if type(x) == str]
        all_word_ind.append(word_ind)
        all_lemma_ind.append(lemma_ind)
    set_ind = {'word_LEMMA': all_lemma_ind, 'word_string': all_word_ind}

    # FILTERING OF THE DATASET
    UPOS_CONFIG = {}

    # based on ASCII characters : see http://www.addressmunger.com/special_ascii_characters/ decimal code
    valid_chr_ASCII = [[33, 34, 39, 44, 45, 46, 58, 59, 63],  # punctuation
                       list(range(48, 58)),  # numbers
                       list(range(65, 91)),  # uppercase
                       list(range(97, 123)),  # lowercase
                       [160]]  # space
    flat_list = [item for sublist in valid_chr_ASCII for item in sublist]
    valid_chars = set([chr(x) for x in flat_list])

    sentence_data_ASCII = []
    for i, sentence in tqdm(enumerate(sentence_data)):
        sentence_chars = set([x for x in ''.join(sentence['word_FORM'])])
        if not bool(sentence_chars - valid_chars):
            sentence_data_ASCII.append(sentence)

    print(f'Filtered out {len(sentence_data) - len(sentence_data_ASCII)} sentences that did not match the ASCII requirements')
    sentence_data_filter = copy.deepcopy(sentence_data_ASCII)

    WORD_FEAT_CONFIG = {'Typo': {'Yes'},
                      'Abbr': {'Yes'},
                      'Foreign': {'Yes'},
                      'NumForm': {'Word','Digit','Roman'}}

    sentence_data_word_FEAT = []

    all_include = []
    for i, sentence in tqdm(enumerate(sentence_data_filter)):
        include_criteria = []
        for key, val in WORD_FEAT_CONFIG.items():
            val_for_key = set([item for sublist in [list(x.get(key, {False})) for x in sentence['word_FEATS']] for item in sublist])
            if not bool(val-val_for_key):
                include_criteria.append(True)
            else:
                include_criteria.append(False)
        all_include.append(include_criteria)
        if not True in include_criteria:
            sentence_data_word_FEAT.append(sentence)

    # finally cut the sentences that are too short or too long:
    sentence_data_filter_len = copy.deepcopy(sentence_data_word_FEAT)

    correct_length = [idx for idx, x in enumerate(sentence_data_filter_len) if x['sentence_length'] > 5 and x['sentence_length'] < 20]
    sentence_data_filter_len = [sentence_data_filter_len[x] for x in correct_length]
    print(f'Filtered out {len(sentence_data_filter_len) - len(correct_length)} sentences that did not meet length requirements')

    # additional filtering of strange characters
    invalid_char = "''"
    valid_idx = [idx for idx, x in enumerate(sentence_data_filter_len) if not(invalid_char in x['word_FORM'])]
    print(f'Filtered out {len(sentence_data_filter_len) - len(valid_idx)} sentences with additional invalid character')

    sentence_data_filter_len = [sentence_data_filter_len[x] for x in valid_idx]
    # remove sentences with many uppercases
    upper_case_valid = [idx for idx, sent in enumerate(sentence_data_filter_len) if np.asarray([np.asarray([y.isupper() for y in x]).sum() for x in sent['word_FORM']]).max() < 4]
    sentence_data_filter_len = [sentence_data_filter_len[x] for x in upper_case_valid]

    # keep sentences with punctuation in the end
    sentence_with_period = [idx for idx, x in enumerate(sentence_data_filter_len) if (x['word_FORM'][-1] == '.')]
    print(f'Filtered out {len(sentence_data_filter_len) - len(sentence_with_period)} sentences without period in the end')
    sentence_data_filter_len = [sentence_data_filter_len[x] for x in sentence_with_period]

    # remove duplicates from the set
    sentence_text = [x['text'] for x in sentence_data_filter_len]
    unique_sentences = list(set(sentence_text))
    duplicate_indices = [sentence_text.index(x) for x in unique_sentences]
    sentence_data_filter_no_dup = [sentence_data_filter[x] for x in duplicate_indices]
    print(f'Filtered out {len(sentence_data_filter_len) - len(sentence_data_filter_len)} duplicate sentences')

    # save as either a pd dataframe, or dictionary 
    text_pd = [x['text'] for x in sentence_data_filter_no_dup]

    stimuli_ids = []
    for num, sent in enumerate(text_pd):
        stimuli_ids.append(f'ud.{str(num).zfill(5)}')

    stimuli_ud = pd.DataFrame(text_pd, index=stimuli_ids, columns=['sentence'])
    stimuli_ud.to_pickle('UD_stimulusset.pkl')

    with open(os.path.join(UD_PARENT, 'UD_sentences.pkl'), 'wb') as fout:
        pickle.dump(sentence_data_filter_no_dup, fout)

    # sentence_data_filter_sample=[sentence_data_filter_no_dup[x] for x in range(200)]
    #
    # with open(os.path.join(UD_PARENT, 'ud_sentence_data_filter_sample_v3_no_dup.pkl'), 'wb') as fout:
    #     pickle.dump(sentence_data_filter_sample, fout)

    # select a random subset
    # s_random_idx=list(np.random.randint(0,len(sentence_data_token),200))
    # sentence_data_token_sample=[sentence_data_token[x] for x in s_random_idx]

    # with open(os.path.join(UD_PARENT, 'ud_sentence_data_token_filter_sample_v3_no_dup.pkl'), 'wb') as fout:
    #     pickle.dump(sentence_data_token_sample, fout)

    # filtering based on Universal features : see https://universaldependencies.org/u/feat/index.html