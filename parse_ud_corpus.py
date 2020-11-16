import os
from tqdm import tqdm
import pyconll
import pandas as pd
import xlrd
import pickle
import numpy as np
import copy
from utils.data_utils import UD_PARENT, LEX_PATH_SET, UD_ENGLISH_PATH_SET, UD_PATH, uppercount
# file path constructor



# extract lexical features
##TODO: add word form to the set 

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
Set_ind = {'word_LEMMA': all_lemma_ind, 'word_string': all_word_ind}
## extract lexical features

for file in LEX_PATH_SET:
    data_ = file['read_instruction'](os.path.join(UD_PARENT, file['tag']))
    print('loaded ', file['name'])
    # add the values to the new set of the set
    A = data_[file['word_column']]
    B = data_[file['metric_column']]
    AB = dict((x, B[i]) for i, x in enumerate(A))
    word_metric_pair = dict((k, AB[k] if k.lower() in AB else float('nan')) for k in Unique_set[file['word_form']])

    for idx, sent in tqdm(enumerate(sentence_data)):
        tags_id = [Set_ind[file['word_form']][idx]][0]
        tags_key = [Unique_set[file['word_form']][x] for x in tags_id]
        sent[file['name']] = [word_metric_pair[k] for k in tags_key]
        sentence_data[idx] = sent

#
with open(os.path.join(UD_PARENT, 'ud_sentence_data.pkl'), 'wb') as fout:
    pickle.dump(sentence_data, fout)

# FILTERING OF THE DATASET
UPOS_CONFIG={}

# based on ASCII characters : see http://www.addressmunger.com/special_ascii_characters/
valid_chr_ASCII=[list(range(39,40)),
                 list(range(44,47)),
                 list(range(65,91)),
                 list(range(97,122)),
                 list(range(160,161))]
flat_list = [item for sublist in valid_chr_ASCII for item in sublist]
valid_chars=set([ chr(x) for x in flat_list])
sentence_data_ASCII=[]
for i, sentence in tqdm(enumerate(sentence_data)):
    sentence_chars = set([x for x in ''.join(sentence['word_FORM'])])
    if not bool(sentence_chars-valid_chars):
        sentence_data_ASCII.append(sentence)

sentence_data_filter=copy.deepcopy(sentence_data_ASCII)

WORD_FEAT_CONFIG={'Typo':{'Yes'},
                  'Abbr':{'Yes'},
                  'Foreign':{'Yes'},
                  'NumForm':{'Word','Digit','Roman'}}
sentence_data_word_FEAT=[]
all_include=[]
for i , sentence in tqdm(enumerate(sentence_data_filter)):
    include_criteria = []
    for key, val in WORD_FEAT_CONFIG.items():
        val_for_key=set([item for sublist in [list(x.get(key, {False})) for x in sentence['word_FEATS']] for item in sublist])
        if not bool(val-val_for_key):
            include_criteria.append(True)
        else:
            include_criteria.append(False)
    all_include.append(include_criteria)
    if not True in include_criteria:
        sentence_data_word_FEAT.append(sentence)
# finally cut the sentences that are too short or too long :

sentence_data_filter=copy.deepcopy(sentence_data_word_FEAT)

correct_lengh = [idx for idx, x in enumerate(sentence_data_filter) if x['sentence_length'] > 5 and x['sentence_length'] < 20]
sentence_data_filter=[sentence_data_filter[x] for x in correct_lengh]

# additional filtering of strange characters
invalid_char="''"
valid_idx=[idx for idx, x in enumerate(sentence_data_filter) if not(invalid_char in x['word_FORM'])]
sentence_data_filter=[sentence_data_filter[x] for x in valid_idx]
# remove sentences with many uppercases
upper_case_valid=[idx for idx, sent in enumerate(sentence_data_filter) if np.asarray([np.asarray([y.isupper() for y in x]).sum() for x in sent['word_FORM']]).max()<4]
#
sentence_data_filter=[sentence_data_filter[x] for x in upper_case_valid]

# removing the period in the end of the sentence
# TODO: remove punctuation from inputs to the model, but check it per model basis.
sent_with_period=[idx for idx, x in enumerate(sentence_data_filter) if (x['word_FORM'][-1]=='.')]
sentence_data_filter=[sentence_data_filter[x] for x in sent_with_period]

with open(os.path.join(UD_PARENT, 'ud_sentence_data_filter_v2.pkl'), 'wb') as fout:
    pickle.dump(sentence_data_filter, fout)

# remove duplicates from the set
sentence_text=[x['text'] for x in sentence_data_filter]
unique_sentences=list(set(sentence_text))
indexes=[sentence_text.index(x) for x in unique_sentences]

sentence_data_filter_no_dup=[sentence_data_filter[x] for x in indexes]
with open(os.path.join(UD_PARENT, 'ud_sentence_data_filter_v3_no_dup.pkl'), 'wb') as fout:
    pickle.dump(sentence_data_filter_no_dup, fout)

test=pd.DataFrame(sentence_data_filter)
test.drop_duplicates(sublist=['text'])
sentence_data_filter_sample=[sentence_data_filter[x] for x in range(200)]

with open(os.path.join(UD_PARENT, 'ud_sentence_data_filter_sample.pkl'), 'wb') as fout:
    pickle.dump(sentence_data_filter_sample, fout)

# clean the last token
sentence_data_token=[]
assert(np.asarray([x['text'][-1]=='.' for x in sentence_data_filter]).sum()==len(sentence_data_filter))
data_keys=list(sentence_data_filter[1].keys())
data_keys=list(set(data_keys)-set(['text_key','meta','id','text']))
sentence_data_token=copy.deepcopy(sentence_data_filter)
for idx, sentence in tqdm(enumerate(sentence_data_token)):
    for key in data_keys:
        if key=='sentence_length':
            sentence[key] = sentence[key]-1
        else:
            sentence[key] = sentence[key][:-1]
    sentence_data_token[idx]=sentence

with open(os.path.join(UD_PARENT, 'ud_sentence_data_token_filter.pkl'), 'wb') as fout:
    pickle.dump(sentence_data_token, fout)

# select a random subset
s_random_idx=list(np.random.randint(0,len(sentence_data_token),200))
sentence_data_token_sample=[sentence_data_token[x] for x in s_random_idx]

with open(os.path.join(UD_PARENT, 'ud_sentence_data_token_filter_sample.pkl'), 'wb') as fout:
    pickle.dump(sentence_data_token_sample, fout)

# filtering based on Universal features : see https://universaldependencies.org/u/feat/index.html