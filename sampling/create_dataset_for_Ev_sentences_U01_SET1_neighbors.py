import numpy as np
from sent_sampling.utils import extract_pool
from sent_sampling.utils.extract_utils import model_extractor
from sent_sampling.utils.optim_utils import optim_pool
import argparse
from sent_sampling.utils.extract_utils import model_extractor, model_extractor_parallel
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj, load_obj
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import importlib
import sys
importlib.reload(sys.modules['utils.data_utils'])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG, UD_PARENT
import matplotlib
import ftfy


if __name__ == '__main__':

    file_name = 'U01_sentselection_Dec18-2020_updDec23_nearest_neighbors_gpt2-xl_dist_metric_correlation_paraphrase - U01_sentselection_Dec18-2020_up.csv'
    df_paraphrase = pd.read_csv(os.path.join(RESULTS_DIR, f"{file_name}"))
    sent_str_paraphrase = list(df_paraphrase['Paraphrase '])
    sent_str_paraphrase=[ftfy.fix_text(x) for x in sent_str_paraphrase]
    # drop '. '
    sent_txt_paraphrase=[x.split(' ') for x in sent_str_paraphrase ]
    for idx,x in tqdm(enumerate(sent_txt_paraphrase)):
        if '' in x:
            x=[value for value in x if value != '']
        sent_txt_paraphrase[idx]=x
    # drop . in the end
    for idx,x in tqdm(enumerate(sent_txt_paraphrase)):
        if x[-1].__contains__('.'): x[-1]=x[-1].replace('.','')
        sent_txt_paraphrase[idx]=x
    sentence_data=[]
    separator = ' '
    for idx, sent_word in tqdm(enumerate(sent_txt_paraphrase)):
        sentence = separator.join(sent_word)
        dat_ = {'text': sentence,
            'text_key': [],
            'sentence_length': len(x),
            'meta': [],
            'id': f" ",
            # see https://universaldependencies.org/format.html
            'word_id': [id for id, x in enumerate(sent_word)],  # FORM: Word form or punctuation symbol.
            'word_FORM': sent_word,  # FORM: Word form or punctuation symbol.
            'word_LEMMA': [],  # LEMMA: Lemma or stem of word form.
            'word_FEATS': [],
            # FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
            'word_XPOS': [],
            # XPOS: Language-specific part-of-speech tag; underscore if not available.
            'word_UPOS': [],  # UPOS: Universal part-of-speech tag.
            'word_string': []}
        sentence_data.append(dat_)

    [x['text'] for x in sentence_data]
    with open(os.path.join(RESULTS_DIR, f"ud_sentences_U01_SET1_paraphrase.pkl"),
              'wb') as fout:
        pickle.dump(sentence_data, fout)

