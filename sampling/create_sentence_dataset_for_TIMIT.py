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

from sent_sampling.utils.data_utils import SENTENCE_CONFIG
import matplotlib
sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
timit_path='/nese/mit/group/evlab/u/ehoseini/MyData/TIMIT/'
from pathlib import Path
def check_no_space_words(sentences):
    for sentence in sentences:
        for word in sentence:
            if word.strip() == "":
                return False  # Found a word that is just spaces
    return True  # No words with only spaces found


if __name__ == '__main__':
    # read the text file from timit directory, with the file name timit.txt
    text_file=Path(timit_path,'timit.txt')
    with open(text_file) as f:
        lines = f.readlines()
    # remove the \n from the end of each line by finding it
    caption_selected = [x[:-1] if x[-1]=='\n' else x for x in lines]




    caption_selected = [x.lower() for x in caption_selected]
    # drop the additiona spaces in the end of each caption_selected word if it exists
    caption_selected = [x[:-1] if x[-1]==' ' else x for x in caption_selected]
    # drop the additional spaces in the beginning of each caption_selected word if it exists
    caption_selected = [x[1:] if x[0]==' ' else x for x in caption_selected]
    # drop any double spaces in the caption_selected
    caption_selected = [x.replace('  ', ' ') for x in caption_selected]
    # drop any double of more spaces in the caption_selected
    caption_selected = [re.sub(' +', ' ', x) for x in caption_selected]

    # drop period in the end of each caption_selected word if it exists
    caption_selected = [x[:-1] if x[-1]=='.' else x for x in caption_selected]
    # make sure the first character of each caption_selected is not a space
    caption_selected = [x[1:] if x[0]==' ' else x for x in caption_selected]
    # make sure the last character of each caption_selected is not a space
    caption_selected = [x[:-1] if x[-1]==' ' else x for x in caption_selected]
    # drop the period in the end of each caption_selected word if it exists
    #caption_selected = [x[:-1] if x[-1]=='.' else x for x in caption_selected]
    words_list = [x.split(' ') for x in caption_selected]
    # assert there is no empty element in words
    assert np.sum([len(x)==0 for x in words_list])==0
    # make sure no words in in words is just a space or a combination of spaces
    result = check_no_space_words(words_list)
    assert(result)  # This will print False because

    word_form = words_list
    word_id = [list(range(len(x))) for x in words_list]
    # create a counter for sentence id based on the words in each sentence
    sent_id = [list(idx * np.ones(len(x)).astype(int)) for idx, x in enumerate(words_list)]

    # flatten the list
    words = [item for sublist in words_list for item in sublist]
    # assert that no word is empty
    assert np.sum([x=='' for x in words])==0
    word_form = [item for sublist in word_form for item in sublist]
    word_id = [item for sublist in word_id for item in sublist]
    sent_id = [item for sublist in sent_id for item in sublist]

    df_extract=pd.DataFrame({'word':words,'word_id':word_id,'sent_id':sent_id,'word_form':word_form})
    p=Path(timit_path,'TIMIT_clean_v1.pkl')
    df_extract.to_pickle(p.__str__())
