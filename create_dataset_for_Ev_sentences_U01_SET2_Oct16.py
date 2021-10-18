import numpy as np
from utils import extract_pool
from utils.extract_utils import model_extractor
from utils.optim_utils import optim_pool
import argparse
from utils.extract_utils import model_extractor, model_extractor_parallel
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import RESULTS_DIR, save_obj, load_obj
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
from utils.data_utils import SENTENCE_CONFIG
import matplotlib


if __name__ == '__main__':
    file_name = 'sentence_group=gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_ev_editOct16.csv'
    df_ev_selected = pd.read_csv(os.path.join(RESULTS_DIR, f"{file_name}"))
    sent_str_filtered_from_ev = list(df_ev_selected[df_ev_selected['use?'] == 1]['sentence'])
    set_num_filtered_from_ev = list(df_ev_selected[df_ev_selected['use?'] == 1]['set_num'])

    uniq_set_num = np.unique(set_num_filtered_from_ev)
    sentence_sets = [f'coca_spok_filter_punct_10K_sample_{x}' for x in uniq_set_num]
    loc_in_sent_config = [[x['name'] for x in SENTENCE_CONFIG].index(y) for y in sentence_sets]
    file_loc = [SENTENCE_CONFIG[x]['file_loc'] for x in loc_in_sent_config]
    data_ = []
    for file in file_loc:
        data_.append(load_obj(file))
    len(data_)
    data_ = list(np.ravel(data_))
    all_sentences = [x['text'] for x in data_]

    corres = [all_sentences.index(x) for x in sent_str_filtered_from_ev]
    data_ev = [data_[x] for x in corres]
    sentence_grp = 'coca_spok_filter_punct_10K_sample_ev_editsOct16'
    # save the data
    ev_data_file = f'{RESULTS_DIR}/{sentence_grp}'
    with open(ev_data_file, 'wb') as fout:
        pickle.dump(data_ev, fout)