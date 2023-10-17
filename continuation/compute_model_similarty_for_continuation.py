import os
import numpy as np
import sys
from pathlib import Path
import getpass
if getpass.getuser() == 'eghbalhosseini':
    SAMPLING_PARENT = '/Users/eghbalhosseini/MyCodes/sent_sampling'
    SAMPLING_DATA = '/Users/eghbalhosseini/MyCodes//fmri_DNN/ds_parametric/'

elif getpass.getuser() == 'ehoseini':
    SAMPLING_PARENT = '/om/user/ehoseini/sent_sampling'
    SAMPLING_DATA = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'
import pickle
sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from utils import extract_pool
from utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
from matplotlib.pyplot import GridSpec
import pandas as pd
import torch

from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
# determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    # ext_obj()
    model_names = ext_obj.model_spec
    ds_sentence_cond = ['ds_min', 'ds_rand', 'ds_max']
    model_ids = [0, 1, 2, 3, 4, 6]
    model_name=model_names[model_ids[5]]
    id_ds=1

    file_path = Path('/om2/user/ehoseini/MyData/sent_sampling/analysis/',
                     f'{model_name}_{ds_sentence_cond[id_ds]}_sentence_generation_dict.pkl')
    with open(file_path, 'rb') as f:
        sentence_generation_dict = pickle.load(f)
    len(sentence_generation_dict['greedy_last'])
    sentence_generation_dict['greedy_last'][0]
    sentence_generation_dict['sentence_text'][0]
    sentence_generation_dict['sentence_continuation'][0]