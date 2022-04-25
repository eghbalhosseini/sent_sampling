import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj, COCA_CORPUS_DIR
import torchaudio.transforms as T
import torchaudio.functional as F
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import torchaudio
import librosa
from librosa.onset import onset_detect
import numpy as np
import re
import matplotlib as mpl

if __name__ == '__main__':
    save_loc = '/om/user/ehoseini/MyData/sent_sampling/results/'
    ann_sentneces=open(os.path.join(save_loc, 'sentence_AnnSet1_ordered_for_RDM_analysis.txt'), 'r')
    content=ann_sentneces.read()
    content_list=content.split('\n')
    content_list=content_list[:-1]
    assert(len(content_list)==200)
    # load an extractor
    extrac_id='group=best_performing_pereira-dataset=ud_sentences_token_filter_v3-activation-bench=None-ave=False'
    ext_obj = extract_pool[extrac_id]()
    ext_obj.load_dataset()
    input_data=ext_obj.data_
    input_sentences=[x['text'] for x in input_data]
    ann_loc=[]
    ann_data=[]
    for ann_sent in content_list:
        overlap_idx=input_sentences.index(ann_sent)
        ann_loc.append(overlap_idx)
        ann_data.append(input_data[overlap_idx])

    save_obj(ann_data,os.path.join(RESULTS_DIR, 'ud_sentences_U01_AnnSET1_ordered_for_RDM.pkl'))
