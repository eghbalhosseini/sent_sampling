from utils import extract_pool
from utils.extract_utils import model_extractor
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj
import os

#
if __name__ == '__main__':
    Ev_sentence_files='/user/ehoseini/MyData/sent_sampling/results/sentence_group%3Dgpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_ev_editSept12.xlsx'