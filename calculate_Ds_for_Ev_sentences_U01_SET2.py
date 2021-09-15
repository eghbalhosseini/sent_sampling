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
import pickle

#
if __name__ == '__main__':

    file_name = 'sentence_group=gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_ev_editSept12.csv'
    df_ev_selected = pd.read_csv(os.path.join(RESULTS_DIR, f"{file_name}"))
    sent_str_filtered_from_ev=list(df_ev_selected[df_ev_selected['use?']==1]['sentence'])
    set_num_filtered_from_ev = list(df_ev_selected[df_ev_selected['use?'] == 1]['set_num'])
    # find sentences in COCA corpus :
    uniq_set_num=np.unique(set_num_filtered_from_ev)
    sentence_sets=[f'coca_spok_filter_punct_10K_sample_{x}' for x in uniq_set_num]
    loc_in_sent_config=[[x['name'] for x in SENTENCE_CONFIG].index(y) for y in sentence_sets]
    file_loc=[SENTENCE_CONFIG[x]['file_loc'] for x in loc_in_sent_config]
    data_=[]
    for file in file_loc:
        data_.append(load_obj(file))
    len(data_)
    data_=list(np.ravel(data_))
    all_sentences=[x['text'] for x in data_]
    corres=[all_sentences.index(x) for x in sent_str_filtered_from_ev]
    data_ev=[data_[x] for x in corres]
    # construct an extractor object and get the sentences representations :
    sentence_grp='coca_spok_filter_punct_10K_sample_ev_editsSep12'
    # save the data
    ev_data_file=f'{RESULTS_DIR}/{sentence_grp}'
    with open(ev_data_file,'wb') as fout:
        pickle.dump(data_ev, fout)


    extract_name = f'gpt2-xl_ctrl_bert_gpt2_openaigpt_lm_1b_layers-dataset={sentence_grp}'
    extract_id = [f'group=gpt2-xl_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=ctrl_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=bert-large-uncased-whole-word-masking_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=gpt2_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                  f'group=openaigpt_layers-dataset={sentence_grp}-activation-bench=None-ave=False',
                 f'group=lm_1b_layers-dataset={sentence_grp}-activation-bench=None-ave=False']
    group_ids= list(np.arange(9))
    extractor_obj = model_extractor(dataset=sentence_grp, datafile=ev_data_file, model_spec='lm_1b')
    extractor_obj.load_dataset()
    pd.read_pickle(ev_data_file)['data']
    extractor_obj()

