import numpy as np
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
from glob import glob
import xarray as xr



if __name__ == '__main__':
    data_folder='/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored/'
    #data_folder = '/om/user/ehoseini/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored/'
    model='distilgpt2'
    stimuli='Pereira2018-243sentences'
    name_pattern=f'identifier={model},stimuli_identifier={stimuli}*'
    file_pattern=f'{data_folder}/{name_pattern}'
    model_files=glob(file_pattern)

    all_data_from_nlp_lst=[]
    for file_id in tqdm(model_files):
        data_from_nlp = pd.read_pickle(file_id)
        data_from_nlp = data_from_nlp['data']
        all_data_from_nlp_lst.append(data_from_nlp)

    all_data_from_nlp = xr.concat(all_data_from_nlp_lst, dim='presentation')

    # reformat to layers
    all_data_from_nlp_layer = []
    for _, grp in all_data_from_nlp.groupby('layer'):
        all_data_from_nlp_layer.append(grp)

    all_data_from_nlp_dict = dict()
    # reformat it so its readable for bplm
    for idx, layer_dat in tqdm(enumerate(all_data_from_nlp_layer)):
        stimulus_id = [id_sent_dict[val][0] for val in layer_dat.stimulus_sentence.values]
        sentence_num = [id_sent_dict[val][1] for val in layer_dat.stimulus_sentence.values]
        layer_dat = layer_dat.assign_coords({'sentence_id': ('presentation', sentence_num)})
        layer_dat = layer_dat.assign_coords({'stimulus_id': ('presentation', stimulus_id)})
        layer_dat = layer_dat.assign_coords({'neuro_id': ('neuroid', layer_dat.coords['neuroid_num'].values)})
        layer_dat = layer_dat.swap_dims({'presentation': 'sentence_id'})
        layer_dat = layer_dat.swap_dims({'neuroid': 'neuro_id'})
        layer_dat = layer_dat.sortby('sentence_id')
        all_data_from_nlp_dict[idx] = layer_dat


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
    sentence_grp = 'coca_spok_filter_punct_10K_sample_ev_editsOct16.pkl'
    # save the data
    ev_data_file = f'{RESULTS_DIR}/{sentence_grp}'
    with open(ev_data_file, 'wb') as fout:
        pickle.dump(data_ev, fout)