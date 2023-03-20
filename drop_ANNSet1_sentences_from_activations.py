import glob
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG, RESULTS_DIR, UD_PARENT, SAVE_DIR,load_obj,save_obj
from utils import extract_pool
import pickle
from neural_nlp.models import model_pool, model_layers
import fnmatch
import re
from utils.extract_utils import model_extractor_parallel
import numpy as np
if __name__ == '__main__':
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=False'
    #extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_wordFORM-activation-bench=None-ave=False'
    extract_id ='group=best_performing_pereira_1-dataset=ud_sentencez_ds_random_100_edited_selected_textNoPeriod-activation-bench=None-ave=False'

    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()

    for k, layer in enumerate(tqdm(ext_obj.layer_spec, desc='layers')):
        True
        model_save_path = os.path.join(SAVE_DIR, ext_obj.model_spec[k])
        model_activation_name = f"{ext_obj.dataset}_{ext_obj.stim_type}_{ext_obj.model_spec[k]}_layer_{k}_{ext_obj.extract_name}_group_*.pkl"
        new_model_activation_name = f"{ext_obj.dataset}_{ext_obj.stim_type}_{ext_obj.model_spec[k]}_layer_{k}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
        if os.path.exists(os.path.join(SAVE_DIR, new_model_activation_name)):
            print(f'{os.path.join(SAVE_DIR, new_model_activation_name)} already exists\n')

        activation_files = []
        for file in os.listdir(model_save_path):
            if fnmatch.fnmatch(file, model_activation_name):
                activation_files.append(os.path.join(model_save_path, file))
        # sort files:
        sorted_files = []
        s = [re.findall('group_\d+', x) for x in activation_files]
        s = [item for sublist in s for item in sublist]
        file_id = [int(x.split('group_')[1]) for x in s]
        # find which numbers from 0 to 18  is missing from file_id
        missing = [x for x in range(0, 19) if x not in file_id]
        # print missing numbers
        print(f'model {ext_obj.model_spec[k]} missing: {missing}\n')
        #sorted_files = [activation_files[x] for x in np.argsort(file_id)]
    # first extract ev sentences
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))

    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]
    ev_sentences

    ext_obj()
    data_sentences=[x['text'] for x in ext_obj.data_]
    # find location of ev_sentences in data_sentences
    ev_sentences_loc = [data_sentences.index(x) for x in ev_sentences]
    ev_sentence_data= [ext_obj.data_[x] for x in ev_sentences_loc]
    # drop ev_sentences_loc from data_sentences
    data_mod = [x for i, x in enumerate(ext_obj.data_) if i not in ev_sentences_loc]

    file_name=Path(UD_PARENT,'ud_sentencez_data_token_filter_v3_no_dup_minus_ev_sentences.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(data_mod, f, protocol=4)

    # create a new dataset where sentence length is limited to be between 7 words and 14 words
    data_mod_7_14 = [x for x in data_mod if len(x['text'].split()) in range(7, 15)]
    file_name=Path(UD_PARENT,'ud_sentencez_data_token_filter_v3_no_dup_minus_ev_sentences_len_7_14.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(data_mod_7_14, f, protocol=4)

    new_dataset_name=ext_obj.dataset+'_minus_ev_sentences'
    old_dataset_name=ext_obj.dataset+'-minus_ev_sentences'
    new_model_sentence_set=[]
    new_model_sentence_set_7_14=[]



    for idx in range(len(ext_obj.model_spec)):
        model_layers_ids = tuple([(idx, x) for idx, x in enumerate(model_layers[ext_obj.model_spec[idx]])])
        for layer_spec in tqdm(model_layers_ids):
            model_activation_name = f"{ext_obj.dataset}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
            assert Path(SAVE_DIR,model_activation_name).exists(), f"model activation file {model_activation_name} does not exist"
            model_activation=load_obj(Path(SAVE_DIR,model_activation_name).__str__())
            # for sentences in model_activation if the ext_obj.identifier has wordFORM, find correponding text from ext_obj.data_
            if ext_obj.stim_type=='wordFORM':
                model_activation_sentences = [x[1] for x in model_activation]
                ev_activation_sentences_loc=[]
                for ev_sent in tqdm(ev_sentences):
                    sent_overlap = [len(set(ev_sent.split()).intersection(set(x.split()))) for x in model_activation_sentences]
                    if max(sent_overlap)>=len(ev_sent.split())-3:
                        ev_activation_sentences_loc.append(sent_overlap.index(max(sent_overlap)))
                    else:
                        ev_activation_sentences_loc.append(None)
            elif ext_obj.stim_type=='textNoPeriod':

            # find model activations for ev_sentences_loc
            # check whether the model activation is a list or a dict
                #go through ext_obj.stimulus_set and extract sentences
                # print model name and layer name


                # for each sentence in ev_sentences, find with strings in sent_strings has the most overlap
                ev_activation_sentences_loc=[]
                model_sentences=[x[1] for x in model_activation]
                for ev_sent in tqdm(ev_sentences):
                    # drop period in the end from ev_sent
                    ev_sent=ev_sent[:-1]
                    ev_activation_sentences_loc.append(model_sentences.index(ev_sent))


            # drop ev_activation_sentences_loc from model_activation
            model_activation_mod = [x for i, x in enumerate(model_activation) if i not in ev_activation_sentences_loc]
            # save model_activation_mod with new_dataset_name
            model_activation_name_mod = f"{new_dataset_name}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
            save_obj(model_activation_mod, Path(SAVE_DIR,model_activation_name_mod).__str__())
            # delete old model_activation_name
            #old_model_activation_name=f"{old_dataset_name}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
            #if Path(SAVE_DIR,old_model_activation_name).exists():
            #    os.remove(Path(SAVE_DIR,old_model_activation_name).__str__())
            new_model_sentence_set.append([x[1] for x in model_activation_mod])
            # create a second set which has sentences between 7 and 14 words
            model_activation_mod_7_14 = [x for x in model_activation_mod if len(x[1].split()) in range(7, 15)]
            model_activation_name_mod_7_14 = f"{new_dataset_name}_len_7_14_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
            save_obj(model_activation_mod_7_14, Path(SAVE_DIR,model_activation_name_mod_7_14).__str__())
            new_model_sentence_set_7_14.append([x[1] for x in model_activation_mod_7_14])





    # check whether the setnences in are the same for all the lists in new_model_sentence_set
    assert all([new_model_sentence_set[0]==x for x in new_model_sentence_set]), "the sentences in the new_model_sentence_set are not the same"

    assert all([new_model_sentence_set_7_14[0] == x for x in
                new_model_sentence_set_7_14]), "the sentences in the new_model_sentence_set are not the same"
    # check whether the sentences in new_model_sentence_set are the same as the sentences in data_mod
    assert [x['text'] for x in data_mod]==new_model_sentence_set[0], "the sentences in the new_model_sentence_set are not the same as the sentences in data_mod"




