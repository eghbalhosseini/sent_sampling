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

from utils.extract_utils import model_extractor_parallel
if __name__ == '__main__':
    extract_ids=['group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_wordFORM-activation-bench=None-ave=False',
                'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_wordFORM-activation-bench=None-ave=True']
    for extract_id in extract_ids:
        ext_obj = extract_pool[extract_id]()
        ext_obj.load_dataset()

        for idx in range(len(ext_obj.model_spec)):
            model_layers_ids = tuple([(idx, x) for idx, x in enumerate(model_layers[ext_obj.model_spec[idx]])])
            for layer_spec in tqdm(model_layers_ids):
                model_activation_name = f"{ext_obj.dataset}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
                old_model_activation_name = f"{ext_obj.dataset}_{ext_obj.model_spec[idx]}_layer_{ext_obj.layer_spec[idx]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
                assert Path(SAVE_DIR,old_model_activation_name).exists()
                # save a copy of old model activation file with model activation name
                os.system(f"cp {Path(SAVE_DIR,old_model_activation_name)} {Path(SAVE_DIR,model_activation_name)}")
                assert Path(SAVE_DIR,model_activation_name).exists()





    new_model_sentence_set=[]
    for idx in range(len(ext_obj.model_spec)):
        model_layers_ids = tuple([(idx, x) for idx, x in enumerate(model_layers[ext_obj.model_spec[idx]])])
        for layer_spec in tqdm(model_layers_ids):
            model_activation_name = f"{ext_obj.dataset}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
            assert Path(SAVE_DIR,model_activation_name).exists(), f"model activation file {model_activation_name} does not exist"
            model_activation=load_obj(Path(SAVE_DIR,model_activation_name).__str__())
            # find model activations for ev_sentences_loc
            model_activation_sentences=[x[1] for x in model_activation]
            # find location of ev_sentences in model_activation_sentences
            ev_activation_sentences_loc = [model_activation_sentences.index(x) for x in ev_sentences]
            # drop ev_activation_sentences_loc from model_activation
            model_activation_mod = [x for i, x in enumerate(model_activation) if i not in ev_activation_sentences_loc]
            # save model_activation_mod with new_dataset_name
            model_activation_name_mod = f"{new_dataset_name}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
            save_obj(model_activation_mod, Path(SAVE_DIR,model_activation_name_mod).__str__())
            # delete old model_activation_name
            old_model_activation_name=f"{old_dataset_name}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{layer_spec[0]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
            if Path(SAVE_DIR,old_model_activation_name).exists():
                os.remove(Path(SAVE_DIR,old_model_activation_name).__str__())
            new_model_sentence_set.append([x[1] for x in model_activation_mod])



    # check whether the setnences in are the same for all the lists in new_model_sentence_set
    assert all([new_model_sentence_set[0]==x for x in new_model_sentence_set]), "the sentences in the new_model_sentence_set are not the same"
    # check whether the sentences in new_model_sentence_set are the same as the sentences in data_mod
    assert [x['text'] for x in data_mod]==new_model_sentence_set[0], "the sentences in the new_model_sentence_set are not the same as the sentences in data_mod"




