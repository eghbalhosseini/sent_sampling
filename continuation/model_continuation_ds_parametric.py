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
    optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
                'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']
    #
    # read the excel that contains the selected sentences
    # %%  RUN SANITY CHECKS
    ds_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN/ds_parametric/ANNSET_DS_MIN_MAX_from_100ev_eh_FINAL.csv')
    # read also the actuall experiment stimuli
    stim_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN//ds_parametric/fMRI_final/stimuli_order_ds_parametric.csv',
                           delimiter='\t')
    # find unique conditions
    unique_cond = np.unique(stim_csv.Condition)
    # for each unique_cond find sentence transcript
    unique_cond_transcript = [stim_csv.Stim_transcript[stim_csv.Condition == x].values for x in unique_cond]
    # remove duplicate sentences in unique_cond_transcript
    unique_cond_transcript = [list(np.unique(x)) for x in unique_cond_transcript]
    ds_min_list = unique_cond_transcript[1]
    ds_max_list = unique_cond_transcript[0]
    ds_rand_list = unique_cond_transcript[2]
    # extract the ds_min sentence that are in min_included column
    ds_min_ = ds_csv.DS_MIN_edited[(ds_csv['min_include'] == 1)]
    ds_max_ = ds_csv.DS_MAX_edited[(ds_csv['max_include'] == 1)]
    ds_rand_ = ds_csv.DS_RAND_edited[(ds_csv['rand_include'] == 1)]
    # check if ds_min_ and ds_min_list have the same set of sentences regardless of the order
    assert len([ds_min_list.index(x) for x in ds_min_]) == len(ds_min_)
    assert len([ds_max_list.index(x) for x in ds_max_]) == len(ds_max_)
    assert len([ds_rand_list.index(x) for x in ds_rand_]) == len(ds_rand_)
    # %% MORE SANITY CHECKS FOR THE ACTIVATIONS
    # get the
    ds_min_sent = ds_csv.DS_MIN[(ds_csv['min_include'] == 1)]
    ds_max_sent = ds_csv.DS_MAX[(ds_csv['max_include'] == 1)]
    ds_rand_sent = ds_csv.DS_RAND[(ds_csv['rand_include'] == 1)]
    # laod the extractor
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    #ext_obj()
    model_names = ext_obj.model_spec
    # %% create a model for causalLM
    ds_sentences=[ds_min_sent,ds_rand_sent,ds_max_sent]
    ds_sentence_cond=['ds_min','ds_rand','ds_max']
    model_ids=[0,1,2,3,4,6]
    for model_id in model_ids:
        model_name=model_names[model_id]
        for id_ds,ds_sent in enumerate(ds_sentences):
            file_path = Path('/om2/user/ehoseini/MyData/sent_sampling/analysis/',
                             f'{model_name}_{ds_sentence_cond[id_ds]}_sentence_generation_dict.pkl')
            if file_path.exists():
                print(f'{file_path.__str__()} exists')
                continue
            # load the model
            print(f'creating {file_path.__str__()}')
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            causal_model = AutoModelForCausalLM.from_config(config)
            causal_model.to(device)
            text_generation_config = GenerationConfig(
                num_beams=5,
                eos_token_id=causal_model.config.eos_token_id,
                pad_token_id=causal_model.config.pad_token_id,
                output_scores=True,
                do_sample=False,
                return_dict_in_generate=True
            )
            sentence_tokens=[]
            sentence_text=[]
            for sent in tqdm(ds_sent,total=len(ds_sent)):
                # split sent by words
                sent_words=sent.split()
                sentence_token=[]
                for kk in range(1,len(sent_words)):
                    sent_tok_id = tokenizer(' '.join(sent_words[:kk]), return_tensors='pt')
                    for key in sent_tok_id.keys():
                        sent_tok_id[key] = sent_tok_id[key].to(device)
                    sentence_token.append(sent_tok_id)
                sentence_tokens.append(sentence_token)
                sentence_text.append(sent)
            all_greedy_continuation=[]
            all_sentence_continuation=[]
            for id_sent,sentence_token in tqdm(enumerate(sentence_tokens),total=len(sentence_tokens)):
                sentence_continuation=[]
                greedy_last=[]
                sentence_prob=[]
                for idx,sent_token in enumerate(sentence_token):
                    # send sent_token to device
                    with torch.no_grad():
                        outputs = causal_model.generate(**sent_token,max_new_tokens=len(sentence_token)-idx, generation_config=text_generation_config,pad_token_id=causal_model.config.eos_token_id)
                    #selected_beam=outputs['beam_indices'][0]
                    # decode the output
                    greedy_continuation = tokenizer.decode(outputs['sequences'][0])
                    greedy_token=tokenizer.decode(outputs['sequences'][0][idx+1])
                    # convert scores to probabilities using softmax
                    #scores_prob = torch.nn.functional.softmax(outputs['scores'][0], dim=-1)
                    # get the probability of the next word
                    sentence_continuation.append(greedy_continuation)
                    greedy_last.append(greedy_token)
                    #sentence_prob.append(scores_prob[-1, :].tolist())
                all_greedy_continuation.append(greedy_last)
                all_sentence_continuation.append(sentence_continuation)
            # save a dictionary of all_sentence_continuation and original sentence text
            sentence_generation_dict={'model':model_name,
                                        'ds_condition':ds_sentence_cond[id_ds],
                                        'sentence_text':sentence_text,
                                        'sentence_continuation':all_sentence_continuation,
                                        'greedy_last':all_greedy_continuation}
            file_path=Path('/om2/user/ehoseini/MyData/sent_sampling/analysis/',f'{model_name}_{ds_sentence_cond[id_ds]}_sentence_generation_dict.pkl')
            # if parent doesnt exist create it
            if not file_path.parent.exists():
                os.makedirs(file_path.parent)
            # save the dictionary
            with open(file_path.__str__(), 'wb') as f:
                pickle.dump(sentence_generation_dict, f)
