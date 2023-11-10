import glob
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG, RESULTS_DIR, UD_PARENT, SAVE_DIR,load_obj,save_obj
from sent_sampling.utils import extract_pool
import pickle
from neural_nlp.models import model_pool, model_layers
import fnmatch
import re
from sent_sampling.utils.extract_utils import model_extractor_parallel
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from TTS.api import TTS

if __name__ == '__main__':
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))

    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]
    ev_sentences
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=False'
    ext_obj = extract_pool[extract_id]()
    ext_obj.model_spec
    for model_spec in ext_obj.model_spec:
        model = AutoModelForCausalLM.from_pretrained(model_spec)
        tokenizer = AutoTokenizer.from_pretrained(model_spec)
        sentence=list(ev_sentences)[0]
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model.generate(inputs, labels=inputs)
        model.generation_config


    #%%
    model_name=TTS.list_models()[6]
    tts=TTS(model_name)



