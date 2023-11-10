import glob
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG, RESULTS_DIR, UD_PARENT, SAVE_DIR,load_obj,save_obj, ANALYZE_DIR
from utils import extract_pool
import pickle
from neural_nlp.models import model_pool, model_layers
import fnmatch
import re
from utils.extract_utils import model_extractor_parallel
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from TTS.api import TTS

if __name__ == '__main__':
    extract_ids = [
        f'group=best_performing_pereira_1-dataset=ud_sentencez_ds_max_100_edited_selected_textNoPeriod-activation-bench=None-ave=False',
        f'group=best_performing_pereira_1-dataset=ud_sentencez_ds_min_100_edited_selected_textNoPeriod-activation-bench=None-ave=False',
        f'group=best_performing_pereira_1-dataset=ud_sentencez_ds_random_100_edited_selected_textNoPeriod-activation-bench=None-ave=False']
    model_names = TTS.list_models()
    #model_name='tts_models/en/ljspeech/tacotron2-DDC'
    #tts = TTS(model_name)
    #model_nam_=model_name.replace('/', '_')
    # selected_models=[
    #     'tts_models/en/ek1/tacotron2',
    #     'tts_models/en/ljspeech/tacotron2-DDC',
    #     'tts_models/en/ljspeech/glow-tts',
    #     'tts_models/en/ljspeech/speedy-speech',
    #     'tts_models/en/ljspeech/tacotron2-DCA',
    #     'tts_models/en/ljspeech/vits',
    #     'tts_models/en/ljspeech/vits--neon',
    #     'tts_models/en/ljspeech/fast_pitch',
    #     'tts_models/en/ljspeech/overflow',
    #     'tts_models/en/ljspeech/neural_hmm',
    #     'tts_models/en/vctk/vits',
    #     'tts_models/en/vctk/fast_pitch',
    #     'tts_models/en/sam/tacotron-DDC',
    #     'tts_models/en/blizzard2013/capacitron-t2-c50',
    #     'tts_models/en/blizzard2013/capacitron-t2-c150_v2',
    # ]
    # for model_name in selected_models:
    #     print(model_name)
    tts = TTS(model_name,gpu=True)
    model_nam_ = model_name.replace('/', '_')
    for extract_id in extract_ids:
        ext_obj=extract_pool[extract_id]()
        ext_obj.load_dataset()
        ext_obj()
        sentences=[x[1] for x in ext_obj.model_group_act[0]['activations']]
        # fix .... to , with space in sentences
        sentences = [re.sub(r'\.\.\.\.', ',', x) for x in sentences]
        # replace period with , in sentences
        sentences = [re.sub(r'\.', ',', x) for x in sentences]
        # add a period in the end of the sentences
        sentences = [x + '.' for x in sentences]
        # save sentences into a text file in analyze dir with the name extract_id
        with open(os.path.join(ANALYZE_DIR, 'sentences_'+extract_id + '.txt'), 'w') as f:
            for item in sentences:
                f.write("%s\n" % item)





#         for idx, sentence in enumerate(sentences):
#             path_id = Path(ANALYZE_DIR,f'audio_{extract_id}', f'{model_nam_}_{idx}.wav')
#             # make sure the parent folder exist
#             path_id.parent.mkdir(parents=True, exist_ok=True)
#             speaker= tts.speakers[0] if tts.speakers else None
#             language = tts.languages[0] if tts.languages else None
#             tts.tts_to_file(text=sentence, file_path=path_id.__str__(),speaker=speaker,language=language)
# #%%



