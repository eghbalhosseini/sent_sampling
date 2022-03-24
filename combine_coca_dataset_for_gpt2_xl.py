from utils import extract_pool
from utils.extract_utils import model_extractor
from utils.data_utils import RESULTS_DIR, save_obj,load_obj,SAVE_DIR,COCA_CORPUS_DIR
from utils.data_utils import SENTENCE_CONFIG
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

if __name__ == '__main__':
    #dataset_id='coca_spok_filter_punct_10K_sample_1'

    #extractor_id=f'group=gpt2-xl_layers-dataset={dataset_id}-activation-bench=None-ave=None'
    #extractor_obj = extract_pool[extractor_id]()
    #extractor_obj.load_dataset()
    #text_lines=[x['text'] for x in extractor_obj.data_]
    #text_lines[1]
    #extractor_obj(overwrite=False)
    #extractor_obj.model_group_act[0]['activations'][0]


    for l in tqdm(range(49)):
        t_all = []
        for sample_1 in range(5):
            dataset_id = f'coca_spok_filter_punct_10K_sample_{sample_1+1}'
            extractor_id = f'group=gpt2-xl_layers-dataset={dataset_id}-activation-bench=None-ave=None'
            extractor_obj = extract_pool[extractor_id]()
            model_actvation_name = f"{extractor_obj.dataset}_{extractor_obj.model_spec[l]}_layer_{extractor_obj.layer_spec[l]}_{extractor_obj.extract_name}_ave_{extractor_obj.average_sentence}.pkl"
            t=load_obj(os.path.join(SAVE_DIR,model_actvation_name))
            t_all.append(t)

        flat_list = [item for sublist in t_all for item in sublist]
        new_dataset_id = f'coca_spok_filter_punct_50K'
        new_model_actvation_name = f"{new_dataset_id}_{extractor_obj.model_spec[l]}_layer_{extractor_obj.layer_spec[l]}_{extractor_obj.extract_name}_ave_{extractor_obj.average_sentence}.pkl"
        save_obj(flat_list,os.path.join(SAVE_DIR, new_model_actvation_name))
    # create a folder that contains text and wav files
    txt_path=Path(SAVE_DIR,new_dataset_id,'text')
    txt_path.mkdir(parents=True, exist_ok=True)
    sentence_text = [x[1] for x in flat_list]
    txt_file = f'{txt_path.__str__()}/{new_dataset_id}_all.txt'
    with open(txt_file,'w') as f:
        for line_id,line in tqdm(enumerate(sentence_text)):
            id_z = str(line_id).zfill(5)
            f.write(f'{line}')
            f.write('\n')
    f.close()
    # for id, txt in tqdm(enumerate(sentence_text)):
    #     id_z=str(id).zfill(5)
    #     txt_file=f'{txt_path.__str__()}/{new_dataset_id}_{id_z}.txt'
    #     text_file = open(txt_file, "w")
    #     n = text_file.write(txt)
    #     text_file.close()



    # save the data into coca folder
    l=0
    t_all = []
    for sample_1 in range(5):
        dataset_id = f'coca_spok_filter_punct_10K_sample_{sample_1+1}'
        extractor_id = f'group=gpt2-xl_layers-dataset={dataset_id}-activation-bench=None-ave=None'
        extractor_obj = extract_pool[extractor_id]()
        extractor_obj.load_dataset()
        t_all.append(extractor_obj.data_)

    flat_list = [item for sublist in t_all for item in sublist]
    new_dataset_id = f'coca_spok_filter_punct_50K'
    file_loc = os.path.join(COCA_CORPUS_DIR,'coca_spok_data_filter_ngram_punct_50K.pkl')
    save_obj(flat_list,file_loc)
