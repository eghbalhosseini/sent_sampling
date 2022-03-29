import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.extract_utils import model_extractor
from utils.data_utils import RESULTS_DIR, save_obj,load_obj,SAVE_DIR,COCA_CORPUS_DIR
from utils.data_utils import SENTENCE_CONFIG
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cmudict
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt

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
    # range of word per min :

    flat_list = [item for sublist in t_all for item in sublist]
    SSP = SyllableTokenizer()
    sent_syll=[];
    for idx, sent_dat in tqdm(enumerate(flat_list)):
        sent_syll.append([SSP.tokenize(token) for token in word_tokenize(sent_dat['text'])])
    # remove last token [.]
    k=0
    bad_sent_id=[]
    good_sent_id=[]
    for idx, syll_dat in tqdm(enumerate(sent_syll)):
        if (syll_dat[-1]==['.']):
            good_sent_id.append(idx)
        else:
            k=k+1
            bad_sent_id.append(idx)
        # sent_syll.append([SSP.tokenize(token) for token in word_tokenize(sent_dat['text'])])
    syll_cnt=[]
    for _,idx in tqdm(enumerate(good_sent_id)):
        syll_cnt.append(len(sent_syll[idx]))

    plt.hist(syll_cnt)
    plt.show()
    # range of word per min :
    # 2 TR to 3 TR
    TR=2
    max_length=2
    min_length = 1
    syl_per_sec=5
    max_buffer=-4
    min_buffer=1

    good_sent_id_np=np.asarray(good_sent_id)

    #min_word_per_sec = min_length * TR * ((162 + 230) / 2) / 60
    min_word_per_sec = min_length * TR * syl_per_sec + min_buffer
    max_word_per_sec = max_length * TR * syl_per_sec + max_buffer

    ones_to_include=np.logical_and(np.asarray(syll_cnt) <=max_word_per_sec , np.asarray(syll_cnt) >=min_word_per_sec)
    #


    new_list=[flat_list[x] for x in good_sent_id_np[ones_to_include]]
    len(new_list)
    pp = [x['sentence_length'] for x in new_list]

    plt.hist(pp)
    plt.show()

    new_dataset_id = f'coca_spok_data_filter_ngram_punct_50K_sylb.pkl'

    file_loc = os.path.join(COCA_CORPUS_DIR,new_dataset_id)
    save_obj(new_list,file_loc)
    # save a syllable version
    dataset_id = f'coca_spok_filter_punct_50K'
    extractor_id = f'group=gpt2-xl_layers-dataset={dataset_id}-activation-bench=None-ave=None'
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()

    for i, layer in enumerate(tqdm(range(49), desc='layers')):
        model_activation_name = f"{extractor_obj.dataset}_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_{extractor_obj.average_sentence}.pkl"
        if os.path.exists(os.path.join(SAVE_DIR, model_activation_name)):
            t = load_obj(os.path.join(SAVE_DIR, model_activation_name))
            t_mod= [t[x] for x in good_sent_id_np[ones_to_include]]
            assert(np.array_equal(pp,np.asarray([x[0].shape[0] for x in t_mod])))
            new_model_actvation_name = f"{extractor_obj.dataset}_sylb_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_None.pkl"
            save_obj(t_mod, os.path.join(SAVE_DIR, new_model_actvation_name))

            t_last=[]
            for id,x in tqdm(enumerate(t_mod),total=len(t_mod)):
                mod=[x[0][[-1],:],x[1],x[2]]
                t_last.append(mod)
            new_model_activation_name=f"{extractor_obj.dataset}_sylb_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_False.pkl"
            save_obj(t_last, os.path.join(SAVE_DIR, new_model_activation_name))



    # test loading 50K extractor
    dataset_id = f'coca_spok_filter_punct_50K'
    extractor_id = f'group=gpt2-xl_layers-dataset={dataset_id}-activation-bench=None-ave=None'
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()

    for i, layer in enumerate(tqdm(range(49), desc='layers')):
        model_activation_name = f"{extractor_obj.dataset}_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_{extractor_obj.average_sentence}.pkl"
        if os.path.exists(os.path.join(SAVE_DIR, model_activation_name)):
            t = load_obj(os.path.join(SAVE_DIR, model_activation_name))
            t_last=[]
            for id,x in tqdm(enumerate(t)):
                mod=[x[0][[-1],:],x[1],x[2]]
                t_last.append(mod)
            new_model_activation_name=f"{extractor_obj.dataset}_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_False.pkl"
            save_obj(t_last, os.path.join(SAVE_DIR, new_model_activation_name))


