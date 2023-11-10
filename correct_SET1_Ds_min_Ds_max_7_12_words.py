import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
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
from pathlib import Path
import torch
from utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import pickle
if __name__ == '__main__':
    modelnames = ['roberta-base',  'xlnet-large-cased',  'bert-large-uncased','xlm-mlm-en-2048', 'gpt2-xl', 'albert-xxlarge-v2','ctrl']

    extract_id = [
        f'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_len_7_14_textNoPeriod-activation-bench=None-ave=False']
    #optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True',
    #             'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=True-run_gpu=True']

    group_id = extract_id[0].split('-')[0].split('=')[1]
    dataset_id=extract_id[0].split('-')[1].split('=')[1]
    # get obj= from optim_id

    # get n_samples from each element in optim_id
    Ds_modified_sentences='ANNSET_DS_MIN_MAX_from_100ev_eh.csv'
    Ds_modified_sentences = Path(ANALYZE_DIR, Ds_modified_sentences)

    Ds_modified_sentences = pd.read_csv(Ds_modified_sentences)



    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    ext_obj()

    # find location of DS_MAX sentences in ext_obj.data_
    Ds_max_sentences = Ds_modified_sentences['DS_MAX']
    Ds_max_selected= Ds_modified_sentences['max_include']
    Ds_max_sentences_edited=Ds_modified_sentences['DS_MAX_edited']
    Ds_min_sentences = Ds_modified_sentences['DS_MIN']
    Ds_min_selected= Ds_modified_sentences['min_include']
    Ds_min_sentences_edited = Ds_modified_sentences['DS_MIN_edited']
    Ds_random_sentences = Ds_modified_sentences['DS_RAND']
    Ds_random_selected= Ds_modified_sentences['rand_include']
    Ds_random_sentences_edited = Ds_modified_sentences['DS_RAND_edited']


    # remove index after 102
    Ds_max_sentences = Ds_max_sentences[:100]
    Ds_max_selected = Ds_max_selected[:100]
    Ds_max_sentences_edited = Ds_max_sentences_edited[:100]


    Ds_min_sentences = Ds_min_sentences[:100]
    Ds_min_selected = Ds_min_selected[:100]
    Ds_min_sentences_edited = Ds_min_sentences_edited[:100]

    Ds_random_sentences = Ds_random_sentences[:100]
    Ds_random_selected = Ds_random_selected[:100]
    Ds_random_sentences_edited = Ds_random_sentences_edited[:100]
    # get text fieild from each element in ext_obj.data_
    sentences = [x['text'] for x in ext_obj.data_]
    # drop the period in the end for each element in sentences
    sentences = [x[:-1] for x in sentences if x[-1] == '.']

    # find the location of sentence in ext_obj.data_
    Ds_max_sentences_location = []
    Ds_min_sentences_location = []
    Ds_random_sentences_location = []
    # get text from ext
    for i in range(len(Ds_max_sentences)):
        Ds_max_sentences_location.append(sentences.index(Ds_max_sentences[i]))
        Ds_min_sentences_location.append(sentences.index(Ds_min_sentences[i]))
        Ds_random_sentences_location.append(sentences.index(Ds_random_sentences[i]))

    # make sure the length Ds_max_sentences_location is 100
    assert len(Ds_max_sentences_location) == 100
    assert len(Ds_min_sentences_location) == 100
    assert len(Ds_random_sentences_location) == 100

    # get the ext.data_ for Ds_max_sentences
    Ds_max_sentences_data = [ext_obj.data_[i] for i in Ds_max_sentences_location]
    Ds_max_sentences_data_selected = [ext_obj.data_[p] for p in [x  for i, x in enumerate(Ds_max_sentences_location) if Ds_max_selected[i]==1.0]]
    Ds_min_sentences_data = [ext_obj.data_[i] for i in Ds_min_sentences_location]
    Ds_min_sentences_data_selected = [ext_obj.data_[p] for p in [x  for i, x in enumerate(Ds_min_sentences_location) if Ds_min_selected[i]==1.0]]
    Ds_random_sentences_data = [ext_obj.data_[i] for i in Ds_random_sentences_location]
    Ds_random_sentences_data_selected = [ext_obj.data_[p] for p in [x  for i, x in enumerate(Ds_random_sentences_location) if Ds_random_selected[i]==1.0]]
    # replace the text field with Ds_max_sentences_edited
    for i in range(len(Ds_max_sentences_data)):
        Ds_max_sentences_data[i]['text'] = Ds_max_sentences_edited[i]
        Ds_min_sentences_data[i]['text'] = Ds_min_sentences_edited[i]
        Ds_random_sentences_data[i]['text'] = Ds_random_sentences_edited[i]
    #
    Ds_max_sentences_edited_selected=[x for i,x in enumerate(Ds_max_sentences_edited) if Ds_max_selected[i]==1.0]
    Ds_min_sentences_edited_selected=[x for i,x in enumerate(Ds_min_sentences_edited) if Ds_min_selected[i]==1.0]
    Ds_random_sentences_edited_selected=[x for i,x in enumerate(Ds_random_sentences_edited) if Ds_random_selected[i]==1.0]
    for i in range(len(Ds_max_sentences_data_selected)):
        Ds_max_sentences_data_selected[i]['text'] = Ds_max_sentences_edited_selected[i]
        Ds_min_sentences_data_selected[i]['text'] = Ds_min_sentences_edited_selected[i]
        Ds_random_sentences_data_selected[i]['text'] = Ds_random_sentences_edited_selected[i]



    # fix the issue with one of the sentences in ds_max data 76
    Ds_max_sentences_data[78]['text']='This woman should be working in Supercuts... if that'


    file_name = Path(UD_PARENT, 'ud_sentencez_ds_max_100_edited.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(Ds_max_sentences_data, f, protocol=4)


    file_name = Path(UD_PARENT, 'ud_sentencez_ds_min_100_edited.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(Ds_min_sentences_data, f, protocol=4)

    file_name = Path(UD_PARENT, 'ud_sentencez_ds_random_100_edited.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(Ds_random_sentences_data, f, protocol=4)

    file_name = Path(UD_PARENT, 'ud_sentencez_ds_max_100_edited_selected.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(Ds_max_sentences_data_selected, f, protocol=4)

    file_name = Path(UD_PARENT, 'ud_sentencez_ds_min_100_edited_selected.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(Ds_min_sentences_data_selected, f, protocol=4)

    file_name = Path(UD_PARENT, 'ud_sentencez_ds_random_100_edited_selected.pkl')
    with open(file_name.__str__(), 'wb') as f:
        pickle.dump(Ds_random_sentences_data_selected, f, protocol=4)


    extract_id =f'group=best_performing_pereira_1-dataset=ud_sentencez_ds_max_100_edited_selected_textNoPeriod-activation-bench=None-ave=False'
    ext_obj=extract_pool[extract_id]()
    ext_obj.load_dataset()
    ext_obj()
    optimizer_obj = optim_pool['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True']()

    optimizer_obj.load_extractor(ext_obj)
    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=False,
                                             save_results=False)

    n_samples = optimizer_obj.N_s
    sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_S,replace=False))
    d_s_r, RDM_r = optimizer_obj.gpu_object_function_debug(sent_random)



