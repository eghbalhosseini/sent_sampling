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
from pathlib import Path
import torch
from utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

if __name__ == '__main__':
    # read the file
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
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
    # add a period to the end of each sentence
    ds_min_ = [x + '.' for x in ds_min_]
    ds_max_ = [x + '.' for x in ds_max_]
    ds_rand_ = [x + '.' for x in ds_rand_]
    # construct a dataframe with 3 columns one for sentence, one fo sentnece type and one for sentence id in group and one for total sentence id
    df=pd.DataFrame({'sentence':ds_min_,'sentence_type':'ds_min','sentence_id_group':list(range(len(ds_min_)))})
    df=df.append(pd.DataFrame({'sentence':ds_max_,'sentence_type':'ds_max','sentence_id_group':list(range(len(ds_max_)))}))
    df=df.append(pd.DataFrame({'sentence':ds_rand_,'sentence_type':'ds_rand','sentence_id_group':list(range(len(ds_rand_)))}))
    df['sentence_id']=list(range(len(df)))

    # get the sentences
    sentences=df['sentence'].values
    sentence_type=df['sentence_type'].values
    sentence_id_in_group=df['sentence_id_group'].values


    len(sentences)
    # sent_id, word_from
    # split sentences into words
    words=[x.split(' ') for x in sentences]
    word_form=words
    # create a counter for word id in each sentence
    word_id=[list(range(len(x))) for x in words]
    # create a counter for sentence id based on the words in each sentence
    sent_id=[list(idx*np.ones(len(x)).astype(int)) for idx, x in enumerate(words)]
    # create a counter for sentence_type id based on the words in each sentence
    sent_type = [np.repeat(sentence_type[idx] ,len(x)) for idx, x in enumerate(words)]
    # create a counter for sentence_id_in_group id based on the words in each sentence
    sent_id_in_group = [np.repeat(sentence_id_in_group[idx] ,len(x)) for idx, x in enumerate(words)]

    # flatten the list
    words=[item for sublist in words for item in sublist]
    word_form=[item for sublist in word_form for item in sublist]
    word_id=[item for sublist in word_id for item in sublist]
    sent_id=[item for sublist in sent_id for item in sublist]
    sent_type=[item for sublist in sent_type for item in sublist]
    sent_id_in_group=[item for sublist in sent_id_in_group for item in sublist]
    # combine words, words_id, and sent_id  to create a dataframe with 3 columns
    df_extract=pd.DataFrame({'word':words,'word_id':word_id,'sent_id':sent_id,'word_form':word_form,'sent_type':sent_type,'sent_id_in_group':sent_id_in_group})
    # save the dataframe as a picklefile in the same directory
    df_extract.to_pickle(os.path.join('/om2/user/ehoseini/fmri_DNN/ds_parametric/','ds_parametric_extract.pkl'))