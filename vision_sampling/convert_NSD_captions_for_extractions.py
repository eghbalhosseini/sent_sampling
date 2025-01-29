import re
import numpy as np
import pandas as pd
import getpass
import os
from pathlib import Path
import sys
from pathlib import Path
import numpy as np
from sent_sampling.utils.data_utils import construct_stimuli_set_from_pd


if __name__ == '__main__':
    # read the file
    p=Path('/Users/eghbalhosseini/MyCodes/DeepJuiceDev/benchmarks/stimulus/NSD_shared1000.csv')
    df=pd.read_csv(p.__str__())
    # read the colum coco_captions as a list
    df['coco_captions']=df['coco_captions'].apply(lambda x: x.split(','))
    # get the sentences
    sentences=df['coco_captions'].values
    # pick the first sentenec from each list
    sentences=[x.split('\',')[0] for x in sentences]

   # drop [' in the beginign of each sentence
    sentences=[re.sub(r'^\[\'','',x) for x in sentences]
    # sent_id, word_from
    # split sentences into words
    words=[x.split(' ') for x in sentences]
    word_form=words
    # create a counter for word id in each sentence
    word_id=[list(range(len(x))) for x in words]
    # create a counter for sentence id based on the words in each sentence
    sent_id=[list(idx*np.ones(len(x)).astype(int)) for idx, x in enumerate(words)]


    # flatten the list
    words=[item for sublist in words for item in sublist]
    word_form=[item for sublist in word_form for item in sublist]
    word_id=[item for sublist in word_id for item in sublist]
    sent_id=[item for sublist in sent_id for item in sublist]
    # combine words, words_id, and sent_id  to create a dataframe with 3 columns
    df_extract=pd.DataFrame({'word':words,'word_id':word_id,'sent_id':sent_id,'word_form':word_form})
    # save the dataframe as a picklefile in the same directory
    df_extract.to_pickle(p.parent.joinpath('beta-control-neural_stimset_D-S_light_freq_extract.pkl').__str__())