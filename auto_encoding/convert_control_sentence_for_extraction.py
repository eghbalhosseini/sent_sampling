import numpy as np
import pandas as pd
import getpass
import os
from pathlib import Path
import sys
from pathlib import Path
import numpy as np
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])

if getpass.getuser() == 'eghbalhosseini':
    data_path = '/Users/eghbalhosseini/MyData/sent_sampling/auto_encoder/'
else:
    data_path = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/auto_encoder/'

from utils.data_utils import construct_stimuli_set_from_pd


if __name__ == '__main__':
    # read the file
    p=Path(data_path,'beta-control-neural_stimset_D-S_light_freq.csv')
    df=pd.read_csv(p.__str__())
    # get the sentences
    sentences=df['sentence'].values
    len(sentences)
    # sent_id, word_from
    # split sentences into words
    words=[x.split(' ') for x in sentences]
    # create a counter for word id in each sentence
    word_id=[list(range(len(x))) for x in words]
    # create a counter for sentence id based on the words in each sentence
    sent_id=[list(idx*np.ones(len(x)).astype(int)) for idx, x in enumerate(words)]


    # flatten the list
    words=[item for sublist in words for item in sublist]
    word_id=[item for sublist in word_id for item in sublist]
    sent_id=[item for sublist in sent_id for item in sublist]
    # combine words, words_id, and sent_id  to create a dataframe with 3 columns
    df_extract=pd.DataFrame({'word':words,'word_id':word_id,'sent_id':sent_id})
    # save the dataframe as a picklefile in the same directory
    df_extract.to_pickle(p.parent.joinpath('beta-control-neural_stimset_D-S_light_freq_extract.pkl').__str__())