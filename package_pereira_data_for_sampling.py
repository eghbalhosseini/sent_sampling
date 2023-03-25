'''Note : run this with neural_nlp_2022 enviornment '''

from glob import glob
import numpy as np
import os
import pandas as pd
import re
import pandas as pd
import xarray as xr
from tqdm import tqdm
import pickle
from scipy.spatial.distance import pdist, squareform
import pickle, dill
def find(s, el):
    for i in s.index:
        if s[i] == el:
            return i
    return None
is_sorted = lambda a: np.all(a[:-1] <= a[1:])

def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

if __name__ == '__main__':
        model_id_list=['xlnet-large-cased',
                        'xlm-mlm-en-2048'
                     ,'albert-xxlarge-v2'
         ,'bert-large-uncased-whole-word-masking'
         ,'roberta-base'
         ,'gpt2-xl'
         ,'ctrl']
        topic='384sentences'
        expr=f'Pereira2018-{topic}'
        pereira_set=pd.read_csv('/om5/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Pereira2018-stimulus_set.csv')
        pereira_243=pereira_set[pereira_set['experiment']==topic]
        ordered_topic=unique(pereira_243.passage_category)
        for model_id in model_id_list:
            pattern = os.path.join(
                 '/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored',
                 f'identifier={model_id},stimuli_identifier={expr}*.pkl')
            sent_set=glob(pattern)
            len(sent_set)
            sent_topic=[re.findall(f'{topic}.\w+', x)[0].replace(f'{topic}.','') for x in sent_set]
            assert(len(ordered_topic)==len(sent_topic))
            sent_order=[sent_topic.index(x) for x in ordered_topic]
            ordered=np.argsort(sent_order )
            a_list=[]
            for kk in ordered:
                 with open(sent_set[kk], 'rb') as f:
                      a=dill.load(f)
                 a=pd.read_pickle(sent_set[kk])
                 a=a['data']
                 a_list.append(a)
            a_concat=xr.concat(a_list,dim='presentation')
             # save based on layer
            SAVE_DIR = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/'
            for idx,grp in tqdm(enumerate(a_concat.groupby('layer'))):
                 g_idx,g=grp
                 sentences=[x[0] for x in g.presentation.values]
                 # save the data in the same format ast sent_sampling dat
                 data_list=[]
                 for id_sent,x in enumerate(sentences):
                     data_list.append([np.reshape(g.values[id_sent,:],(1,-1)),x])
                 with open(os.path.join(SAVE_DIR,f'{expr.lower()}_{model_id}_layer_{idx}_activation_ave_False.pkl'),'wb') as f:
                     pickle.dump(data_list, f, protocol=4)