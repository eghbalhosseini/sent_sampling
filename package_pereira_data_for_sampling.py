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
        expr_save=f'Pereira2018_{topic}'
        textformat='textNoPeriod'
        pereira_set=pd.read_csv('/om5/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Pereira2018-stimulus_set.csv')
        pereira_243=pereira_set[pereira_set['experiment']==topic]
        # get P243 sentences
        pereira_243_sentences=pereira_243['sentence'].values
        pereira_243_sentence_index=pereira_243['sentence_num'].values
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
            #ordered=np.argsort(sent_order )
            a_list=[]
            for kk in sent_order:
                 a=pd.read_pickle(sent_set[kk])
                 a=a['data']
                 a_list.append(a)
            a_concat=xr.concat(a_list,dim='presentation')
             # save based on layer
            SAVE_DIR = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/'
            # sort a_concat.stimulus_sentence based on pereira_243_sentences
            # sort a_concat based on pereira_243_sentence
            reordering_a=[list(a_concat.stimulus_sentence.values).index(x) for x in pereira_243_sentences]
            # sort a_concat based on reordering_a
            a_concat=a_concat.isel(presentation=reordering_a)
            assert(np.all(a_concat.stimulus_sentence.values==pereira_243_sentences))

            for idx,grp in tqdm(enumerate(a_concat.groupby('layer'))):
                g_idx,g=grp
                # sort based on sentence order
                sentences=[x[0] for x in g.presentation.values]
                # make sure sentences and pereria_243_sentences are the same
                assert(np.all(sentences==pereira_243_sentences))
                # save the data in the same format ast sent_sampling dat
                data_list=[]
                # find the row in g that correspond to each sentence in
                # pereira_243_sentences
                for id_sent,sent in enumerate(pereira_243_sentences):
                    row=sentences.index(sent)
                    data_list.append([g[row].values,sent,pereira_243_sentence_index[id_sent]])
                #for id_sent,x in enumerate(sentences):
                #     data_list.append([np.reshape(g.values[id_sent,:],(1,-1)),x])
                with open(os.path.join(SAVE_DIR,f'{expr_save.lower()}_{textformat}_{model_id}_layer_{idx}_activation_ave_False.pkl'),'wb') as f:
                     pickle.dump(data_list, f, protocol=4)