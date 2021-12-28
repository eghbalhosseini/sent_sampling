import importlib
import getpass
import sys
if getpass.getuser() == 'eghbalhosseini':
    sys.path.insert(1, '/Users/eghbalhosseini/MyCodes/sent_sampling/')
elif getpass.getuser() == 'ehoseini':
    sys.path.insert(1, '/om/user/ehoseini/sent_sampling/')

from neural_nlp.benchmarks.neural import read_words, listen_to
from neural_nlp.models import model_pool, model_layers
from neural_nlp import FixedLayer

import utils
importlib.reload(utils)
from utils.data_utils import SENTENCE_CONFIG, COCA_PREPROCESSED_DIR,construct_stimuli_set_from_pd
from neural_nlp.models import model_pool
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from neural_nlp.stimuli import load_stimuli, StimulusSet
import itertools
from neural_nlp.utils import ordered_set

from neural_nlp.utils import ordered_set
import utils.extract_utils
importlib.reload(utils.extract_utils)
import glob
import xarray as xr
import itertools
import numpy as np


# function similar to the one used in neural nlp
def align_tokens_debug(tokenized_sentences, sentences, max_num_words, additional_tokens, use_special_tokens,special_tokens):
    # sliding window approach (see https://github.com/google-research/bert/issues/66)
    # however, since this is a brain model candidate, we don't let it see future words (just like the brain
    # doesn't receive future word input). Instead, we maximize the past context of each word
    sentence_index = 0
    sentences_chain = ' '.join(sentences).split()
    previous_indices = []
    all_context=[]
    for token_index in tqdm(range(len(tokenized_sentences)), desc='token features',position=2,leave=False,disable=True):
        if tokenized_sentences[token_index] in additional_tokens:
            continue  # ignore altogether
        # combine e.g. "'hunts', '##man'" or "'jennie', '##s'"
        tokens = [
            # tokens are sometimes padded by prefixes, clear those here
            word.lstrip('##').lstrip('▁').rstrip('@@')
            for word in tokenized_sentences[previous_indices + [token_index]]]
        token_word = ''.join(tokens).lower()
        for special_token in special_tokens:
            token_word = token_word.replace(special_token, '')
        if sentences_chain[sentence_index].lower() != token_word:
            previous_indices.append(token_index)
            continue
        previous_indices = []
        sentence_index += 1

        context_start = max(0, token_index - max_num_words + 1)
        context = tokenized_sentences[context_start:token_index + 1]
        if use_special_tokens and context_start > 0:  # `cls_token` has been discarded
            # insert `cls_token` again following
            # https://huggingface.co/pytorch-transformers/model_doc/roberta.html#pytorch_transformers.RobertaModel
            context = np.insert(context, 0, tokenized_sentences[0])
        #context_ids = self.tokenizer.convert_tokens_to_ids(context)
        all_context.append(context)
        #yield context
    return all_context

def auto_merge(datasets):
    """
    Automatically merge a split xarray Dataset. This is designed to behave like
    `xarray.open_mfdataset`, except it supports concatenation along multiple
    dimensions.
    Parameters
    ----------
    datasets : str or list of str or list of xarray.Dataset
        Either a glob expression or list of paths as you would pass to
        xarray.open_mfdataset, or a list of xarray datasets. If a list of
        datasets is passed, you should make sure that they are represented
        as dask arrays to avoid reading the whole dataset into memory.
    Returns
    -------
    xarray.Dataset
        The merged dataset.
    """
    # Treat `datasets` as a glob expression
    if isinstance(datasets, str):
        datasets = glob.glob(datasets)

    # Treat `datasets` as a list of file paths
    if isinstance(datasets[0], str):
        # Pass chunks={} to ensure the dataset is read as a dask array
        datasets = [xr.open_dataset(path, chunks={}) for path in datasets]

    def _combine_along_last_dim(datasets):
        merged = []

        # Determine the dimension along which the dataset is split
        split_dims = [d for d in datasets[0].dims if
                      len(np.unique([ds[d].values[0] for ds in datasets])) > 1]

        # Concatenate along one of the split dimensions
        concat_dim = split_dims[-1]

        # Group along the remaining dimensions and concatenate within each
        # group.
        sorted_ds = sorted(datasets, key=lambda ds: tuple(ds[d].values[0]
                                                          for d in split_dims))
        for _, group in itertools.groupby(
                sorted_ds,
                key=lambda ds: tuple(ds[d].values[0] for d in split_dims[:-1])
                ):
            merged.append(xr.auto_combine(group, concat_dim=concat_dim))

        return merged

    merged = datasets
    while len(merged) > 1:
        merged = _combine_along_last_dim(merged)

    return merged[0]

from xarray import concat
import numpy as np

def _concat_dicts(dict_objs, dim, data_vars, **kwargs):
    objs = [dict_obj['key'] for dict_obj in dict_objs]
    return {'key': concat(objs, dim, data_vars, **kwargs)}

def _concat_nd(obj_grid, concat_dims=None, data_vars=None, **kwargs):
    # Combine datasets along one dimension at a time,
    # Have to start with last axis and finish with axis=0 otherwise axes will disappear before the loop reaches them
    for axis in reversed(range(obj_grid.ndim)):
        obj_grid = np.apply_along_axis(_concat_dicts, axis, arr=obj_grid,
                                       dim=concat_dims[axis], data_vars=data_vars[axis], **kwargs)

    # Grid should now only contain one dict which contains the concatenated xarray object
    return obj_grid.item()['key']

model='xlnet-large-cased'
model_activation_set=[]
model_impl = model_pool[model]
layers = model_layers[model]
layer_id=0
candidate=FixedLayer(model_impl, layers[layer_id], prerun=layers if layer_id == 0 else None)
stimuli_pd=pd.read_pickle(str(Path(COCA_PREPROCESSED_DIR,'coca_preprocessed_all_clean_100K_sample_1.pkl')))
ordered_sets=construct_stimuli_set_from_pd(stimuli_pd)

all_num_words=[]
all_num_words_aligned=[]
all_correspondances=[]
for id in tqdm(range(len(ordered_sets)),desc='set:',leave=True):
    stimulus_set = ordered_sets[id]
    all_sentences = stimulus_set.groupby('sentence_id').apply(lambda x: ' '.join(x.word))
    num_words = [len(sentence.split()) for sentence in all_sentences.values]
    num_words_aligned = []
    for idx in tqdm(range(len(all_sentences.values)),desc='sentences:',position=1,leave=False,disable=True):
        sentence=all_sentences.values[idx]
        tokenized_sentences = [candidate._model._model_container.tokenizer.tokenize(s) for s in [sentence]]
        tokenized_sentences = list(itertools.chain.from_iterable(tokenized_sentences))
        tokenized_sentences = np.array(tokenized_sentences)
        aligned_tokens_debug=align_tokens_debug(tokenized_sentences=tokenized_sentences,additional_tokens=[],sentences=[sentence],max_num_words=512,use_special_tokens=False,special_tokens=candidate._model._model_container.tokenizer_special_tokens)
        num_words_aligned.append(len(aligned_tokens_debug))
        assert(len(aligned_tokens_debug)==len(sentence.split()))
        sentence_set=stimulus_set[stimulus_set.sentence_id==all_sentences.index[idx]]
        #model_activations = read_words(candidate, sentence_set, copy_columns=['stimulus_id'], average_sentence=False)

    correspondsance=np.equal(np.asarray(num_words),np.asarray(num_words_aligned))
    all_correspondances.append(correspondsance)
flat_list = [item for sublist in all_correspondances for item in sublist]
np.sum(np.asarray(flat_list))/len(flat_list)
# kk=0

idx=0
stimulus_set = ordered_sets[id]
all_sentences = stimulus_set.groupby('sentence_id').apply(lambda x: ' '.join(x.word))

sentence_set=stimulus_set[stimulus_set.sentence_id<all_sentences.index[4]]
model_activations = read_words(candidate, sentence_set, copy_columns=['stimulus_id'], average_sentence=False)


reset_column='sentence_id'
copy_columns=['stimulus_id']
activations = []
for i, reset_id in enumerate(ordered_set(sentence_set[reset_column].values)):
    part_stimuli = sentence_set[sentence_set[reset_column] == reset_id]
    # stimulus_ids = part_stimuli['stimulus_id']
    sentence_stimuli = StimulusSet({'sentence': ' '.join(part_stimuli['word']),
                                    reset_column: list(set(part_stimuli[reset_column]))})
    sentence_stimuli.name = f"{stimulus_set.name}-{reset_id}"
    print(f"running {sentence_stimuli.name} : {' '.join(part_stimuli['word'])}\n")
    sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=False)
    for column in copy_columns:
        sentence_activations[column] = ('presentation', part_stimuli[column])
    activations.append(sentence_activations)
model_activations = xr.concat(activations,dim='presentation')

idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]

assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
model_activations = model_activations[{'presentation': idx}]


_concat_nd(activations)