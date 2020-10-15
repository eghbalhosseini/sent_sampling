from utils.data_utils import load_obj, construct_stimuli_set, SENTENCE_CONFIG, BENCHMARK_CONFIG
from neural_nlp.benchmarks.neural import read_words
from neural_nlp.models import model_pool, model_layers
from neural_nlp import FixedLayer
import os
import pandas as pd
import numpy as np
import copy
import xarray as xr



class extractor:
    def __init__(self,dataset=None,datafile=None,model_spec=None,layer_spec=None,extract_type='activations',extract_benchmark='',average_sentence=False):
        ##### DATA ####
        self.dataset=dataset # name of the dataset
        self.datafile = datafile  # name of the dataset
        self.model_spec=model_spec # set of models to run plus the layers to extract from
        self.layer_spec=layer_spec
        self.extract_type=extract_type # is the extraction based on activations or predicted brain response
        self.extract_benchmark=extract_benchmark
        self.average_sentence=average_sentence # which representation to output, last token, or average of all tokens.


    def load_dataset(self):
        data_ = load_obj(self.datafile)
        stimuli_set = construct_stimuli_set(data_, self.dataset)
        self.stimuli_set=stimuli_set
        self.N_S=int(np.unique(self.stimuli_set['sentence_id']).shape[0])

    def get_last_word_activations(self,activations):
        sentence_id=np.unique(np.asarray(activations.sentence_id))
        last_word_activations=[]
        for id in sentence_id:
            sentence_activation=activations.where(activations.sentence_id == id, drop=True)
            last_word_activations.append(sentence_activation.values[-1,:])
        return last_word_activations

    def get_mean_activations(self,activations):
        sentence_id=np.unique(np.asarray(activations.sentence_id))
        mean_activations=[]
        for id in sentence_id:
            sentence_activation=activations.where(activations.sentence_id == id, drop=True)
            mean_activations.append(sentence_activation.mean(dim='presentation').values)
        return mean_activations

    def extract_brain_response(self,model,layer):
        results_file = 'benchmark=' + self.extract_benchmark + ',model=' + model + ',subsample=None.pkl'
        results_ = pd.read_pickle(os.path.join(BENCHMARK_CONFIG['file_loc'], results_file))
        results_ = results_['data']
        # get specific layer results
        layer_weights=results_.attrs['layer_weights'][layer]
        # get layer activations
        test1 = model_pool[model]
        layers = model_layers[model]
        candidate = FixedLayer(test1, layers[layer], prerun=None)
        model_activations = read_words(candidate, self.stimuli_set, copy_columns=['stimulus_id'],
                                       average_sentence=self.average_sentence)  #
        brain_response_split=[]
        for split in layer_weights:
            brain_response=xr.dot(split,model_activations,dims=['neuroid'])
            brain_response_split.append(brain_response.transpose())

        return brain_response_split

    def extract_representation(self,model,layer):
        test1 = model_pool[model]
        layers = model_layers[model]
        candidate = FixedLayer(test1, layers[layer], prerun=None)
        model_activations = read_words(candidate, self.stimuli_set, copy_columns=['stimulus_id'],
                                       average_sentence=self.average_sentence)  #
        if self.average_sentence:
            model_activations=self.get_mean_activations(model_activations)
        else:
            model_activations=self.get_last_word_activations(model_activations)

        return model_activations
    # TODO : for fMRI do average,
    # TODO : settle on how to get sentence representation for ECOG --> Concatenate, average --> ask Ev


    def __call__(self, *args, **kwargs):
        model_grp_activations=[]
        for idx, model_id in enumerate(self.model_spec):
            if self.extract_type=='network_act':
                print(f"extracting network activations for {self.model_spec[idx]}")
                model_activation=self.extract_representation(self.model_spec[idx],self.layer_spec[idx])
                activation=dict(model_name=self.model_spec[idx],layer=self.layer_spec[idx],activations=model_activation)
                model_grp_activations.append(activation)
            elif self.extract_type=='brain_resp':
                print(f"extracting brain response for {self.model_spec[idx]}")
                brain_response_split=self.extract_brain_response(self.model_spec[idx],self.layer_spec[idx])
                activation_set=[]
                for x in brain_response_split:
                    last_word_response=self.get_last_word_activations(x)
                    activation_set.append(dict(model_name=self.model_spec[idx], layer=self.layer_spec[idx],
                                      activations=last_word_response))
                model_grp_activations.append(activation_set)

            self.model_group_act=model_grp_activations

        pass

