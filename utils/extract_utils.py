from utils.data_utils import load_obj, construct_stimuli_set, SENTENCE_CONFIG, BENCHMARK_CONFIG, save_obj, SAVE_DIR
from neural_nlp.benchmarks.neural import read_words
from neural_nlp.models import model_pool, model_layers
from neural_nlp import FixedLayer
from brainio_base.assemblies import NeuroidAssembly
import os
import pandas as pd
import numpy as np
import copy
import xarray as xr
from tqdm import tqdm


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
        self.N_S=int(len(data_))

    def get_last_word_activations(self,activations):
        sentence_id=np.unique(np.asarray(activations.sentence_id))
        last_word_activations=[]
        for idx, id in tqdm(enumerate(sentence_id)):
            sentence_activation=activations.where(activations.sentence_id == id, drop=True)
            last_word_activations.append(sentence_activation.values[-1,:])
        return last_word_activations

    def get_mean_activations(self,activations):
        sentence_id=np.unique(np.asarray(activations.sentence_id))
        mean_activations=[]
        for idx, id in tqdm(enumerate(sentence_id)):
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
        # additionally check if the result for network activation exist, that can be used here!
        model_activation_ver = f"{self.dataset}_{model}_layer_{layer}_network_act_ave_{self.average_sentence}.pkl"
        if os.path.exists(os.path.join(SAVE_DIR, model_activation_ver)):
            print(f"{model_activation_ver} can be used for computing brain prediction, loading...")
            model_activation = load_obj(os.path.join(SAVE_DIR, model_activation_ver))
            model_activation_flat=model_activation
        else:
            print(f"{model_activation_ver} was not found ,computing ...")
            model_activation_set = []
            test1 = model_pool[model]
            layers = model_layers[model]
            candidate = FixedLayer(test1, layers[layer], prerun=None)
            for stim_id, stim in enumerate(self.stimuli_set):
                model_activations = read_words(candidate, stim, copy_columns=['stimulus_id'],
                                       average_sentence=self.average_sentence)  #
                if self.average_sentence:
                    model_activations=self.get_mean_activations(model_activations)
                else:
                    model_activations=self.get_last_word_activations(model_activations)
                model_activation_set.append(model_activations)
            model_activation_flat = [item for sublist in model_activation_set for item in sublist]
        brain_response_split=[]
        model_activation_assembly=xr.DataArray(model_activation_flat,dims=('presentation', 'neuroid')
                                               ,coords={'neuroid':layer_weights.coords['neuroid']})
        for split in layer_weights:
            brain_response=xr.dot(split,model_activation_assembly,dims=['neuroid'])
            brain_response_split.append(brain_response.transpose())

        return brain_response_split

    def extract_representation(self,model,layer):
        model_activation_set=[]
        test1 = model_pool[model]
        layers = model_layers[model]
        candidate = FixedLayer(test1, layers[layer], prerun=None)
        for stim_id, stim in enumerate(self.stimuli_set):
            model_activations = read_words(candidate, stim, copy_columns=['stimulus_id'],average_sentence=self.average_sentence)  #
            if self.average_sentence:
                model_activations=self.get_mean_activations(model_activations)
            else:
                model_activations=self.get_last_word_activations(model_activations)
            model_activation_set.append(model_activations)
        model_activation_flat=[item for sublist in model_activation_set for item in sublist]
        return model_activation_flat
    # TODO : for fMRI do average,
    # TODO : settle on how to get sentence representation for ECOG --> Concatenate, average --> ask Ev


    def __call__(self, *args, **kwargs):
        model_grp_activations=[]
        for idx, model_id in enumerate(self.model_spec):

            if self.extract_type=='network_act':
                model_activation_name = f"{self.dataset}_{self.model_spec[idx]}_layer_{self.layer_spec[idx]}_{self.extract_type}_ave_{self.average_sentence}.pkl"
                print(f"extracting network activations for {self.model_spec[idx]}")
                # see whether model activation already extracted
                if os.path.exists(os.path.join(SAVE_DIR,model_activation_name)):
                    print(f"{model_activation_name} already exists, loading...")
                    model_activation=load_obj(os.path.join(SAVE_DIR,model_activation_name))
                else:
                    print(f"{model_activation_name} doesn't exists, creating...")
                    model_activation=self.extract_representation(self.model_spec[idx],self.layer_spec[idx])
                    save_obj(model_activation, os.path.join(SAVE_DIR, model_activation_name))
                print('adding activations to set')
                activation=dict(model_name=self.model_spec[idx],layer=self.layer_spec[idx],activations=model_activation)
                model_grp_activations.append(activation)
            elif self.extract_type=='brain_resp':
                brain_resp_name = f"{self.dataset}_{self.model_spec[idx]}_layer_{self.layer_spec[idx]}_{self.extract_type}_{self.extract_benchmark}_ave_{self.average_sentence}.pkl"
                print(f"extracting brain response for {self.model_spec[idx]}")
                if os.path.exists(os.path.join(SAVE_DIR, brain_resp_name)):
                    print(f"{brain_resp_name} already exists, loading...")
                    brain_response_split = load_obj(os.path.join(SAVE_DIR, brain_resp_name))
                else:
                    print(f"{brain_resp_name} doesn't exists, creating...")
                    brain_response_split=self.extract_brain_response(self.model_spec[idx],self.layer_spec[idx])
                    save_obj(brain_response_split, os.path.join(SAVE_DIR, brain_resp_name))

                print('computed brain responses')
                activation_set=[]
                for x in brain_response_split:
                    activation_set.append(dict(model_name=self.model_spec[idx], layer=self.layer_spec[idx],
                                      activations=x))
                model_grp_activations.append(activation_set)

            self.model_group_act=model_grp_activations

        pass

