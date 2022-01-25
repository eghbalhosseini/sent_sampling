from utils.data_utils import load_obj, construct_stimuli_set, BENCHMARK_CONFIG, save_obj, SAVE_DIR,construct_stimuli_set_from_pd
from neural_nlp.benchmarks.neural import read_words, listen_to
from neural_nlp.stimuli import load_stimuli, StimulusSet
from neural_nlp.models import model_pool, model_layers
from neural_nlp.utils import ordered_set
from neural_nlp import FixedLayer
import os , itertools , fnmatch , re
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from result_caching import store
from pathlib import Path
temp_store = store()
temp_store._storage_directory
# relative path to neural_nlp storage
neural_nlp_stor_rel='neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored'
neural_nlp_store_abs=os.path.join(temp_store._storage_directory,neural_nlp_stor_rel)


# modified read_words
def read_words_eh(candidate, stimulus_set, reset_column='sentence_id', copy_columns=(), average_sentence=False,overwrite=False):
    """
    Pass a `stimulus_set` through a model `candidate`.
    In contrast to the `listen_to` function, this function operates on a word-based `stimulus_set`.
    modified be eh to work with xarray 0.16+ and pandas 1.2.4
    """
    # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
    activations = []
    # remove previous saved activation
    if overwrite:
        print('removing previous run\n')
        for i, reset_id in tqdm(enumerate(ordered_set(stimulus_set[reset_column].values))):
            part_stimuli = stimulus_set[stimulus_set[reset_column] == reset_id]
            # stimulus_ids = part_stimuli['stimulus_id']
            sentence_stimuli = StimulusSet({'sentence': ' '.join(part_stimuli['word']),
                                            reset_column: list(set(part_stimuli[reset_column]))})
            sentence_stimuli.name = f"{stimulus_set.name}-{reset_id}"
            name_format = f'identifier={candidate._model.identifier},stimuli_identifier={sentence_stimuli.name}.pkl'
            file_loc = Path(os.path.join(neural_nlp_store_abs, name_format))
            if file_loc.exists() : file_loc.unlink()
    # run sentences
    for i, reset_id in enumerate(ordered_set(stimulus_set[reset_column].values)):
        part_stimuli = stimulus_set[stimulus_set[reset_column] == reset_id]
        # stimulus_ids = part_stimuli['stimulus_id']
        sentence_stimuli = StimulusSet({'sentence': ' '.join(part_stimuli['word']),
                                        reset_column: list(set(part_stimuli[reset_column]))})
        sentence_stimuli.name = f"{stimulus_set.name}-{reset_id}"
        print(f"running {sentence_stimuli.name} : {' '.join(part_stimuli['word'])}\n")

        # if overwrite:
        #     name_format = f'identifier={candidate._model.identifier},stimuli_identifier={sentence_stimuli.name}.pkl'
        #     file_loc=Path(os.path.join(neural_nlp_store_abs,name_format))
        #     if file_loc.exists():
        #         print(f"removing and rerunning {sentence_stimuli.name} : {' '.join(part_stimuli['word'])}\n")
        #         file_loc.unlink()
        # else:
        #     pass
        sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
        for column in copy_columns:
            sentence_activations[column] = ('presentation', part_stimuli[column])
        activations.append(sentence_activations)
    model_activations = xr.concat(activations,dim='presentation')
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]

    return model_activations


class extractor:
    def __init__(self,dataset=None,datafile=None,model_spec=None,layer_spec=None,layer_name=None,extract_name='activation',extract_type='activation',extract_benchmark='',atlas=None,average_sentence='False',modality=None):
        ##### DATA ####
        self.dataset=dataset # name of the dataset
        self.datafile = datafile  # name of the dataset
        self.model_spec=model_spec # set of models to run plus the layers to extract from
        self.layer_spec=layer_spec
        self.layer_name=layer_name
        self.extract_type=extract_type # is the extraction based on activations or predicted brain response
        self.extract_benchmark=extract_benchmark
        self.average_sentence=average_sentence # which representation to output, last token, or average of all tokens.
        self.atlas=atlas
        self.modality=modality
        self.extract_name=extract_name


    def load_dataset(self,silent=True):
        data_ = load_obj(self.datafile,silent=silent)
        self.data_=data_
        if isinstance(self.data_, pd.DataFrame):
            stimuli_set=construct_stimuli_set_from_pd(data_, self.dataset)
            self.stimuli_set = stimuli_set
            self.N_S=int(len(data_.groupby('sent_id')))
            assert(hasattr(self, 'stimuli_set'))
        else:
            stimuli_set = construct_stimuli_set(data_, self.dataset)
            self.stimuli_set = stimuli_set
            self.N_S=int(len(data_))

    def get_last_word_activations(self,activations):
        sentence_id=np.unique(np.asarray(activations.sentence_id))
        last_word_activations=[]
        for idx, id in enumerate(sentence_id):
            sentence_activation=activations.where(activations.sentence_id == id, drop=True)
            sent_string = list(set(sentence_activation.stimulus_sentence.values))[0]
            last_word_activations.append([sentence_activation.values[-1,:], sent_string, id])
            #last_word_activations.append(sentence_activation.values[-1,:])
        return last_word_activations

    def get_mean_activations(self,activations):
        sentence_id=np.unique(np.asarray(activations.sentence_id))
        mean_activations=[]
        for idx, id in tqdm(enumerate(sentence_id)):
            sentence_activation=activations.where(activations.sentence_id == id, drop=True)
            sent_string = list(set(sentence_activation.stimulus_sentence.values))[0]
            mean_activations.append([sentence_activation.mean(dim='presentation').values,sent_string,id])
            #mean_activations.append(sentence_activation.mean(dim='presentation').values)
        return mean_activations

    def get_all_activations(self,activations):
        sentence_id=np.unique(np.asarray(activations.sentence_id))
        all_activations=[]
        for idx, id in tqdm(enumerate(sentence_id)):
            sentence_activation=activations.where(activations.sentence_id == id, drop=True)
            sent_string=list(set(sentence_activation.stimulus_sentence.values))[0]
            all_activations.append([sentence_activation.values,sent_string,id])
        return all_activations

    def extract_brain_response(self,model,layer):
        results_file = 'benchmark=' + self.extract_benchmark + ',model=' + model + ',subsample=None.pkl'
        results_ = pd.read_pickle(os.path.join(BENCHMARK_CONFIG['file_loc'], results_file))
        results_ = results_['data']
        # get specific layer results
        layer_weights=results_.attrs['layer_weights'][layer]
        # get layer activations
        # additionally check if the result for network activation exist, that can be used here!
        model_activation_ver = f"{self.dataset}_{model}_layer_{layer}_activation_ave_{self.average_sentence}.pkl"
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
            if not(self.modality=='fMRI'):
                for stim_id, stim in enumerate(self.stimuli_set):
                    model_activations = read_words_eh(candidate, stim, copy_columns=['stimulus_id'],
                                       average_sentence=False)  #
                    if self.average_sentence == 'True':
                        model_activations = self.get_mean_activations(model_activations)
                    elif self.average_sentence == 'False':
                        model_activations = self.get_last_word_activations(model_activations)
                    elif self.average_sentence == 'None':
                        model_activations = self.get_all_activations(model_activations)
                    model_activation_set.append(model_activations)
            elif self.modality=='fMRI':
                #TODO check if listen_to and read_words_eh perform the same set of computation.
                for stim_id, stim in enumerate(self.stimuli_set):
                    model_activations = read_words_eh(candidate, stim, copy_columns=['stimulus_id'],
                                       average_sentence=False)  #
                    model_activations=self.get_mean_activations(model_activations)
                    model_activation_set.append(model_activations)


            model_activation_flat = [item for sublist in model_activation_set for item in sublist]

        brain_response_split=[]
        # get dividers based on the weight file, Fedorenko doesnt have an atlas:
        if self.atlas==None:
            model_activation_assembly=xr.DataArray(model_activation_flat,dims=('presentation', 'neuroid')
                                                    ,coords={'neuroid':layer_weights.coords['neuroid']})

            for split in layer_weights:
                brain_response=xr.dot(split,model_activation_assembly,dims=['neuroid'])+split.attrs['intercept']
                brain_response_split.append(brain_response.transpose())
        else:
            for exp,atlas in self.atlas:
                atlas_list = [x for x in layer_weights
                          if x.attrs['divider'][0] == exp and x.attrs['divider'][1] == atlas]
                if (not atlas_list) == False:
                    assert(len(atlas_list)==1)
                    atlas_weight=atlas_list[0]
                    atlas_activation_assembly = xr.DataArray(model_activation_flat, dims=('presentation', 'neuroid')
                                                         , coords={'neuroid': atlas_weight.coords['neuroid']})

                    brain_response = xr.dot(atlas_weight, atlas_activation_assembly, dims=['neuroid'])+atlas_weight.attrs['intercept']
                    brain_response.attrs['divider']=atlas_weight.attrs['divider']
                    brain_response_split.append(brain_response.transpose())

        return brain_response_split

    def extract_representation(self,model,layer_id):
        model_activation_set=[]
        model_impl = model_pool[model]
        layers = model_layers[model]
        candidate=FixedLayer(model_impl, layers[layer_id], prerun=layers if layer_id == 0 else None)
        #candidate = FixedLayer(test1, layers[layer], prerun=None)
        for stim_id, stim in enumerate(self.stimuli_set):
            model_activations = read_words_eh(candidate, stim, copy_columns=['stimulus_id'],average_sentence=False)#
            if self.average_sentence == 'True':
                model_activations = self.get_mean_activations(model_activations)
            elif self.average_sentence == 'False':
                model_activations = self.get_last_word_activations(model_activations)
            elif self.average_sentence == 'None':
                model_activations = self.get_all_activations(model_activations)
            model_activation_set.append(model_activations)

        model_activation_flat = [item for sublist in model_activation_set for item in sublist]
        return model_activation_flat

    def __call__(self, *args, **kwargs):
        model_grp_activations=[]
        for idx, model_id in enumerate(self.model_spec):

            if self.extract_type=='activation':
                model_activation_name = f"{self.dataset}_{self.model_spec[idx]}_layer_{self.layer_spec[idx]}_{self.extract_name}_ave_{self.average_sentence}.pkl"
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
                brain_resp_name = f"{self.dataset}_{self.model_spec[idx]}_layer_{self.layer_spec[idx]}_{self.extract_name}_{self.extract_benchmark}_ave_{self.average_sentence}.pkl"
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

class model_extractor:
    def __init__(self,dataset=None,datafile=None,model_spec=None,extract_name='activation',extract_type='activation',atlas=None,average_sentence='False'):
        self.dataset=dataset
        self.datafile=datafile
        self.model_spec=model_spec
        self.extract_type=extract_type
        self.extract_name=extract_name
        self.atlas=atlas
        self.average_sentence=average_sentence
        self.extractor=extractor(datafile=self.datafile,dataset=self.dataset,extract_name=self.extract_name,extract_type=self.extract_type,average_sentence=self.average_sentence)

    # delegations from extractor
    def load_dataset(self):
        self.extractor.load_dataset()

    def __call__(self,overwrite=False, *args, **kwargs):
        # get layers for model
        model_impl = model_pool[self.model_spec]
        layers=model_layers[self.model_spec]

        for i, layer in enumerate(tqdm(layers, desc='layers')):
            model_activation_name = f"{self.dataset}_{self.model_spec}_layer_{i}_{self.extract_name}_ave_{self.average_sentence}.pkl"
            print(f"\nextracting network activations for {self.model_spec}\n")
            if os.path.exists(os.path.join(SAVE_DIR, model_activation_name)):
                print(f"\n{model_activation_name} already exists, skipping...\n")
                pass
            else:
                print(f"\n{model_activation_name} doesn't exists, creating...\n")
                #model_activation = self.extractor.extract_representation(self.model_spec, i)
                model_activation_set = []
                candidate = FixedLayer(model_impl, layer, prerun=layers if i==0 else None)
                for stim_id, stim in enumerate(self.extractor.stimuli_set):
                    model_activations = read_words_eh(candidate, stim, copy_columns=['stimulus_id'],average_sentence=False,overwrite=overwrite)  #
                    if self.average_sentence=='True':
                        model_activations = self.extractor.get_mean_activations(model_activations)
                    elif self.average_sentence=='False':
                        model_activations = self.extractor.get_last_word_activations(model_activations)
                    elif self.average_sentence=='None':
                        model_activations = self.extractor.get_all_activations(model_activations)
                    model_activation_set.append(model_activations)


                model_activation_flat = [item for sublist in model_activation_set for item in sublist]

                #model_activation_flat = [item for sublist in model_activation_set for item in sublist]
                save_obj(model_activation_flat, os.path.join(SAVE_DIR, model_activation_name))

class model_extractor_parallel:
    def __init__(self, dataset=None, datafile=None, model_spec=None, extract_name='activation',
                 extract_type='activation', atlas=None, average_sentence='False',total_runs=0):
        self.dataset = dataset
        self.datafile = datafile
        self.model_spec = model_spec
        self.extract_type = extract_type
        self.extract_name = extract_name
        self.atlas = atlas
        self.average_sentence = average_sentence
        self.total_runs = total_runs
        self.extractor = extractor(datafile=self.datafile, dataset=self.dataset, extract_name=self.extract_name,
                                   extract_type=self.extract_type, average_sentence=self.average_sentence)

    # delegations from extractor
    def load_dataset(self):
        self.extractor.load_dataset()
        self.total_runs=len(self.extractor.stimuli_set)
    def combine_runs(self,overwrite=False):
        if type(self.model_spec)==str:
            model_set=[self.model_spec]
        else:
            model_set=self.model_spec
        for i, mdl_name in enumerate(model_set):
            model_save_path = os.path.join(SAVE_DIR, mdl_name)
            layers= model_layers[mdl_name]
            for k, layer in enumerate(tqdm(layers, desc='layers')):
                model_activation_name = f"{self.dataset}_{mdl_name}_layer_{k}_{self.extract_name}_group_*.pkl"
                new_model_activation_name=f"{self.dataset}_{self.model_spec}_layer_{k}_{self.extract_name}_ave_{self.average_sentence}.pkl"
                if os.path.exists(os.path.join(SAVE_DIR, new_model_activation_name)) and overwrite == False:
                    print(f'{os.path.join(SAVE_DIR, new_model_activation_name)} already exists\n')
                else:
                    if os.path.exists(os.path.join(SAVE_DIR, new_model_activation_name)) and overwrite == True:
                        print(f'{new_model_activation_name} already exists, but overwriting\n')
                    else:
                        print(f'{new_model_activation_name} doesnt exist, creating\n')
                    activation_files=[]
                    for file in os.listdir(model_save_path):
                        if fnmatch.fnmatch(file,model_activation_name):
                            activation_files.append(os.path.join(model_save_path, file))
                    # sort files:
                    sorted_files=[]
                    s = [re.findall('group_\d+', x) for x in activation_files]
                    s = [item for sublist in s for item in sublist]
                    file_id = [int(x.split('group_')[1]) for x in s]
                    sorted_files = [activation_files[x] for x in np.argsort(file_id)]
                    model_activation_set = []
                    if len(sorted_files)==self.total_runs:
                        for file in sorted_files:
                            model_activations = load_obj(os.path.join(SAVE_DIR, file),silent=True)
                            if self.average_sentence=='True':
                                model_activations = self.extractor.get_mean_activations(model_activations)
                            elif self.average_sentence=='False':
                                model_activations = self.extractor.get_last_word_activations(model_activations)
                            elif self.average_sentence=='None':
                                model_activations = self.extractor.get_all_activations(model_activations)
                            model_activation_set.append(model_activations)
                        # save the dataset
                        model_activation_flat = [item for sublist in model_activation_set for item in sublist]
                        save_obj(model_activation_flat, os.path.join(SAVE_DIR, new_model_activation_name))
                        print(f'saved {new_model_activation_name}\n')
                    else:
                        print(f'{self.dataset}_{mdl_name}_layer_{i}_{self.extract_name} is missing groups!\n')
        pass
    def __call__(self,group_id,overwrite=False, *args, **kwargs):
        # get layers for model
        model_impl = model_pool[self.model_spec]
        layers = model_layers[self.model_spec]
        # make a directory for groups data
        model_save_path=os.path.join(SAVE_DIR,self.model_spec)
        if os.path.exists(model_save_path):
            pass
        else:
            os.mkdir(model_save_path)
        for i, layer in enumerate(tqdm(layers, desc='layers')):
            model_activation_name = f"{self.dataset}_{self.model_spec}_layer_{i}_{self.extract_name}_group_{group_id}.pkl"
            print(f"\nextracting network activations for {self.model_spec}\n")
            if os.path.exists(os.path.join(model_save_path, model_activation_name)) and overwrite==False:
                print(f"\n{model_activation_name} already exists, skipping...\n")
                pass
            else:
                if overwrite==True:
                    print(f"\n{model_activation_name} exists but overwriting...\n")
                else:
                    print(f"\n{model_activation_name} doesn't exists, creating...\n")
                candidate = FixedLayer(model_impl, layer, prerun=layers if i == 0 else None)
                stim=self.extractor.stimuli_set[group_id]
                model_activations = read_words_eh(candidate, stim, copy_columns=['stimulus_id'], average_sentence=False,overwrite=overwrite)  #
                save_obj(model_activations, os.path.join(model_save_path, model_activation_name))