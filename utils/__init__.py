
import itertools
from utils.data_utils import SENTENCE_CONFIG
import copy
from importlib import import_module
from utils.extract_utils import extractor
model_grps_config = [dict(grp_id= 'test_early_layer', grp_set=('gpt2', 'bert-base-uncased', 'xlm-mlm-en-2048'), layer_set=(1,1,1)),
                     dict(grp_id= 'test_brain_act', grp_set=('distilgpt2', 'gpt2'), layer_set=(1,1)),
                     dict(grp_id= 'set_1', grp_set=('bert-large-uncased', 'xlm-mlm-100-1280','gpt2-large'), layer_set=(22,14,34)),
                     dict(grp_id= 'set_2', grp_set=('xlm-mlm-100-1280',), layer_set=(14,)),
                     dict(grp_id= 'set_3', grp_set=('bert-large-uncased', 'xlm-mlm-100-1280','gpt2'), layer_set=(22,14,7)),]

activation_extract_config=[dict(type='network_act',benchmark='None'),
                           dict(type='brain_resp',benchmark='Fedorenko2016v3-encoding-weights'),
                           dict(type='brain_resp',benchmark='Fedorenko2016v3-encoding-weights_v2')]
extract_configuration = []
for model_grp, dataset, extract_type, average in itertools.product(model_grps_config, SENTENCE_CONFIG,
                                                                   activation_extract_config, [True, False]):
    extract_identifier = f"[group={model_grp['grp_id']}]-[dateset={dataset['name']}]-[{extract_type['type']}]-[bench={extract_type['benchmark']}]-[ave={average}]"
    extract_identifier = extract_identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    extract_configuration.append(dict(identifier=extract_identifier,model_set=model_grp['grp_set'],
                                      layer_set=model_grp['layer_set'], dataset=dataset['name'],datafile=dataset['file_loc'],
                                      extract_type=extract_type['type'],benchmark=extract_type['benchmark'],average=average))


extract_pool={}
# create the pool
for config in extract_configuration:
    configuration=copy.deepcopy(config)
    extract_identifier=configuration['identifier']
    def extract_instantiation(configure=frozenset(configuration.items())):
        configure = dict(configure)
        #module = import_module('utils.model_utils')
        #model=getattr(module,configure['model'])
        extractor_param=extractor(dataset=configure['dataset'],
                                  datafile=configure['datafile'],
                                  model_spec=configure['model_set'],
                                  layer_spec=configure['layer_set'],
                                  extract_type=configure['extract_type'],
                                  extract_benchmark=configure['benchmark'],
                                  average_sentence=configure['average'],)
        return extractor_param

    extract_pool[extract_identifier] = extract_instantiation