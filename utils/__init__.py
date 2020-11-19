import itertools
from utils.data_utils import SENTENCE_CONFIG
import copy
from utils.extract_utils import extractor
from utils.optim_utils import optim
from neural_nlp.models import model_pool, model_layers
model_grps_config = [dict(grp_id= 'test_early_layer', grp_layer_tuple=(('gpt2',1),
                                                                       ('bert-base-uncased',1),
                                                                        ('xlm-mlm-en-2048',1)),layer_by_name=False),
                    dict(grp_id='set_2', grp_layer_tuple=(('roberta-base', 2),
                                                                      ('roberta-base', 3)), layer_by_name=False),
                     dict(grp_id='set_3', grp_layer_tuple=(('bert-large-uncased', 22),
                                                                      ('xlm-mlm-100-1280', 14),
                                                                      ('gpt2', 7)), layer_by_name=False),
                     dict(grp_id= 'set_4', grp_layer_tuple=(('bert-large-uncased-whole-word-masking','encoder.layer.11.output'),
                                                            ('xlm-mlm-en-2048','encoder.layer_norm2.11'),
                                                            ('gpt2-xl','encoder.h.43'),
                                                            ('t5-3b','encoder.block.16'),
                                                            ('albert-xxlarge-v2','encoder.albert_layer_groups.4'),
                                                            ('ctrl','h.46')),layer_by_name=True)]

activation_extract_config=[dict(type='network_act',benchmark='None',atlas=None,modality=None),
                           dict(type='brain_resp',benchmark='Fedorenko2016v3-encoding-weights',atlas=None,modality='ECoG'),
                           dict(type='brain_resp',benchmark='Pereira2018-encoding-weights',atlas=(('384sentences', 'language'),('243sentences', 'language')),modality='fMRI')]
# define extraction configuration
extract_configuration = []
for model_grp, dataset, extract_type, average in itertools.product(model_grps_config, SENTENCE_CONFIG,
                                                                   activation_extract_config, [True, False]):
    extract_identifier = f"[group={model_grp['grp_id']}]-[dataset={dataset['name']}]-[{extract_type['type']}]-[bench={extract_type['benchmark']}]-[ave={average}]"
    extract_identifier = extract_identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    model_set = tuple(x[0] for x in model_grp['grp_layer_tuple'])
    if model_grp['layer_by_name']:
        layer_name = tuple(x[1] for x in model_grp['grp_layer_tuple'])
        layer_set = tuple([model_layers[x].index(layer_name[idx]) for idx, x in enumerate(model_set)])
    else:
        layer_set = tuple(x[1] for x in model_grp['grp_layer_tuple'])
        layer_name = tuple([model_layers[x][layer_set[idx]] for idx, x in enumerate(model_set)])

    extract_configuration.append(dict(identifier=extract_identifier,model_set=model_set,
                                      layer_set=layer_set,layer_name=layer_name, dataset=dataset['name'],datafile=dataset['file_loc'],
                                      extract_type=extract_type['type'],benchmark=extract_type['benchmark'],atlas=extract_type['atlas'],modality=extract_type['modality'],average=average))


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
                                  layer_name=configure['layer_name'],
                                  extract_type=configure['extract_type'],
                                  extract_benchmark=configure['benchmark'],
                                  atlas=configure['atlas'],
                                  modality=configure['modality'],
                                  average_sentence=configure['average'],)
        return extractor_param

    extract_pool[extract_identifier] = extract_instantiation



# define optimization configuration