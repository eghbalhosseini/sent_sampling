import itertools
from utils.data_utils import SENTENCE_CONFIG
import copy
from utils.extract_utils import extractor
import neural_nlp
from neural_nlp.models import model_pool, model_layers
from itertools import count
# add the layerwise model definition

model_grps_config = [

                     dict(grp_id='best_performing_pereira',description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('bert-large-uncased-whole-word-masking','encoder.layer.11.output'),
                                                            ('xlm-mlm-en-2048','encoder.layer_norm2.11'),
                                                            ('gpt2-xl','encoder.h.43'),
                                                            ('albert-xxlarge-v2','encoder.albert_layer_groups.4'),
                                                            ('ctrl','h.46')),layer_by_name=True),
                    dict(grp_id='best_performing_pereira_1',description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('roberta-base','encoder.layer.1'),
                                           #('transfo-xl-wt103','encoder.layers.13'),
                                           ('xlnet-large-cased','encoder.layer.23'),
                                           #('t5-3b', 'encoder.block.18'),
                                           ('bert-large-uncased-whole-word-masking','encoder.layer.11.output'),
                                           ('xlm-mlm-en-2048','encoder.layer_norm2.11'),
                                           ('gpt2-xl','encoder.h.43'),
                                           ('albert-xxlarge-v2','encoder.albert_layer_groups.4'),
                                           ('ctrl','h.46')),layer_by_name=True),
                    dict(grp_id='best_performing_pereira_2',description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('roberta-base','encoder.layer.1'),
                                           #('transfo-xl-wt103','encoder.layers.13'),
                                           ('xlnet-large-cased','encoder.layer.23'),
                                           #('t5-3b', 'encoder.block.18'),
                                           ('bert-large-uncased-whole-word-masking','encoder.layer.11.output'),
                                           ('xlm-mlm-en-2048','encoder.layer_norm2.11'),
                                           ('gpt2-xl','encoder.h.43'),
                                           ('albert-xxlarge-v2','encoder.albert_layer_groups.4'),
                                           ('ctrl','h.46')),layer_by_name=True),
                     dict(grp_id='best_performing_pereira_3', description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('roberta-base', 'encoder.layer.1'),
                                           # ('transfo-xl-wt103','encoder.layers.13'),
                                           ('xlnet-large-cased', 'encoder.layer.23'),
                                           # ('t5-3b', 'encoder.block.18'),
                                           ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                                           ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                                           ('gpt2-xl', 'encoder.h.43'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                                           ('ctrl', 'h.46')), layer_by_name=True),
                     dict(grp_id='best_performing_pereira_4', description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('roberta-base', 'encoder.layer.1'),
                                           # ('transfo-xl-wt103','encoder.layers.13'),
                                           ('xlnet-large-cased', 'encoder.layer.23'),
                                           # ('t5-3b', 'encoder.block.18'),
                                           ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                                           ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                                           ('gpt2-xl', 'encoder.h.43'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                                           ('ctrl', 'h.46')), layer_by_name=True),
                    dict(grp_id='best_performing_pereira_5', description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('roberta-base', 'encoder.layer.1'),
                                           # ('transfo-xl-wt103','encoder.layers.13'),
                                           ('xlnet-large-cased', 'encoder.layer.23'),
                                           # ('t5-3b', 'encoder.block.18'),
                                           ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                                           ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                                           ('gpt2-xl', 'encoder.h.43'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                                           ('ctrl', 'h.46')), layer_by_name=True),
]

activation_extract_config=[dict(name='activation',type='activation',benchmark='None',atlas=None,modality=None),
                           #dict(name='brain_resp',type='brain_resp',benchmark='Fedorenko2016v3-encoding-weights',atlas=None,modality='ECoG'),
                           #dict(name='brain_resp',type='brain_resp',benchmark='Pereira2018-encoding-weights',atlas=(('384sentences', 'language'),('243sentences', 'language')),modality='fMRI'),
                           #dict(name='brain_resp_Pereira_exp1',type='brain_resp',benchmark='Pereira2018-encoding-weights',atlas=(('384sentences', 'language'),),modality='fMRI'),
                           #dict(name='brain_resp_Pereira_exp2',type='brain_resp',benchmark='Pereira2018-encoding-weights',atlas=(('243sentences', 'language'),),modality='fMRI')
                           ]
# define extraction configuration
extract_configuration = []
stim_types=['wordFORM', 'textPeriod', 'textNoPeriod']
average_style=['True','False','None']
for model_grp, dataset, extract_type,stim_type, average in itertools.product(model_grps_config, SENTENCE_CONFIG,
                                                                   activation_extract_config,stim_types, average_style):
    extract_identifier = f"[group={model_grp['grp_id']}]-[dataset={dataset['name']}_{stim_type}]-[{extract_type['name']}]-[bench={extract_type['benchmark']}]-[ave={average}]"
    extract_identifier = extract_identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    model_set = tuple(x[0] for x in model_grp['grp_layer_tuple'])
    if model_grp['layer_by_name']:
        layer_name = tuple(x[1] for x in model_grp['grp_layer_tuple'])
        layer_set = tuple([model_layers[x].index(layer_name[idx]) for idx, x in enumerate(model_set)])
    else:
        layer_set = tuple(x[1] for x in model_grp['grp_layer_tuple'])
        layer_name = tuple([model_layers[x][layer_set[idx]] for idx, x in enumerate(model_set)])

    extract_configuration.append(dict(identifier=extract_identifier,model_set=model_set,
                                      layer_set=layer_set,layer_name=layer_name, dataset=dataset['name'],stim_type=stim_type,datafile=dataset['file_loc'],
                                      extract_name=extract_type['name'],extract_type=extract_type['type'],benchmark=extract_type['benchmark'],atlas=extract_type['atlas'],modality=extract_type['modality'],average=average))


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
                                  stim_type=configure['stim_type'],
                                  datafile=configure['datafile'],
                                  identifier=configure['identifier'],
                                  model_spec=configure['model_set'],
                                  layer_spec=configure['layer_set'],
                                  layer_name=configure['layer_name'],
                                  extract_name=configure['extract_name'],
                                  extract_type=configure['extract_type'],
                                  extract_benchmark=configure['benchmark'],
                                  atlas=configure['atlas'],
                                  modality=configure['modality'],
                                  average_sentence=configure['average'],)
        return extractor_param

    extract_pool[extract_identifier] = extract_instantiation
# define optimization configuration