import itertools
from utils.data_utils import SENTENCE_CONFIG
import copy
from utils.extract_utils import extractor
import neural_nlp
from neural_nlp.models import model_pool, model_layers
from itertools import count
# add the layerwise model definition
# gpt2
modl_name='gpt2'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='lm_1b'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
lm_1b_grp_v1_config=dict(grp_id=f"{modl_name}_layer_compare_v1",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='lm_1b'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
lm_1b_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-untrained'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_untrained_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-xl-untrained'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_xl_untrained_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)


modl_name='openaigpt'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
openaigpt_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='openaigpt-untrained'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
openaigpt_untrained_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-xl'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_xl_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-medium'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_medium_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-large'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_large_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)


modl_name='roberta-base'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
roberta_base_grp_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='xlnet-large-cased'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
xlnet_large_cased_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='bert-large-uncased-whole-word-masking'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
bert_large_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='xlm-mlm-en-2048'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
xlm_mlm_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='albert-xxlarge-v2'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
albert_xxlarge_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='ctrl'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
ctrl_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='t5-large'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
t5_large_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='distilgpt2'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
distilgpt2_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)


model_grps_config = [gpt2_xl_untrained_grp_config,
    distilgpt2_config,
                    lm_1b_grp_config,
                    lm_1b_grp_v1_config,
                    t5_large_config,
                    openaigpt_grp_config,
                    gpt2_grp_config,
                    gpt2_medium_config,
                    gpt2_large_config,
                     gpt2_xl_grp_config,
                     roberta_base_grp_config,
                     xlnet_large_cased_config,
                     bert_large_config,
                     xlm_mlm_config,
                     albert_xxlarge_config,
                     ctrl_config,
                     gpt2_untrained_grp_config,
                    openaigpt_untrained_grp_config,
                     dict(grp_id='roberta-base_layer_compare', description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('roberta-base', 'encoder.layer.0'),
                                           ('roberta-base', 'encoder.layer.1'),
                                           ('roberta-base', 'encoder.layer.4'),
                                           ('roberta-base', 'encoder.layer.5'),
                                           ('roberta-base', 'encoder.layer.11')), layer_by_name=True),

                     dict(grp_id='roberta-base_layer_compare',description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('roberta-base','encoder.layer.0'),
                                           ('roberta-base','encoder.layer.1'),
                                           ('roberta-base','encoder.layer.4'),
                                           ('roberta-base','encoder.layer.5'),
                                           ('roberta-base','encoder.layer.11')),layer_by_name=True),
                     dict(grp_id='gpt2_layer_compare_v1', description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('gpt2', 'encoder.h.0'),
                                           ('gpt2', 'encoder.h.8'),
                                           ('gpt2', 'encoder.h.11')), layer_by_name=True),
                     dict(grp_id='openaigpt_layer_compare_v1', description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('openaigpt', 'encoder.h.0.ln_2'),
                                           ('openaigpt', 'encoder.h.7.ln_2'),
                                           ('openaigpt', 'encoder.h.9.ln_2'),
                                           ('openaigpt', 'encoder.h.11.ln_2')), layer_by_name=True),
                    dict(grp_id='gpt2-xl_layer_compare',description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('gpt2-xl','encoder.h.0'),
                                           ('gpt2-xl','encoder.h.20'),
                                           ('gpt2-xl','encoder.h.28'),
                                           ('gpt2-xl','encoder.h.33'),
                                           ('gpt2-xl','encoder.h.43'),
                                           ('gpt2-xl','encoder.h.47')),layer_by_name=True),
                     dict(grp_id='gpt2-xl_layer_compare_v1', description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('gpt2-xl', 'encoder.h.0'),
                                           ('gpt2-xl', 'encoder.h.25'),
                                           ('gpt2-xl', 'encoder.h.32'),
                                           ('gpt2-xl', 'encoder.h.43'),
                                           ('gpt2-xl', 'encoder.h.47')), layer_by_name=True),
                    dict(grp_id='gpt2-xl_layer_compare_v2',description='best layer/Pereira benchmark',
                          grp_layer_tuple=(('gpt2-xl','encoder.h.20'),
                                           ('gpt2-xl','encoder.h.28'),
                                           ('gpt2-xl','encoder.h.33'),
                                           ('gpt2-xl','encoder.h.43')),layer_by_name=True),
                     dict(grp_id='ctrl_layer_compare_v1', description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('ctrl', 'h.0'),
                                           ('ctrl', 'h.13'),
                                           ('ctrl', 'h.21'),
                                           ('ctrl', 'h.46'),
                                           ('ctrl', 'h.47')), layer_by_name=True),

                    dict(grp_id='bert-large-uncased-whole-word-masking_layer_compare_v1', description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('bert-large-uncased-whole-word-masking', 'encoder.layer.0.output'),
                                           ('bert-large-uncased-whole-word-masking', 'encoder.layer.5.output'),
                                           ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                                           ('bert-large-uncased-whole-word-masking', 'encoder.layer.20.output'),
                                           ('bert-large-uncased-whole-word-masking', 'encoder.layer.23.output')), layer_by_name=True),
                     dict(grp_id='albert-xxlarge-v2_layer_compare_v1',
                          description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('albert-xxlarge-v2', 'encoder.albert_layer_groups.0'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.5'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.7'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.11')),
                          layer_by_name=True),


                     dict(grp_id='roberta-base_layer_compare_v1',
                          description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('roberta-base', 'encoder.layer.0'),
                                           ('roberta-base', 'encoder.layer.1'),
                                           ('roberta-base', 'encoder.layer.3'),
                                           ('roberta-base', 'encoder.layer.5'),
                                           ('roberta-base', 'encoder.layer.11')),
                          layer_by_name=True),

                     dict(grp_id='xlm-mlm-en-2048_layer_compare_v1',
                          description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('xlm-mlm-en-2048', 'encoder.layer_norm2.0'),
                                           ('xlm-mlm-en-2048', 'encoder.layer_norm2.7'),
                                           ('xlm-mlm-en-2048', 'encoder.layer_norm2.11')),
                          layer_by_name=True),
                    dict(grp_id='xlnet-large-cased_layer_compare_v1',
                          description='best layer/Pereira benchmark/ud_corpus',
                          grp_layer_tuple=(('xlnet-large-cased', 'encoder.layer.0'),
                                           ('xlnet-large-cased', 'encoder.layer.8'),
                                           ('xlnet-large-cased', 'encoder.layer.23')),
                          layer_by_name=True),

                     dict(grp_id= 'test_early_layer', grp_layer_tuple=(('gpt2',1),
                                                                       ('bert-base-uncased',1),
                                                                        ('xlm-mlm-en-2048',1)),layer_by_name=False),
                     dict(grp_id='set_2', grp_layer_tuple=(('albert-xxlarge-v2',"encoder.albert_layer_groups.2"),
                                                        ('albert-xxlarge-v2', "encoder.albert_layer_groups.3")), layer_by_name=True),
                    dict(grp_id='test_pereira', grp_layer_tuple=(('roberta-base',2),
                                                        ('roberta-base', 3)), layer_by_name=False),
                    dict(grp_id='test_pereira_full', grp_layer_tuple=(('bert-large-uncased-whole-word-masking', 2),
                                                                  ('bert-large-uncased-whole-word-masking', 3)), layer_by_name=False),
                     dict(grp_id='set_3', grp_layer_tuple=(('bert-large-uncased', 22),
                                                                      ('xlm-mlm-100-1280', 14),
                                                                      ('gpt2', 7)), layer_by_name=False),
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
                     dict(grp_id='second_best_performing_pereira', description='second best layer/Pereira benchmark',
                          grp_layer_tuple=(('bert-large-uncased-whole-word-masking', 'encoder.layer.12.output'),
                                           ('xlm-mlm-en-2048', 'encoder.layer_norm2.6'),
                                           ('gpt2-xl', 'encoder.h.42'),
                                           ('albert-xxlarge-v2', 'encoder.albert_layer_groups.2'),
                                           ('ctrl', 'h.47')), layer_by_name=True)]

activation_extract_config=[dict(name='activation',type='activation',benchmark='None',atlas=None,modality=None),
                           dict(name='brain_resp',type='brain_resp',benchmark='Fedorenko2016v3-encoding-weights',atlas=None,modality='ECoG'),
                           dict(name='brain_resp',type='brain_resp',benchmark='Pereira2018-encoding-weights',atlas=(('384sentences', 'language'),('243sentences', 'language')),modality='fMRI'),
                           dict(name='brain_resp_Pereira_exp1',type='brain_resp',benchmark='Pereira2018-encoding-weights',atlas=(('384sentences', 'language'),),modality='fMRI'),
                           dict(name='brain_resp_Pereira_exp2',type='brain_resp',benchmark='Pereira2018-encoding-weights',atlas=(('243sentences', 'language'),),modality='fMRI')]
# define extraction configuration
extract_configuration = []
average_style=['True','False','None']
for model_grp, dataset, extract_type, average in itertools.product(model_grps_config, SENTENCE_CONFIG,
                                                                   activation_extract_config, average_style):
    extract_identifier = f"[group={model_grp['grp_id']}]-[dataset={dataset['name']}]-[{extract_type['name']}]-[bench={extract_type['benchmark']}]-[ave={average}]"
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