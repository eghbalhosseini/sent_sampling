import itertools
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
import copy
from sent_sampling.utils.extract_utils import extractor
import neural_nlp
from neural_nlp.models import model_pool, model_layers
from itertools import count
# add the layerwise model definition

def make_shorthand(extractor_id, optimizer_id):
    # create a short hand version for the file name
    # get group= from extract_id
    group = extractor_id.split('-')[0].split('=')[1]
    # get dataset= from extract_id
    dataset = extractor_id.split('-')[1].split('=')[1]
    # get activation from extract_id
    activation = extractor_id.split('-')[2]
    # get bench from extract_id
    bench = extractor_id.split('-')[3].split('=')[1]
    # get ave from extract_id
    ave = extractor_id.split('-')[4].split('=')[1]
    # get coord from optim_id
    coord = optimizer_id.split('-')[0]
    # make an auxilary name by removing coord from optimizer_id
    aux_name = '-'.join(optimizer_id.split('-')[1:])
    # get obj from aux_name by finding -n_iter and taking the first part
    obj = aux_name.split('-n_iter')[0].split('=')[1]
    # get n_iter from aux_name
    n_iter = aux_name.split('-n_iter')[1].split('-')[0].split('=')[1]
    # get n_samples from aux_name
    n_samples = aux_name.split('-n_samples')[1].split('-')[0].split('=')[1]
    # get n_init from aux_name
    n_init = aux_name.split('-n_init')[1].split('-')[0].split('=')[1]
    # get low_dim from aux_name
    low_dim = aux_name.split('-low_dim')[1].split('-')[0].split('=')[1]
    # get pca_var from aux_name
    pca_var = aux_name.split('-pca_var')[1].split('-')[0].split('=')[1]
    # get pca_type from aux_name
    pca_type = aux_name.split('-pca_type')[1].split('-')[0].split('=')[1]
    # get run_gpu from aux_name
    run_gpu = aux_name.split('-run_gpu')[1].split('-')[0].split('=')[1]
    # create a short hand name from the above
    optim_short_hand = f"[{coord}]-[O={obj}]-[Nit={n_iter}]-[Ns={n_samples}]-[Nin={n_init}]-[LD={low_dim}]-[V={pca_var}]-[T={pca_type}]-[GPU={run_gpu}]"
    optim_short_hand = optim_short_hand.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    # extract shorthand
    extract_short_hand = f"[G={group}]-[D={dataset}]-[{activation}]-[B={bench}]-[AVE={ave}]"
    extract_short_hand = extract_short_hand.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    return (extract_short_hand, optim_short_hand)

# modl_name='mistral-caprica-gpt2-small-x81-ckpnt-0'
# layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
# mistral_chkpnt_0_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)
#
# modl_name='mistral-caprica-gpt2-small-x81-ckpnt-40'
# layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
# mistral_chkpnt_40_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)
#
# modl_name='mistral-caprica-gpt2-small-x81-ckpnt-400'
# layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
# mistral_chkpnt_400_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)
#
# modl_name='mistral-caprica-gpt2-small-x81-ckpnt-4000'
# layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
# mistral_chkpnt_4000_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)
#
# modl_name='mistral-caprica-gpt2-small-x81-ckpnt-40000'
# layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
# mistral_chkpnt_40000_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)
#
# modl_name='mistral-caprica-gpt2-small-x81-ckpnt-400000'
# layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
# mistral_chkpnt_400000_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)
#
#

modl_name='distilgpt2'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
distilgpt2_config=dict(grp_id= f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)



modl_name='gpt2'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-medium'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_medium_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-large'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_large_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-xl'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_xl_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-untrained'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_untrained_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='gpt2-xl-untrained'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
gpt2_xl_untrained_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

modl_name='lm_1b'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
lm_1b_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)

# make one for xlnet-large-cased
modl_name='xlnet-large-cased'
layer_tuple=tuple([(modl_name,x) for x in model_layers[modl_name]])
xlnet_large_cased_grp_config=dict(grp_id=f"{modl_name}_layers",grp_layer_tuple=tuple([layer_tuple[id] for id in range(0,len(layer_tuple))]),layer_by_name=True)


model_grps_config = [
    # mistral_chkpnt_0_config,
    # mistral_chkpnt_40_config,
    # mistral_chkpnt_400_config,
    # mistral_chkpnt_4000_config,
    # mistral_chkpnt_40000_config,
    # mistral_chkpnt_400000_config,
    # distilgpt2_config,
    gpt2_grp_config,
    gpt2_medium_config,
    gpt2_large_config,
    gpt2_xl_grp_config,
    gpt2_untrained_grp_config,
    gpt2_xl_untrained_grp_config,
    lm_1b_grp_config,
    xlnet_large_cased_grp_config,
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