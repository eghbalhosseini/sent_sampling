import importlib
#import utils
#importlib.reload(utils)
from utils import extract_pool
#name='group=test_early_layer-dateset=ud_sentences_filter-brain_resp-ave=False'
#name= 'group=test_brain_act-dateset=ud_sentences_filter_sample-brain_resp-bench=Fedorenko2016v3-encoding-weights-ave=False'
name='group=set_1-dateset=ud_sentences_filter_sample-network_act-bench=None-ave=False'
#name='group=set_1-dateset=ud_sentences_filter_sample-network_act-bench=None-ave=True'
#name_brain='group=set_1-dateset=ud_sentences_filter_sample-brain_resp-bench=Fedorenko2016v3-encoding-weights_v2-ave=False'

import utils.optim_utils
importlib.reload(utils.optim_utils)
from utils.optim_utils import optim

test=extract_pool[name]()

test.load_dataset()
test()

optim_obj=optim(extractor_obj=test,N_s=50,n_iter=200,n_init=2)
S_opt_d, DS_opt_d=optim_obj(test)