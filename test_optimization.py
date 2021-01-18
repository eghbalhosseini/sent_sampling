import importlib
#import utils
#importlib.reload(utils)

from utils import extract_pool
#name='group=test_early_layer-dateset=ud_sentences_filter-brain_resp-ave=False'
#name= 'group=test_brain_act-dateset=ud_sentences_filter_sample-brain_resp-bench=Fedorenko2016v3-encoding-weights-ave=False'
#name='group=set_2-dataset=ud_sentences_token_filter_v3-brain_resp-bench=Fedorenko2016v3-encoding-weights-ave=False'
#name='group=set_2-dataset=ud_sentences_token_filter_v3-activation-bench=None-ave=False'
#name='group=set_2-dataset=ud_sentences_token_filter_v3_sample-brain_resp-bench=Pereira2018-encoding-weights-ave=False'
#name='group=test_pereira-dataset=ud_sentences_filter_v3_sample-brain_resp-bench=Pereira2018-encoding-weights-ave=False'
name='group=best_performing_pereira_5-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False'
#name='group=set_3-dataset=ud_sentences_filter-activation-bench=None-ave=False'
#name='group=set_1-dateset=ud_sentences_token_filter_sample-activation-bench=None-ave=True'
#name_brain='group=set_1-dateset=ud_sentences_filter_sample-brain_resp-bench=Fedorenko2016v3-encoding-weights_v2-ave=False'


import utils.optim_utils
importlib.reload(utils.optim_utils)
from utils.optim_utils import optim, optim_pool

test=extract_pool[name]()

test.load_dataset()
test()

optim_name='coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=20-n_init=1'
optim_obj=optim_pool[optim_name]()

optim_obj.load_extractor(test)
S_opt_d, DS_opt_d=optim_obj()
