import os
import numpy as np
import sys
from pathlib import Path
import getpass
if getpass.getuser() == 'eghbalhosseini':
    SAMPLING_PARENT = '/Users/eghbalhosseini/MyCodes/sent_sampling'
    SAMPLING_DATA = '/Users/eghbalhosseini/MyCodes//fmri_DNN/ds_parametric/'

elif getpass.getuser() == 'ehoseini':
    SAMPLING_PARENT = '/om/user/ehoseini/sent_sampling'
    SAMPLING_DATA = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'



sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
from matplotlib.pyplot import GridSpec
import pandas as pd
from pathlib import Path
import torch
from sent_sampling.utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sent_sampling.utils.optim_utils import low_dim_project

if __name__ == '__main__':
    dataset_id='coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K'
    extract_id = f'group=best_performing_pereira_1-dataset={dataset_id}_textNoPeriod-activation-bench=None-ave=False'
    optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset(splits=20)
    ext_obj()
    optim_obj = optim_pool[optim_id]()
    optim_obj.load_extractor(ext_obj)


    n_components=[100,250,500,650]
    for n_comp in n_components:
        pca_dict = []
        for idx, act_dict in tqdm(enumerate(optim_obj.activations)):
        # backward compatibility
            act_ = torch.tensor([x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']])
            act = low_dim_project(act_, var_explained=n_comp,pca_type='sklearn')
            x=dict()
            act[0].shape
            x['act']=act[0]
            x['var_explained']=act[1]
            x['model']=act_dict['model_name']
            x['layer']=act_dict['layer']
            pca_dict.append(x)
    # save the pca_dict in Analyze dir
        pathname=Path(ANALYZE_DIR,f'{extract_id}_pca_n_comp_{n_comp}.pkl')
        save_obj(pca_dict,pathname)