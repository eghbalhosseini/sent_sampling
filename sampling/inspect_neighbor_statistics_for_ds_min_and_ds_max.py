import os
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import getpass
if getpass.getuser() == 'eghbalhosseini':
    SAMPLING_PARENT = '/Users/eghbalhosseini/MyCodes/sent_sampling'
    SAMPLING_DATA = '/Users/eghbalhosseini/MyCodes//fmri_DNN/ds_parametric/'

elif getpass.getuser() == 'ehoseini':
    SAMPLING_PARENT = '/om2/user/ehoseini/sent_sampling'
    SAMPLING_DATA = '/om2/user/ehoseini/fmri_DNN/ds_parametric/'

from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
def euclidean_distance(point1, point2):
    return torch.sqrt(torch.sum((point1 - point2)**2))

if __name__ == '__main__':
    model_names=["roberta-base", "xlnet-large-cased", "bert-large-uncased-whole-word-masking" ,"xlm-mlm-en-2048", "albert-xxlarge-v2", "ctrl","gpt2-xl"]
    layers=[2, 24, 12, 12, 5, 47, 44]
    n_neighbors=2000
    pc_ratio = .8
    for id_l, model_name in enumerate(model_names):
        model_layer_activations=f'coca_preprocessed_all_clean_no_dup_100K_sample_1_textNoPeriod_{model_name}_layer_{layers[id_l]}_activation_ave_False.pkl'

        model_data = load_obj(Path(SAVE_DIR, model_layer_activations).__str__())
        model_act=torch.stack([torch.tensor(x[0]) for x in model_data])
        save_path = Path(ANALYZE_DIR, 'DsParametric', f'act_all_dsparametric_{model_name}.pkl')
        act_all= load_obj(save_path.__str__())
        min_loc=act_all['min_loc']
        max_loc=act_all['max_loc']
        act_ = act_all['act_all']
        act_min=torch.tensor(act_[min_loc,:])
        act_max=torch.tensor(act_[max_loc,:])
        a_list=[]
        b_list=[]
        for idx in tqdm(range(len(act_min))):
            a=[euclidean_distance(act_min[idx],x) for x in model_act]
            b=[euclidean_distance(act_max[idx],x) for x in model_act]
            a_list.append(a)
            b_list.append(b)


        a_list=torch.tensor(a_list)
        b_list=torch.tensor(b_list)
        save_obj({'min_dist': a_list, 'max_dist': b_list}, Path(ANALYZE_DIR, 'DsParametric',
                                                                f'act_min_max_dsparametric_distance_to_coca_preprocessed_all_clean_no_dup_100K_sample_{model_name}_layer_{layers[id_l]}.pkl').__str__())

        a_t=torch.sort(a_list,dim=1)

        b_t=torch.sort(b_list,dim=1)

        a_mean=a_t[0][:,:n_neighbors].mean(dim=1)
        b_mean=b_t[0][:,:n_neighbors].mean(dim=1)
        # plot a distribution of
        import matplotlib.pyplot as plt
        plt.hist(a_mean.numpy(), bins=20, alpha=0.5, label='min')
        plt.hist(b_mean.numpy(), bins=20, alpha=0.5, label='max')
        plt.legend(loc='upper right')
        plt.show()

        num_dims_for90pca_min=[]
        num_dims_for90pca_max=[]

        pca = PCA(n_components=pc_ratio)
        for k in tqdm(range(80)):
            neigh_v=model_act[a_t[1][:,:n_neighbors][k],:]
            pca.fit(neigh_v)
            num_dims_for90pca_min.append(len(pca.explained_variance_ratio_))
            neigh_v = model_act[b_t[1][:, :n_neighbors][k], :]
            pca.fit(neigh_v)
            num_dims_for90pca_max.append(len(pca.explained_variance_ratio_))


        plt.hist(num_dims_for90pca_min, bins=20, alpha=0.5, label='min')
        plt.hist(num_dims_for90pca_max, bins=20, alpha=0.5, label='max')
        plt.legend(loc='upper right')
        plt.show()
        # save the results
        save_obj({'num_dims_min': num_dims_for90pca_min, 'num_dims_max': num_dims_for90pca_max,'num_neighbors':n_neighbors,'pc_ratio':pc_ratio},
                 Path(ANALYZE_DIR, 'DsParametric',
                      f'act_min_max_dsparametric_num_dims_for90pca_num_points_{n_neighbors}_{model_name}_layer_{layers[id_l]}.pkl').__str__())





