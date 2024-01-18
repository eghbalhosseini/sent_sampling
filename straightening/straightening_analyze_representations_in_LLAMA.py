import numpy as np
from sent_sampling.utils import extract_pool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from transformers import LlamaForCausalLM, LlamaTokenizer,LlamaConfig
import torch
from sent_sampling.utils.curvature_utils import compute_model_activations,compute_model_curvature
from sent_sampling.utils.data_utils import ANALYZE_DIR
torch.cuda.device_count()
from pathlib import Path
# ad arg parser
import argparse
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch,infer_auto_device_map
from accelerate.utils import get_balanced_memory
import pickle
from tqdm import tqdm
from glob import glob
from scipy.stats import spearmanr
num_devices = torch.cuda.device_count()
if __name__ == '__main__':
    modelname='65B'
    masked = False
    dataset = 'ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # get data
    ext_obj = extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    sentences = [x['text'] for x in ext_obj.data_]
    model_activation_path=Path(ANALYZE_DIR,'straightening',f'LLAMA_2_{modelname}/' )
    activation_files=glob(Path(model_activation_path.__str__(),f'LLAMA_2_{modelname}_activations_{dataset}_layer_*').__str__())
    # sort the files based on the layer number which is the last number in the file name
    activation_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # ectract the layer number
    sentence_numbers=[int(x.split('_')[-1].split('.')[0]) for x in activation_files]
    # assert that the are sorted
    assert sentence_numbers==sorted(sentence_numbers)
    # load the activations
    all_layers_activations=[]
    for i in tqdm(activation_files):
        all_layers_activations.append(torch.load(i))

    # define the lowest layer curvature
    low_curve_layer=13
    xx=[x[:low_curve_layer+1,:,:] for x in all_layers_activations]
    # compute the curvature
    curvature_drop_dict = compute_model_curvature(xx)
    curve_drop=curvature_drop_dict['curve']
    # find the nan values
    good_sent_ids=np.where(~np.isnan(curve_drop).any(axis=0))[0]
    good_sent_ids
    # drop nan values
    curve_drop=curve_drop[:,good_sent_ids]
    good_sentences=[sentences[x] for x in good_sent_ids]
    # compute the rank correlation between each row and the last row in curve_drop
    drop_corr=[]
    for i in range(curve_drop.shape[0]-1):
        drop_corr.append(spearmanr(curve_drop[i,:],curve_drop[-1,:])[0])

    # to create a random baseline, shuffle the first row and compute the rank correlation
    drop_corr_random=[]
    for i in range(500):
        drop_corr_random.append(spearmanr(curve_drop[0,:],np.random.permutation(curve_drop[-1,:]))[0])
    drop_corr_random=np.array(drop_corr_random)
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio = 8 / 11
    ax = plt.axes((.1, .6, .25, .35 * pap_ratio))

    ax.scatter(.2 * np.random.normal(size=(np.asarray(drop_corr_random).shape)), drop_corr_random, color=(.6, .6, .6),
               s=2, alpha=.3)
    ax.scatter(0, drop_corr_random.mean(), color=(0, 0, 0), s=50)
    ax.scatter(np.arange(len(drop_corr)), drop_corr, color='r', s=50,label=f'actual', edgecolor='k')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.set_xlim((-1, len(drop_corr)+1))
    ax.set_ylim((-.25, 1.1))
    # plot a line at 0
    ax.plot([-1, len(drop_corr)+1], [0, 0], 'k--', linewidth=1)
    ax.set_ylabel('curvature rank correlation', fontsize=8)
    fig.show()
    # find index of 10 sentences with lowest curvature at the last layer
    low_curv_sent_ids=np.argsort(curve_drop[0,:])[:10]
    # find sentences with highest curvature at the last layer
    high_curv_sent_ids=np.argsort(curve_drop[0,:])[-10:]
    # print the sentences
    for i in low_curv_sent_ids:
        print(good_sentences[i])

    for i in high_curv_sent_ids:
        print(good_sentences[i])

    tok_length=[x.shape[1] for x in all_layers_activations]
    long_sentences=np.argwhere(np.asarray(tok_length)>14)
    curve_drop=curvature_drop_dict['curve']
    # find the overlap of long sentences and good_sent_ids

    layer_act_for_pca=[all_layers_activations[int(x)][:low_curve_layer+1,:14,:] for x in long_sentences]
    curv_drop_for_pca=curve_drop[:,long_sentences]
    curvature_index_=np.argsort(curv_drop_for_pca[-1,:])
    # first look at the last layer
    layer_act=torch.concat([layer_act_for_pca[x][-2,:1,:] for x in curvature_index_[-100:]],dim=0).to('cuda').float()
    [U,S,V]=torch.pca_lowrank(layer_act,center=True)
    # compute the variace explained
    var_explained=torch.cumsum(S**2,dim=0)/torch.sum(S**2)



