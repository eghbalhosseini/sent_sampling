import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
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
from sent_sampling.utils.curvature_utils import compute_model_activations, compute_model_curvature
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM,AutoTokenizer
import scipy as sp
import matplotlib
import copy
from minicons import scorer
import pickle
import seaborn as sns
# add a set of argument for modelname, dataset, ablation_type
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--modelname', type=str, default='gpt2-xl',
                    help='modelname')
# skip dataset for now

parser.add_argument('--ablation_type', type=str, default='Attn_key',
                    help='ablation_type')
parser.add_argument('--layer_to_ablate', type=int, default=45,
                    help='layer_to_ablate')
# skip k_size for now

# parse the arguments
args = parser.parse_args()
if __name__ == '__main__':
    #%%
    # get arguments
    modelname=args.modelname
    ablation_type=args.ablation_type
    layer_to_ablate=args.layer_to_ablate


    #modelname= 'gpt2-xl'
    masked = False
    #dataset = 'ud_sentencez_token_filter_v3_textNoPeriod'
    #ablation_type='Attn_key'
    # emcpty cuda cache
    torch.cuda.empty_cache()
    # get data
    save_sentence_path=Path('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/sent_sampling/analysis/straightening/generation/sentences_ud_sentencez_token_filter_v3_textNoPeriod_cntx_3_cont_7.pkl')
    if save_sentence_path.exists():
        with open(save_sentence_path.__str__(),'rb') as f:
            sentences_=pickle.load(f)
    # random select 1000 sentences
    np.random.seed(0)
    k_size=2000
    sentences = np.random.choice(sentences_, k_size, replace=False)
    nonablated_model = AutoModelForCausalLM.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    # define an ablated model
    ablated_config = copy.deepcopy(nonablated_model.config)
    # Ablate MLP layers in the first transformer layer
    ablated_config.architectures = ["GPT2LMHeadModel-attn_ablated"]
    ablated_model = AutoModelForCausalLM.from_config(ablated_config)

    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
    mincons_model = scorer.IncrementalLMScorer(modelname, device='cuda')
    #layer_to_ablate=45
    ablated_curvatures=[]
    ablated_model_surprisal=[]
    ablated_model_cond_p=[]
    ablated_state_dict=copy.deepcopy(nonablated_model.state_dict())
    nonablated_state_dict=copy.deepcopy(nonablated_model.state_dict())
    for name, param_src in ablated_state_dict.items():
        if f"h.{layer_to_ablate}.attn.c_attn" in name:
            # print the name of the layer
            print(f'{name} is ablated')
            # Modify parameters for the specified layer
            param_dst = ablated_model.state_dict()[name]
            param_dst.copy_(param_src.data)
            # create an random matrix with shape of param_dst
            #random_matrix = torch.randn_like(param_dst)
            if 'weight' in name:
                dim_2=ablated_config.n_embd/ablated_config.n_head
                eye_matrix=torch.eye(param_dst.shape[0],int(ablated_config.n_head*dim_2),device=param_dst.device)
                # repeat eye_matrix 3 times
                # replace the beging part of param_dst with eye_matrix
                if ablation_type=='Attn_key':
                    weight_to_replace = copy.deepcopy(param_dst[:, 0: int(ablated_config.n_head * dim_2)])
                    new_weight=torch.cat([eye_matrix, param_dst[:,int(ablated_config.n_head*dim_2):]],dim=1)
                elif ablation_type=='Attn_all':
                #new_weight=torch.cat([eye_matrix, param_dst[:,int(ablated_config.n_head*dim_2):]],dim=1)
                # repeat eye_matrix 3 times along y axis
                    new_weight=eye_matrix.repeat(1,3).to(param_dst.device)
                #new_weight=torch.concat([eye_matrix, param_dst[:,int(config.n_head*dim_2):]],dim=1)
                # make sure new_weight has the same shape as param_dst
                assert new_weight.shape==param_dst.shape
            else:
                #create an identity matrix with shape of param_dst
                new_weight=torch.ones_like(param_dst,device=param_dst.device)
            # repalce the param_dst with random matrix
            param_dst.copy_(new_weight)
            # save the ablated state dict
            ablated_state_dict[name] = param_dst
        else:
            ablated_state_dict[name] = param_src.data
        # add nonablated state dict to the ablated state dict
    #nonablated_state_dict[name] = param_src.data
    # print that you're doing ablation on the layer
    print(f'ablation on layer {layer_to_ablate}')
    # compute the forbenous norm of weight_to_replace and Identity matrix
    print(torch.norm(weight_to_replace - eye_matrix, p='fro'), torch.norm(weight_to_replace, p='fro'))


    # repalce the weights in minicons_model
    mincons_model.model.load_state_dict(ablated_state_dict, strict=True)
    model_cond_p = []
    model_surp = []
    for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        cond_p = mincons_model.compute_stats(mincons_model.prepare_text(sentence))
        modl_surp = mincons_model.token_score(sentence, surprisal=True)
        model_cond_p.append(cond_p)
        model_surp.append(modl_surp)

    ablated_model.load_state_dict(ablated_state_dict, strict=False)
    ablated_model.cuda()
    all_layers_unt = compute_model_activations(ablated_model, indexed_tokens,device='cuda')
    curvature_dict_ablated = compute_model_curvature(all_layers_unt)

    ablated_curvatures=curvature_dict_ablated
    ablated_model_surprisal=model_surp
    ablated_model_cond_p=model_cond_p
    mincons_model.model.load_state_dict(nonablated_state_dict, strict=True)
    model_cond_p = []
    model_surp = []

    for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        cond_p = mincons_model.compute_stats(mincons_model.prepare_text(sentence))
        modl_surp = mincons_model.token_score(sentence, surprisal=True)
        model_cond_p.append(cond_p)
        model_surp.append(modl_surp)

    nonablated_model.to('cuda')
    all_layers_nonablated=compute_model_activations(nonablated_model,indexed_tokens,device='cuda')
    curvature_dict_nonablated=compute_model_curvature(all_layers_nonablated)
    surprisal_nonablated=model_surp
    cond_p_nonablated=model_cond_p

    model_surp_avg_nonablated = np.asarray([np.mean([y[1] for y in x[0]][2:]) for x in surprisal_nonablated])
    model_surp_avg_ablated = np.asarray([np.mean([y[1] for y in x[0]][2:]) for x in ablated_model_surprisal])

    surp_word_nonablated = [np.asarray([y[1] for y in x[0]][2:]) for x in surprisal_nonablated]
    surp_word_ablated = [np.asarray([y[1] for y in x[0]][2:]) for x in ablated_model_surprisal]

    # compute a correaltion between surp_word_nonablated and surp_word_ablated
    corr_surp = [sp.stats.pearsonr(x, y)[0] for x, y in zip(surp_word_nonablated, surp_word_ablated)]

    fig = plt.figure(figsize=(8,11), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # plot the surprisal for ablatted and nonablated models
    ax = plt.axes((.05, .05, .4, .4*pap_ratio))
    # plot them as scatter against each other
    x = model_surp_avg_nonablated
    y = model_surp_avg_ablated
    sns.scatterplot(x=x, y=y, s=15, color=".5", ax=ax)

    # set x and y axis to be equal
    min_=.8*min(model_surp_avg_nonablated.min(),model_surp_avg_ablated.min())
    max_=1.02*max(model_surp_avg_nonablated.max(),model_surp_avg_ablated.max())
    ax.set_xlim((min_,max_))
    ax.set_ylim((min_,max_))
    # plot unity line
    ax.plot([min_,max_],[min_,max_],color=(0,0,0),linewidth=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = plt.axes((.05, .45, .4, .4*pap_ratio))
    x=model_surp_avg_nonablated
    y=corr_surp
    sns.scatterplot(x=x, y=y, s=15, color=".5",ax=ax)
    bins = np.linspace(x.min(), x.max(), 13)
    edges = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    indices = np.digitize(x, bins)
    y_vals = [np.mean([y[ii] for ii in np.where(indices == i)[0]]) for i in range(1, len(bins))]
    # ax = plt.axes((.1, .85, .5, .15 * pap_ratio))
    ax.plot(edges, y_vals, color=(0, 0, 0), linewidth=2, marker='o', markersize=5)

    ax.set_xlim((min_, max_))
    ax.set_ylim((.7,1.))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    fig.savefig(os.path.join(ANALYZE_DIR, f'surprisal_{modelname}_layer_{layer_to_ablate}_{ablation_type}_ablated_long_sent_k_{k_size}.pdf'), transparent=True)
    fig.savefig(os.path.join(ANALYZE_DIR, f'surprisal_{modelname}_layer_{layer_to_ablate}_{ablation_type}_ablated_long_sent_k_{k_size}.png'), transparent=True)

