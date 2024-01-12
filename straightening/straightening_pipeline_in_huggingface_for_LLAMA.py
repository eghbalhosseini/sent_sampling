import numpy as np
from sent_sampling.utils import extract_pool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from transformers import LlamaForCausalLM, LlamaTokenizer,LlamaConfig
import torch
import os
from sent_sampling.utils.curvature_utils import compute_model_activations,compute_model_curvature
from sent_sampling.utils.data_utils import ANALYZE_DIR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
from pathlib import Path
# ad arg parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='LLAMA/13B')


if __name__ == '__main__':
    #%%
    args = parser.parse_args()
    modelname=args.modelname
    weight_path=f'/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA/{modelname}'
    tokenizer = LlamaTokenizer.from_pretrained(weight_path)
    model = LlamaForCausalLM.from_pretrained(weight_path)
    model.to(device)
    masked=False
    dataset='ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences = [x['text'] for x in ext_obj.data_]
    del ext_obj
    # tokenize sentences
    tokenized_text = [tokenizer.tokenize(x) for x in sentences]
    # get ids
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

    # if model continuation dict doesnt exsist create it
    all_layers=compute_model_activations(model,indexed_tokens,device)
    curvature_dict_true=compute_model_curvature(all_layers)
    # create a dict for saving
    save_dict={}
    save_dict['curvature_dict_true']=curvature_dict_true
    save_dict['modelname']=modelname
    activations = all_layers['activations']
    save_dict['activations']=activations
    # save the dict
    save_path=Path(ANALYZE_DIR,'straightening', f'{modelname}_curvature_{dataset}.pt')
    #%%
    curve_ = curvature_dict_true['curve']
    fig = plt.figure(figsize=(5.5, 9), dpi=200, frameon=False)
    pap_ratio = 5.5 / 9
    matplotlib.rcParams['font.size'] = 6
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # create colors for lines based on the number of models
    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 3
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    ax = plt.axes((.1, .55, .55, .25 * pap_ratio))
    kk = 0


    num_colors = curve_.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    for i, curv in enumerate(curve_):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=5, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=(np.nanstd(curv) * 180 / np.pi)/np.sqrt(curv.shape[0]), linewidth=0, elinewidth=1,
                color=line_cols[i, :], zorder=0, alpha=1)
           # plot a line for the average

    ax.plot(np.arange(curve_.shape[0]), np.nanmean(curve_, axis=1) * 180 / np.pi, color=(0, 0, 0),
            linewidth=2,
            zorder=1)
    # plot sem around the average as fill_between


    ax.fill_between(np.arange(curve_.shape[0]),
                    (np.nanmean(curve_, axis=1) - np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    (np.nanmean(curve_, axis=1) + np.nanstd(curve_, axis=1)/np.sqrt(curve_.shape[1])) * 180 / np.pi,
                    color=(0, 0, 0), alpha=.2, zorder=1)

    ax.fill_between(np.arange(curve_.shape[0]),
                    (np.nanmean(curve_, axis=1) - np.nanstd(curve_, axis=1)) * 180 / np.pi,
                    (np.nanmean(curve_, axis=1) + np.nanstd(curve_, axis=1)) * 180 / np.pi,
                    color=(0, 0, 0), alpha=.2, zorder=1)
    ax.set_ylim([102.5, 125])
    fig.show()
    save_path=Path(ANALYZE_DIR,'straightening', f'{modelname}_curvature_{dataset}.pdf')
    # make sure it paernt dir exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.__str__(), transparent=True)
