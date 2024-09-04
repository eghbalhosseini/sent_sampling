import numpy as np
import pandas as pd

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
parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='LLAMA_13B')
args = parser.parse_args()
# get the number of available gpus
from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':
    #%%
    modelname=str(args.modelname)
    #modelname='LLAMA_7B'
    #'sftp://ehoseini:@evnode.mit.edu/%2Fnese/mit/group/evlab/u/ehoseini/MyData/1000_samples_charades/charades_1000_test_sentences.csv'
    weight_path=f'/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA/{modelname}'
    config_path=f'/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA/{modelname}/config.json'
    dataset='charades_1000_test_sentences'

    tokenizer = LlamaTokenizer.from_pretrained(weight_path)
    modelConfig=LlamaConfig.from_json_file(config_path)

    # get the name of saved dictionary
    curvature_dict_path=Path(ANALYZE_DIR, 'straightening', f'{modelname}', f'{modelname}_{dataset}_curvature_dict_.pkl')
    if curvature_dict_path.exists():
        with open(curvature_dict_path.__str__(),'rb') as f:
            curvature_dict_all = pickle.load(f)
    else:

        with init_empty_weights():
            model = LlamaForCausalLM(modelConfig)

        #device_map = infer_auto_device_map(model,no_split_module_classes=['LlamaDecoderLayer'],max_memory={0: "48GiB", 1: "48GiB" })
        #device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'],max_memory={0: "44GiB", 1: "44GiB",2: "44GiB",3: "44GiB" })
        device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'],
                                           max_memory={0: "72GiB", 1: "72GiB", 2: "72GiB", 3: "72GiB"})
        # print device map
        print(device_map)
        model = load_checkpoint_and_dispatch(model, checkpoint=weight_path, device_map=device_map)
        for i in model.named_parameters():
            print(f"{i[0]} -> {i[1].device}")
        # test model
        # reshape it
        masked=False

        # get data


        # get sentences from ext_obj
        sentence_set = pd.read_csv('/nese/mit/group/evlab/u/ehoseini/MyData/1000_samples_charades/charades_1000_test_sentences.csv')
        sentences=sentence_set['sentence'].tolist()
        #[x['text'] for x in ext_obj.data_]
        #del ext_obj
        # tokenize sentences
        tokenized_text = [tokenizer.tokenize(x) for x in sentences]
        # get ids
        indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]


        batch_size = 16
        num_batches = int(np.ceil(len(indexed_tokens) / batch_size))
        curvature_dict_all = []
        all_batch=[]
        for i in tqdm(range(num_batches)):
            batch = indexed_tokens[i * batch_size:(i + 1) * batch_size]
            batch_d = compute_model_activations(model, batch, model.device)
            all_batch.append(batch_d)
        # flatten the all_batch
        all_batch = [item for sublist in all_batch for item in sublist]
        curvature_dict_all = compute_model_curvature(all_batch)

        # make sure parent dir exists
        curvature_dict_path.parent.mkdir(parents=True, exist_ok=True)
        with open(curvature_dict_path.__str__(), 'wb') as f:
            pickle.dump(curvature_dict_all, f)

    # load curvature_dict_all from pickle

    #%%
    #curve_ = np.concatenate([x['curve'] for x in curvature_dict_all], axis=1)
    curve_ = curvature_dict_all['curve']
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

    curve_change = (curve_[1:, :] - curve_[1, :])
    ax = plt.axes((.1, .1, .55, .25 * pap_ratio))
    kk = 0

    num_colors = curve_change.shape[0] + 2
    color_fact = num_colors + 10
    h0 = cm.get_cmap('inferno', color_fact)
    line_cols = (h0(np.arange(color_fact) / color_fact))
    line_cols = line_cols[2:, :]
    for i, curv in enumerate(curve_change):
        ax.scatter(i, np.nanmean(curv) * 180 / np.pi, s=5, color=line_cols[i, :], zorder=2, edgecolor=(0, 0, 0),
                   linewidth=.5, alpha=1)
        ax.errorbar(i, np.nanmean(curv) * 180 / np.pi, yerr=(np.nanstd(curv) * 180 / np.pi) / np.sqrt(curv.shape[0]),
                    linewidth=0, elinewidth=1,
                    color=line_cols[i, :], zorder=0, alpha=1)
        # plot a line for the average

    ax.plot(np.arange(curve_change.shape[0]), np.nanmean(curve_change, axis=1) * 180 / np.pi, color=(0, 0, 0),
            linewidth=2,
            zorder=1)
    # plot sem around the average as fill_between

    ax.fill_between(np.arange(curve_change.shape[0]),
                    (np.nanmean(curve_change, axis=1) - np.nanstd(curve_change, axis=1) / np.sqrt(
                        curve_change.shape[1])) * 180 / np.pi,
                    (np.nanmean(curve_change, axis=1) + np.nanstd(curve_change, axis=1) / np.sqrt(
                        curve_change.shape[1])) * 180 / np.pi,
                    color=(0, 0, 0), alpha=.2, zorder=1)

    ax.fill_between(np.arange(curve_change.shape[0]),
                    (np.nanmean(curve_change, axis=1) - np.nanstd(curve_change, axis=1)) * 180 / np.pi,
                    (np.nanmean(curve_change, axis=1) + np.nanstd(curve_change, axis=1)) * 180 / np.pi,
                    color=(0, 0, 0), alpha=.2, zorder=1)

    ax.set_ylim([-15., 2])

    fig_save_path=Path(ANALYZE_DIR,'straightening', f'{modelname}_curvature_{dataset}.pdf')
    # make sure it paernt dir exists
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path.__str__(), transparent=True)