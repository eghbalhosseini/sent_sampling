import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from transformers import LlamaForCausalLM, LlamaTokenizer,LlamaConfig
import torch
from sent_sampling.utils.curvature_utils import compute_model_activations,compute_model_curvature,compute_model_curvature_timescale
from sent_sampling.utils.data_utils import ANALYZE_DIR
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch,infer_auto_device_map
from torch.utils.data import DataLoader
import pickle
# Iterate through each CUDA device
def tokenize_function(examples):
    outputs = tokenizer(examples['text'])
    return outputs


if __name__ == '__main__':
    #%%

    modelname='LLAMA_7B'
    weight_path=f'/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA/{modelname}'
    config_path=f'/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA/{modelname}/config.json'
    modelConfig=LlamaConfig.from_json_file(config_path)
    tokenizer = LlamaTokenizer.from_pretrained(weight_path)
    range_low=380
    range_high=400
    with init_empty_weights():
        model = LlamaForCausalLM(modelConfig)

    device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'],
                                       max_memory={0: "72GiB"})

    wiki_data = load_dataset("wikitext",'wikitext-103-raw-v1')
    # get text for train split
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = wiki_data.map(tokenize_function, batched=True, remove_columns=["text"])
    # select subest that are between 2048 and 4096 tokens
    tokenized_datasets=tokenized_datasets['train']
    tokenized_datasets_long = tokenized_datasets.filter(lambda x: range_low <= len(x['input_ids']) <range_high)
    for i in model.named_parameters():
        print(f"{i[0]} -> {i[1].device}")
    model.to('cuda')
    # test model
    # reshape it
    masked=False
    # get sentences from ext_obj

    indexed_tokens = tokenized_datasets_long['input_ids']
    # sample 2000 sentences randomly
    np.random.seed(0)
    random_ids=np.random.choice(len(indexed_tokens),2000,replace=False)
    indexed_tokens=[indexed_tokens[x] for x in random_ids]
    # pad the indexed tokens so they are all 400
    # save the indexed tokens length
    indexed_tokens_len=[len(x) for x in indexed_tokens]
    #indexed_tokens=[x+[tokenizer.pad_token_id]*(400-len(x)) for x in indexed_tokens]

    #token_dataloader = DataLoader(indexed_tokens, shuffle=False, batch_size=1)
    # breakdown the index tokens into batches of 16 and do compute_model_activations
    batch_size=16
    num_batches=int(np.ceil(len(indexed_tokens)/batch_size))
    curvature_dict_all=[]
    # create alist that goes from 1 to 21 in steps of 3 and call it time_scales
    time_scales=np.arange(1,21,3)
    for i in tqdm(range(num_batches)):
        batch=indexed_tokens[i*batch_size:(i+1)*batch_size]
        all_batch = compute_model_activations(model, batch, model.device)
        curve_dict_timescale=[]
        for time_scale in time_scales:
            curvature_dict_batch  = compute_model_curvature_timescale(all_batch,time_scale)
            curve_dict_timescale.append(curvature_dict_batch)
        curvature_dict_all.append(curve_dict_timescale)

    # save curvature_dict_all as pickle
    with open(Path(ANALYZE_DIR, f'{modelname}_curvature_dict_timescales_all_range_{range_low}_{range_high}.pkl'), 'wb') as f:
        pickle.dump(curvature_dict_all, f)
    curvature_dict_all=[[x[k] for x in curvature_dict_all] for k in range(len(time_scales))]
    # # turn it into a tensor
    for tt in range(len(time_scales)):
        curve_ = np.concatenate([x['curve'] for x in curvature_dict_all[tt]], axis=1)
        #curve_ = curvature_dict_true['curve']
        fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
        pap_ratio = 8 / 11
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
#        ax.set_ylim([110., 125])


        curve_change = (curve_[1:, :] - curve_[1, :])
        ax = plt.axes((.1, .15, .55, .25 * pap_ratio))
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
                        (np.nanmean(curve_change, axis=1) - np.nanstd(curve_change, axis=1) / np.sqrt(curve_change.shape[1])) * 180 / np.pi,
                        (np.nanmean(curve_change, axis=1) + np.nanstd(curve_change, axis=1) / np.sqrt(curve_change.shape[1])) * 180 / np.pi,
                        color=(0, 0, 0), alpha=.2, zorder=1)

        ax.fill_between(np.arange(curve_change.shape[0]),
                        (np.nanmean(curve_change, axis=1) - np.nanstd(curve_change, axis=1)) * 180 / np.pi,
                        (np.nanmean(curve_change, axis=1) + np.nanstd(curve_change, axis=1)) * 180 / np.pi,
                        color=(0, 0, 0), alpha=.2, zorder=1)

#        ax.set_ylim([-10., 2])

        fig_save_path=Path(ANALYZE_DIR,'traightening','time_scales', f'{modelname}_curvature_long_context_timescale_{time_scales[tt]}_range_{range_low}_{range_high}.pdf')
        # make sure the parent of fig_save_path exists
        fig_save_path.parent.mkdir(parents=True, exist_ok=True)
        # make sure it paernt dir exists
        fig_save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_save_path.__str__(), transparent=True)
