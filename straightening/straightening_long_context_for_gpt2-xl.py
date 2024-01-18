import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from transformers import LlamaForCausalLM, LlamaTokenizer,LlamaConfig
import torch
from sent_sampling.utils.curvature_utils import compute_model_activations,compute_model_curvature
from sent_sampling.utils.data_utils import ANALYZE_DIR
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,AutoModelForCausalLM, AutoTokenizer,AutoModel,AutoModelForMaskedLM, AutoConfig
import pickle
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

# Iterate through each CUDA device

def tokenize_function(examples):
    outputs = tokenizer(examples['text'])
    return outputs
import psutil

def foo(model,batch):
        with torch.no_grad():
            batch=torch.stack(batch,1).to(mps_device)
            out = model(batch)
        return out
ANALYZE_DIR='/Users/eghbalhosseini/MyData/sent_sampling/analysis/straightening/long_context'
if __name__ == '__main__':
    #%%
    modelname='gpt2-xl'

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModel.from_pretrained(modelname)
    wiki_data = load_dataset("wikitext",'wikitext-103-raw-v1')
    # get text for train split
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = wiki_data.map(tokenize_function, batched=True, remove_columns=["text"])
    # select subest that are between 2048 and 4096 tokens
    tokenized_datasets=tokenized_datasets['train']
    tokenized_datasets_long = tokenized_datasets.filter(lambda x: 380 <= len(x['input_ids']) <400)
    for i in model.named_parameters():
        print(f"{i[0]} -> {i[1].device}")
    model.to(mps_device)
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
    for i in tqdm(range(num_batches)):
        batch=indexed_tokens[i*batch_size:(i+1)*batch_size]
        all_batch = compute_model_activations(model, batch, model.device)
        curvature_dict_batch  = compute_model_curvature(all_batch)
        curvature_dict_all.append(curvature_dict_batch)

    # save curvature_dict_all as pickle
    with open(Path(ANALYZE_DIR, f'{modelname}_curvature_dict_all.pkl'), 'wb') as f:
        pickle.dump(curvature_dict_all, f)

    # load curvature_dict_all from pickle
    with open(Path(ANALYZE_DIR, f'{modelname}_curvature_dict_all.pkl'), 'rb') as f:
        curvature_dict_all = pickle.load(f)
    curvature_dict_true = {}
    for key in curvature_dict_all[0].keys():
        combined_val=[d[key] for d in curvature_dict_all]
        # combine the list into 1
        combined_val = [item for sublist in combined_val for item in sublist]
        curvature_dict_true[key] = combined_val

    # turn it into a tensor
    curve_ = np.concatenate([x['curve'] for x in curvature_dict_all],axis=1)
    fig = plt.figure(figsize=(8, 11), dpi=200, frameon=False)
    pap_ratio = 8/ 11
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
    ax.set_ylim([110., 125])


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

    fig.show()
    ax.set_ylim([-10., 2])

    fig_save_path=Path(ANALYZE_DIR,'straightening','long_context', f'{modelname}_curvature_long_context.pdf')
    # make sure it paernt dir exists
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path.__str__(), transparent=True)
