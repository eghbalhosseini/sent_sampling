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
#parser = argparse.ArgumentParser()
#parser.add_argument('--modelname', type=str, default='LLAMA_13B')
#args = parser.parse_args()
# get the number of available gpus
device_count=torch.cuda.device_count()
device_type=torch.cuda.get_device_name(0)



if __name__ == '__main__':
    #%%
    #modelname=str(args.modelname)
    modelname='13B'
    weight_path=f'/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA_2_hf/{modelname}'
    config_path=f'/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA_2_hf/{modelname}/config.json'
    tokenizer = LlamaTokenizer.from_pretrained(weight_path)
    modelConfig=LlamaConfig.from_json_file(config_path)
    with init_empty_weights():
        model = LlamaForCausalLM(modelConfig)

    device_map = infer_auto_device_map(model,no_split_module_classes=['LlamaDecoderLayer'],max_memory={0: "40GiB", 1: "40GiB" })
    #device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'],max_memory={0: "44GiB", 1: "44GiB",2: "44GiB",3: "44GiB" })
    #device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'],
    #                                   max_memory={0: "42GiB", 1: "78GiB", 2: "78GiB", 3: "78GiB"})
    # print device map
    print(device_map)
    model = load_checkpoint_and_dispatch(model, checkpoint=weight_path, device_map=device_map)
    for i in model.named_parameters():
        print(f"{i[0]} -> {i[1].device}")
    # test model
    # reshape it
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
    all_layers=compute_model_activations(model,indexed_tokens,model.device)
    curvature_dict_true=compute_model_curvature(all_layers)
    # save it as pickle
    # save individual layers as pt file
    for idk,layer_ in enumerate(all_layers):
        layer_=torch.stack(layer_).half()
        layer_=layer_.cpu()
        layer_save_path=Path(ANALYZE_DIR,'straightening',f'LLAMA_2_{modelname}', f'LLAMA_2_{modelname}_activations_{dataset}_layer_{idk}.pt')
        # make sure it paernt dir exists
        layer_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(layer_,layer_save_path.__str__())

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
    fig_save_path=Path(ANALYZE_DIR,'straightening', f'LLAMA_2_{modelname}_curvature_{dataset}.pdf')
    # make sure it paernt dir exists
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path.__str__(), transparent=True)
