import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
desired_path = '/om/weka/evlab/ehoseini/JointMDS'
sys.path.extend([desired_path,desired_path])
sys.path.append(desired_path)
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from utils import extract_pool
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
import torch
import itertools
import matplotlib
import re
import scipy as sp
import transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,AutoModelForCausalLM, AutoTokenizer,AutoModel,AutoModelForMaskedLM, AutoConfig
import xarray as xr
from minicons import scorer
from transformers import PreTrainedTokenizer
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import Dataset, DatasetDict
from scipy import stats
from utils.curvature_utils import compute_model_activations, compute_model_curvature
# make 2 arguments for continuattion and context and get it
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--continuation', type=str, default='7',
                    help='continuation')
parser.add_argument('--context', type=str, default='3',
                    help='context')
if __name__ == '__main__':
    #%%
    # get arguments
    args = parser.parse_args()
    continuation_k=int(args.continuation)
    context_k=int(args.context)

    #continuation_k=7
    #context_k=3
    basemodel = 'gpt2-xl'
    masked=False
    dataset='ud_sentencez_token_filter_v3_textNoPeriod'
    extract_id = ['group=gpt2_layers-dataset=ud_sentencez_token_filter_v3_textNoPeriod-activation-bench=None-ave=None']
    # get data
    ext_obj=extract_pool[extract_id[0]]()
    ext_obj.load_dataset()
    # get sentences from ext_obj
    sentences=[x['text'] for x in ext_obj.data_]
    sentences_words=[x['word_FORM'] for x in ext_obj.data_]
    good_sent_id=np.where(np.asarray([len(x['word_FORM'])==len(x['surprisal_3']) for x in ext_obj.data_]))[0]
    sentences_=[sentences[i] for i in good_sent_id]
    # go through the sentences and find sentences that are more than 6 words long and only take the frist 4 words from them and put them in sentence_piece list
    sentence_piece=[]
    sentence_full=[]
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    sentence_tokens=[]
    sent_token_all=[]
    for sent in sentences_:
        sent_tok_id=tokenizer(sent,return_tensors='pt')['input_ids'][0]
        sent_token_all.append(tokenizer(sent,return_tensors='pt'))
        sentence_tokens.append(sent_tok_id.tolist())

    # find sentence tokens that are longer than context_k+continuation_k
    long_sent_id=np.where(np.asarray([len(x)>context_k+continuation_k for x in sentence_tokens]))[0]
    # filter sentence on long_sent_id
    sentences_=[sentences_[i] for i in long_sent_id]
    sent_token_=[sent_token_all[i] for i in long_sent_id]

    # make a true continuation based on continuation_k+context_k
    true_continuation=[x[:context_k+continuation_k] for x in sentence_tokens]
    true_continuation=[true_continuation[i] for i in long_sent_id]
    # compute greed continuation

    model=AutoModelForCausalLM.from_pretrained(basemodel,cache_dir='/om2/user/ehoseini/.cache/huggingface/transformers/')
    model.to('cuda')
    # if model continuation dict doesnt exsist create it
    path_continuation_dict = Path(ANALYZE_DIR, f'{basemodel}_continuation_dict_cntx_{context_k}_cont_{continuation_k}.pkl')
    if not path_continuation_dict.exists():
    # go to evey element in sent_token_ and and for each key get the first context_k tokens and add the to greedy_inputs
        greedy_inputs=[dict(input_ids=x['input_ids'][:1,:context_k],attention_mask=x['attention_mask'][:1,:context_k]) for x in sent_token_]
        greedy_tok_id=[]
        beam_tok_id=[]
        sample_tok_id=[]
        top_k_tok_id=[]
        top_p_tok_id=[]
        for i in tqdm(range(len(greedy_inputs))):
            tokens_tensor = greedy_inputs[i]
            # move values of torch tensor to cuda
            for key in tokens_tensor.keys():
                tokens_tensor[key]=tokens_tensor[key].to('cuda')
            with torch.no_grad():
                output_greedy = model.generate(**tokens_tensor, max_new_tokens=continuation_k, return_dict_in_generate=False, output_scores=False)
                output_beam = model.generate(**tokens_tensor,num_beams=4, max_new_tokens=continuation_k,return_dict_in_generate=False, output_scores=False)
                output_sample = model.generate(**tokens_tensor, max_new_tokens=continuation_k, return_dict_in_generate=False,do_sample=True, output_scores=False)
                output_top_k = model.generate(**tokens_tensor, max_new_tokens=continuation_k, return_dict_in_generate=False,top_k=50, output_scores=False)
                output_top_p = model.generate(**tokens_tensor, max_new_tokens=continuation_k, return_dict_in_generate=False,top_p=0.9, output_scores=False)

            #greedy_continuation.append(tokenizer.decode(outputs['sequences'][0]))
            greedy_tok_id.append(output_greedy[0].tolist())
            beam_tok_id.append(output_beam[0].tolist())
            sample_tok_id.append(output_sample[0].tolist())
            top_k_tok_id.append(output_top_k[0].tolist())
            top_p_tok_id.append(output_top_p[0].tolist())

        # convert greedy_tok_id to list
        # make a dictionary for both greedy and true continuation
        continuations_dict=dict(greedy=greedy_tok_id,true=true_continuation,beam=beam_tok_id,sample=sample_tok_id,top_k=top_k_tok_id,top_p=top_p_tok_id)
        # define a path for continuation_dict and basemodel and save it

        with open(path_continuation_dict.__str__(),'wb') as f:
            pickle.dump(continuations_dict,f)
    else:
        with open(path_continuation_dict.__str__(),'rb') as f:
            continuations_dict=pickle.load(f)