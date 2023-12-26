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
import pickle
import scipy
import seaborn as sns
sys.path.extend([SAMPLING_PARENT, SAMPLING_PARENT])
from sent_sampling.utils import extract_pool
from tqdm import tqdm
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig,AutoModelForMaskedLM
# determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import minicons
from minicons import scorer

if __name__ == '__main__':
    extract_id = 'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'
    optim_id = ['coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True',
                'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-run_gpu=True']
    #
    # read the excel that contains the selected sentences
    # %%  RUN SANITY CHECKS
    ds_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN/ds_parametric/ANNSET_DS_MIN_MAX_from_100ev_eh_FINAL.csv')
    # read also the actuall experiment stimuli
    stim_csv = pd.read_csv('/om2/user/ehoseini/fmri_DNN//ds_parametric/fMRI_final/stimuli_order_ds_parametric.csv',
                           delimiter='\t')
    # find unique conditions
    unique_cond = np.unique(stim_csv.Condition)
    # for each unique_cond find sentence transcript
    unique_cond_transcript = [stim_csv.Stim_transcript[stim_csv.Condition == x].values for x in unique_cond]
    # remove duplicate sentences in unique_cond_transcript
    unique_cond_transcript = [list(np.unique(x)) for x in unique_cond_transcript]
    ds_min_list = unique_cond_transcript[1]
    ds_max_list = unique_cond_transcript[0]
    ds_rand_list = unique_cond_transcript[2]
    # extract the ds_min sentence that are in min_included column
    ds_min_ = ds_csv.DS_MIN_edited[(ds_csv['min_include'] == 1)]
    ds_max_ = ds_csv.DS_MAX_edited[(ds_csv['max_include'] == 1)]
    ds_rand_ = ds_csv.DS_RAND_edited[(ds_csv['rand_include'] == 1)]
    # check if ds_min_ and ds_min_list have the same set of sentences regardless of the order
    assert len([ds_min_list.index(x) for x in ds_min_]) == len(ds_min_)
    assert len([ds_max_list.index(x) for x in ds_max_]) == len(ds_max_)
    assert len([ds_rand_list.index(x) for x in ds_rand_]) == len(ds_rand_)
    # %% MORE SANITY CHECKS FOR THE ACTIVATIONS
    # get the
    ds_min_sent = ds_csv.DS_MIN[(ds_csv['min_include'] == 1)]
    ds_max_sent = ds_csv.DS_MAX[(ds_csv['max_include'] == 1)]
    ds_rand_sent = ds_csv.DS_RAND[(ds_csv['rand_include'] == 1)]
    # laod the extractor
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    #ext_obj()
    model_names = ext_obj.model_spec
    # %% create a model for causalLM
    ds_sentences=[ds_min_sent,ds_rand_sent,ds_max_sent]
    ds_sentence_cond=['ds_min','ds_rand','ds_max']
    #%%
    all_models_scores=dict()
    bidirectional_ids=[0,2,5]
    for bidir_id in bidirectional_ids:
        ds_scores_parametric=[]
        model_ = AutoModelForMaskedLM.from_pretrained(model_names[bidir_id], return_dict=True).to(device)
        model_tokenizer = AutoTokenizer.from_pretrained(model_names[bidir_id], use_fast=False)
        mlm_model = scorer.MaskedLMScorer(model_names[bidir_id], device)
        #toke=AutoTokenizer.from_pretrained(model_names[bidir_id])
        #mlm_model.tokenizer=toke
        for id_ds,ds_sent in enumerate(ds_sentences):
            stimuli=ds_sent
            #ilm_model.compute_stats(ilm_model.prepare_text(stimuli))
            ds_scores=[]
            for stim in tqdm(stimuli):
                ds_score=mlm_model.sequence_score(stim, reduction=lambda x: -x.sum(0).item(), PLL_metric='within_word_l2r')
                ds_scores.append(ds_score[0])
            ds_scores_parametric.append(ds_scores)
            # plot the distirbution of the scores
        all_models_scores[model_names[bidir_id]]=ds_scores_parametric

    unidirect_ids=[1,3,4,6]
    for unidir_id in unidirect_ids:
        ds_scores_parametric=[]
        ilm_model = scorer.IncrementalLMScorer(model_names[unidir_id], device)
        for id_ds,ds_sent in enumerate(ds_sentences):
            stimuli=ds_sent
            #ilm_model.compute_stats(ilm_model.prepare_text(stimuli))
            ds_scores=[]
            for stim in tqdm(stimuli):
                ds_score=ilm_model.sequence_score(stim, reduction=lambda x: -x.sum(0).item())
                ds_scores.append(ds_score[0])
            ds_scores_parametric.append(ds_scores)
            # plot the distirbution of the scores
        all_models_scores[model_names[unidir_id]]=ds_scores_parametric

    #%%
    # create a figure and plot the distribution of the scores for each model, using a boxplot
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]
    # make a dataset with
    # create a list of labels
    fig = plt.figure(figsize=(11, 8))
    # fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.1, .4, .35, .35))
    x = np.arange(len(all_models_scores))
    planets = sns.load_dataset("planets")
    model_ds_min=np.asarray([all_models_scores[x][0] for x in all_models_scores.keys()])
    # make a panda dataframe with 2 columns, model and distance
    model_sh = ['RoBERTa','BERT-L','ALBERT-XXL','XLNet-L',  'XLM', 'GPT2-XL','CTRL']

    #model_ds_min=pd.DataFrame(model_ds_min.T,columns=model_sh)
    #sns.boxplot(x="distance", y="method", data=planets,
    #            whis=[0, 100], width=.6, palette="vlag")

    # Add in points to show each observation
    #sns.stripplot(x="distance", y="method", data=planets,
    #              size=4, color=".3", linewidth=0)

    # plot model_ds_min using boxplot with mean and std
    ax.boxplot(model_ds_min.T, positions=x-.25, showmeans=True, meanline=True, showfliers=False, widths=0.2,boxprops=dict(color=colors[0]))
    # plot ds_rand
    model_ds_rand = np.asarray([all_models_scores[x][1] for x in all_models_scores.keys()])
    ax.boxplot(model_ds_rand.T, positions=x, showmeans=True, meanline=True, showfliers=False, widths=0.2,boxprops=dict(color=colors[1]))
    # plot ds_max
    model_ds_max = np.asarray([all_models_scores[x][2] for x in all_models_scores.keys()])
    ax.boxplot(model_ds_max.T, positions=x+.25, showmeans=True, meanline=True, showfliers=False, widths=0.2,boxprops=dict(color=colors[2]))
    ax.set_xticks(x)
    ax.set_xticklabels(model_sh, rotation=45)
    fig.show()

    # for each model in all_models_scores do a ttest between min,rand and max distributions and print the results
    for model_name in all_models_scores.keys():
        ds_min=all_models_scores[model_name][0]
        ds_rand=all_models_scores[model_name][1]
        ds_max=all_models_scores[model_name][2]
        print(f'{model_name} ds_min vs ds_rand {scipy.stats.ttest_ind(ds_min,ds_rand)}')
        print(f'{model_name} ds_min vs ds_max {scipy.stats.ttest_ind(ds_min,ds_max)}')
        print(f'{model_name} ds_rand vs ds_max {scipy.stats.ttest_ind(ds_rand,ds_max)}')

    # save figure
    fig.savefig('/om2/user/ehoseini/MyData/sent_sampling/analysis/parametric_ds_sentence_scores_minicon.png',dpi=300)
    fig.savefig('/om2/user/ehoseini/MyData/sent_sampling/analysis/parametric_ds_sentence_scores_minicon.eps',format='eps')

    # ilm_model.next_word_distribution([stimuli[1][0:2]])
    # ilm_model.tokenizer.decode(ilm_model.next_word_distribution([stimuli[1][0:10]]).topk(1).indices[0])
    #
    # ilm_model.compute_stats(ilm_model.prepare_text(["I don't know what"]))
    # ilm_model.next_word_distribution(["I don't know"]).topk(1)
    # ilm_model.tokenizer.decode(ilm_model.next_word_distribution(["I don't know"]).topk(1).indices[0])
    #
    # mlm_model = scorer.MaskedLMScorer('roberta-base', 'cuda')
    # mlm_model.compute_stats(mlm_model.prepare_text(["I don't know."]))
    # mlm_model.token_score(["I don't care"],'within_word_l2r')
    # mlm_model.tokenizer.decode(mlm_model.cloze_distribution([("I don't know"," know")]).topk(1).indices[0])
    # (logits,index)=mlm_model.cloze_distribution([("I don't know", "know")]).topk(1)
    # logits
    # mlm_model.distribution(mlm_model.prepare_text(["I don't care"])).topk(1)
    # mlm_model.device
    # model_ids=[0,1,2,3,4,6]
    # for model_id in model_ids:
    #     model_name=model_names[model_id]
    #     for id_ds,ds_sent in enumerate(ds_sentences):
    #         file_path = Path('/om2/user/ehoseini/MyData/sent_sampling/analysis/',
    #                          f'{model_name}_{ds_sentence_cond[id_ds]}_sentence_generation_dict.pkl')
    #         if file_path.exists():
    #             print(f'{file_path.__str__()} exists')
    #             continue
    #         # load the model
    #         print(f'creating {file_path.__str__()}')
    #         config = AutoConfig.from_pretrained(model_name)
    #         tokenizer = AutoTokenizer.from_pretrained(model_name)
    #         causal_model = AutoModelForCausalLM.from_config(config)
    #         causal_model.to(device)
    #         text_generation_config = GenerationConfig(
    #             num_beams=5,
    #             eos_token_id=causal_model.config.eos_token_id,
    #             pad_token_id=causal_model.config.pad_token_id,
    #             output_scores=True,
    #             do_sample=False,
    #             return_dict_in_generate=True
    #         )
    #         sentence_tokens=[]
    #         sentence_text=[]
    #         for sent in tqdm(ds_sent,total=len(ds_sent)):
    #             # split sent by words
    #             sent_words=sent.split()
    #             sentence_token=[]
    #             for kk in range(1,len(sent_words)):
    #                 sent_tok_id = tokenizer(' '.join(sent_words[:kk]), return_tensors='pt')
    #                 for key in sent_tok_id.keys():
    #                     sent_tok_id[key] = sent_tok_id[key].to(device)
    #                 sentence_token.append(sent_tok_id)
    #             sentence_tokens.append(sentence_token)
    #             sentence_text.append(sent)
    #         all_greedy_continuation=[]
    #         all_sentence_continuation=[]
    #         for id_sent,sentence_token in tqdm(enumerate(sentence_tokens),total=len(sentence_tokens)):
    #             sentence_continuation=[]
    #             greedy_last=[]
    #             sentence_prob=[]
    #             for idx,sent_token in enumerate(sentence_token):
    #                 # send sent_token to device
    #                 with torch.no_grad():
    #                     outputs = causal_model.generate(**sent_token,max_new_tokens=len(sentence_token)-idx, generation_config=text_generation_config,pad_token_id=causal_model.config.eos_token_id)
    #                 #selected_beam=outputs['beam_indices'][0]
    #                 # decode the output
    #                 greedy_continuation = tokenizer.decode(outputs['sequences'][0])
    #                 greedy_token=tokenizer.decode(outputs['sequences'][0][idx+1])
    #                 # convert scores to probabilities using softmax
    #                 #scores_prob = torch.nn.functional.softmax(outputs['scores'][0], dim=-1)
    #                 # get the probability of the next word
    #                 sentence_continuation.append(greedy_continuation)
    #                 greedy_last.append(greedy_token)
    #                 #sentence_prob.append(scores_prob[-1, :].tolist())
    #             all_greedy_continuation.append(greedy_last)
    #             all_sentence_continuation.append(sentence_continuation)
    #         # save a dictionary of all_sentence_continuation and original sentence text
    #         sentence_generation_dict={'model':model_name,
    #                                     'ds_condition':ds_sentence_cond[id_ds],
    #                                     'sentence_text':sentence_text,
    #                                     'sentence_continuation':all_sentence_continuation,
    #                                     'greedy_last':all_greedy_continuation}
    #         file_path=Path('/om2/user/ehoseini/MyData/sent_sampling/analysis/',f'{model_name}_{ds_sentence_cond[id_ds]}_sentence_generation_dict.pkl')
    #         # if parent doesnt exist create it
    #         if not file_path.parent.exists():
    #             os.makedirs(file_path.parent)
    #         # save the dictionary
    #         with open(file_path.__str__(), 'wb') as f:
    #             pickle.dump(sentence_generation_dict, f)
