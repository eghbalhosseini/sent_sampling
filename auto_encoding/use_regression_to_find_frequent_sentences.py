import numpy as np
import pandas as pd
import getpass
import os
import sys
from pathlib import Path
from sent_sampling.utils.data_utils import load_obj, construct_stimuli_set, BENCHMARK_CONFIG, save_obj, SAVE_DIR,ANALYZE_DIR
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
from sent_sampling.utils import extract_pool
from sent_sampling.utils.data_utils import COCA_PREPROCESSED_DIR

# import PCA
from sklearn.decomposition import PCA
#suppress warnings
warnings.filterwarnings('ignore')
import fnmatch
from scipy.stats import ttest_ind

import os
import hashlib
import pickletools
import matplotlib

#matplotlib.rcParams['font.size'] = 10
#matplotlib.rcParams['pdf.fonttype'] = 4
matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 10,'font.weight':'bold'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['ps.fonttype'] = 42#
def add_significance_info(ax, x1, x2, y, height, p_value,added_text=None):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.5, c='black')
    star = ''
    if p_value < 0.05:
        star = '*'
    if p_value < 0.01:
        star = '**'
    if p_value < 0.001:
        star = '***'
    else:
        star = 'n.s.'
    ax.text((x1 + x2) * 0.5, y + height, added_text+'\n'+star, ha='center', va='bottom', fontsize=10)


if getpass.getuser() == 'eghbalhosseini':
    data_path = '/Users/eghbalhosseini/MyData/sent_sampling/auto_encoder/'
else:
    data_path = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/auto_encoder/'
if __name__ == '__main__':
    # first set up a regression model
    model_id = 'gpt2-xl'
    dataset_id='neural_ctrl_stim'
    stim_type='textNoPeriod'
    p = Path(data_path, 'beta-control-neural_stimset_D-S_light_freq.csv')
    df = pd.read_csv(p.__str__())
    # get the sentences
    sentences = df['sentence'].values
    extractor_id = f'group={model_id}_layers-dataset={dataset_id}_{stim_type}-activation-bench=None-ave=False'

    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # get rating_frequency_mean for sentences
    rating_frequency_mean=df['rating_frequency_mean'].values
    rating_conversation_mean=df['rating_conversational_mean'].values
    # b
    pca_operator = PCA(n_components=0.8)
    # pick a layer
    layer_id=29
    # samples from the model
    k_sample=5000
    layers=extractor_obj.model_group_act[layer_id]
    sentence_from_ext = [x[1] for x in layers['activations']]
    activations_from_ext = [x[0] for x in layers['activations']]
    X = np.array(activations_from_ext)
    y=rating_frequency_mean
    # do a pca and reduce the dimensionality of the data so that 90% of the variance is explained
    # do a pca that select 10 components
    np.all([x == y for x, y in zip(sentences, sentence_from_ext)])
    X = pca_operator.fit_transform(X)
    regress_model = LinearRegression()
    regress_model.fit(X, y)
    # get the weights
    weights = regress_model.coef_

    # load the sentence from coca
    #dataset_id='coca_preprocessed_all_clean_100K_sample_1'
    all_input_data=[]
    all_sentences_ds_min=[]
    all_perceived_freq_ds_min=[]
    all_perceived_freq_ds_max = []
    all_sentences_act_min=[]
    all_sentences_id_min=[]
    all_sentences_ds_max=[]
    all_sentences_act_max=[]
    all_sentences_id_max=[]

    for dataset_ids in ['coca_preprocessed_all_clean_no_dup_100K_sample_1','coca_preprocessed_all_clean_no_dup_100K_sample_2']:
        #dataset_id_1 = 'coca_preprocessed_all_clean_no_dup_100K_sample_1'
        #/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/coca_preprocessed_all_clean_no_dup_100K_sample_1_textNoPeriod_gpt2-xl_layer_29_activation_ave_False.pkl
        sample_extractor_id = f'group={model_id}_layers-dataset={dataset_ids}_{stim_type}-activation-bench=None-ave=False'
        sampler_obj_1 = extract_pool[sample_extractor_id]()
        sampler_obj_1.load_dataset(splits=200)
        input_data = sampler_obj_1.data_
        all_input_data.append(input_data)
        model_activation_name = f"{sampler_obj_1.dataset}_{sampler_obj_1.stim_type}_{sampler_obj_1.model_spec[layer_id]}_layer_{layer_id}_{sampler_obj_1.extract_name}_ave_{sampler_obj_1.average_sentence}.pkl"
        sample_layer = load_obj(os.path.join(SAVE_DIR, model_activation_name))
        sample_act = [x[0] for x in sample_layer]
        sample_sent=[x[1] for x in sample_layer]
        sample_sent_id=[x[2] for x in sample_layer]
        X_sample = np.array(sample_act)
        # use pca operator to reduce the dimensionality of the X_sample
        X_sample_pca = pca_operator.transform(X_sample)
        # use regressor to get prediction of the frequency
        y_sample_pred = regress_model.predict(X_sample_pca)
        # plot the historgram of the prediction
    #     fig, ax = plt.subplots()
    # # plot both prediction and actual frequency and normalize them
    #     ax.hist(y_sample_pred, bins=100, alpha=0.5, label='predicted', density=True)
    #     ax.hist(rating_frequency_mean, bins=100, alpha=0.5, label='actual', density=True)
    #     fig.show()
        # find the k_sample sentences with highest prediction
        index_ds_min = np.argsort(y_sample_pred)[-k_sample:]
        # find the k_sample sentences with lowest prediction
        index_ds_max = np.argsort(y_sample_pred)[:k_sample]
        # get also the y_pred for these sentences
        y_pred_ds_min = y_sample_pred[index_ds_min]
        y_pred_ds_max = y_sample_pred[index_ds_max]
    # sentences with highest prediction
        sentences_ds_min = [sample_sent[x] for x in index_ds_min]
        sentence_act_min = [X_sample_pca[x] for x in index_ds_min]
        sentence_id_min = [sample_sent_id[x] for x in index_ds_min]
        # sentences with lowest prediction
        sentences_ds_max = [sample_sent[x] for x in index_ds_max]
        sentence_act_max = [X_sample_pca[x] for x in index_ds_max]
        sentence_id_max = [sample_sent_id[x] for x in index_ds_max]
        all_sentences_ds_min.append(sentences_ds_min)
        all_sentences_act_min.append(sentence_act_min)
        all_sentences_id_min.append(sentence_id_min)
        all_sentences_ds_max.append(sentences_ds_max)
        all_sentences_act_max.append(sentence_act_max)
        all_sentences_id_max.append(sentence_id_max)
        all_perceived_freq_ds_min.append(y_pred_ds_min)
        all_perceived_freq_ds_max.append(y_pred_ds_max)
        # get the perceived frequency for these sentences


##
    # save the sentences as text files
        with open(os.path.join(SAVE_DIR,'analysis', f'{dataset_ids}_estimated_ds_min_{model_id}_layer_{layer_id}_samples_{2*k_sample}.txt'), 'w') as f:
            for item in sentences_ds_min:
                f.write("%s\n" % item)

        with open(os.path.join(SAVE_DIR,'analysis', f'{dataset_ids}_estimated_ds_max_sentences_model_{model_id}_layer_{layer_id}_samples_{2*k_sample}.txt'), 'w') as f:
            for item in sentences_ds_max:
                f.write("%s\n" % item)
    # flatten the lists
    all_sentences_ds_min_flat=[item for sublist in all_sentences_ds_min for item in sublist]
    all_sentences_act_min_flat=[item for sublist in all_sentences_act_min for item in sublist]
    all_sentences_id_min_flat=[item for sublist in all_sentences_id_min for item in sublist]
    all_sentences_ds_max_flat=[item for sublist in all_sentences_ds_max for item in sublist]
    all_sentences_act_max_flat=[item for sublist in all_sentences_act_max for item in sublist]
    all_sentences_id_max_flat=[item for sublist in all_sentences_id_max for item in sublist]
    all_perceived_freq_ds_min_flat=[item for sublist in all_perceived_freq_ds_min for item in sublist]
    all_perceived_freq_ds_max_flat=[item for sublist in all_perceived_freq_ds_max for item in sublist]
    # flatten input data
    all_input_data_flat=pd.concat(all_input_data)
    # get ds_parametric sentences
    dataset_id = 'ds_parametric'
    sample_extractor_id = f'group={model_id}_layers-dataset={dataset_id}_{stim_type}-activation-bench=None-ave=False'
    ds_param_obj = extract_pool[sample_extractor_id]()
    ds_param_obj.load_dataset()
    model_activation_name = f"{ds_param_obj.dataset}_{ds_param_obj.stim_type}_{ds_param_obj.model_spec[layer_id]}_layer_{layer_id}_{ds_param_obj.extract_name}_ave_{ds_param_obj.average_sentence}.pkl"
    ds_parametric_layer = load_obj(os.path.join(SAVE_DIR, model_activation_name))
    ds_stim=ds_param_obj.data_
    # select part of the data framne that have sentence_type =='ds_min'
    ds_min_loc = [idx for idx, x in enumerate(ds_stim['sent_type']) if x == 'ds_min']
    ds_max_loc=[idx for idx, x in enumerate(ds_stim['sent_type']) if x == 'ds_max']
    # get the sentence id
    ds_min_id = np.unique([ds_stim['sent_id'][x] for x in ds_min_loc])
    ds_max_id = np.unique([ds_stim['sent_id'][x] for x in ds_max_loc])

    # go through the ds_parametric layer and select elements that have the same sentence id
    ds_min_loc_layer = [x for idx, x in enumerate(ds_parametric_layer) if x[2] in ds_min_id]
    ds_max_loc_layer = [x for idx, x in enumerate(ds_parametric_layer) if x[2] in ds_max_id]
    # do the same thin
    # get the activations
    ds_min_act = [x[0] for x in ds_min_loc_layer]
    ds_max_act = [x[0] for x in ds_max_loc_layer]
    # do the pca on the ds_min_act and ds_max_act
    ds_min_act_pca = pca_operator.transform(ds_min_act)
    ds_max_act_pca = pca_operator.transform(ds_max_act)
    # predict their frequency
    ds_min_pred = regress_model.predict(ds_min_act_pca)
    ds_max_pred = regress_model.predict(ds_max_act_pca)
    all_corr_min_min=[]
    all_corr_min_max=[]
    for x in tqdm(ds_min_act_pca):
        # compute correlation between x and each row of act_est_ds_min
        corr_min = [np.corrcoef(x, y)[0, 1] for y in all_sentences_act_min_flat]
        corr_max = [np.corrcoef(x, y)[0, 1] for y in all_sentences_act_max_flat]
        all_corr_min_min.append(corr_min)
        all_corr_min_max.append(corr_max)

    # do the same for act_max
    all_corr_max_min=[]
    all_corr_max_max=[]
    for x in tqdm(ds_max_act_pca):
        # compute correlation between x and each row of act_est_ds_min
        corr_min = [np.corrcoef(x, y)[0, 1] for y in all_sentences_act_min_flat]
        corr_max = [np.corrcoef(x, y)[0, 1] for y in all_sentences_act_max_flat]
        all_corr_max_min.append(corr_min)
        all_corr_max_max.append(corr_max)

    # do the same for act_rand

    # plot the histogram of correlations for each group in a figure with 3 subplots
    # make them numpy arrays
    all_corr_min_min=np.asarray(all_corr_min_min)
    all_corr_min_max=np.asarray(all_corr_min_max)
    all_corr_max_min=np.asarray(all_corr_max_min)
    all_corr_max_max=np.asarray(all_corr_max_max)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .2, .25, .55))
    # make the first axis
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]
    # do a box plot for corr
    x=np.array([1,2])
    gap=0.125
    ax.boxplot(np.stack([all_corr_min_min.flatten(),all_corr_min_max.flatten()]).T,positions=x-.125,showmeans=True, meanline=True, showfliers=False, widths=0.25,boxprops=dict(color=colors[0]),zorder=2)
    ax.boxplot(np.stack([all_corr_max_min.flatten(),all_corr_max_max.flatten()]).T,positions=x+.125,showmeans=True, meanline=True, showfliers=False, widths=0.25,boxprops=dict(color=colors[2]),zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(['Percepturally\n frequent','Percepturally \ninfrequent'])
    t, p = ttest_ind(np.asarray(all_corr_min_min).flatten(), np.asarray(all_corr_max_min).flatten(),alternative='greater')
    print(f't-test between all_corr_min_min and all_corr_max_min: t={t}, p={p}')
    add_significance_info(ax, 1-gap, 1+gap, max(np.max(all_corr_min_min[0]), np.max(all_corr_min_min[1])), .05, p,added_text='ds_min>ds_max')

    t, p = ttest_ind(np.asarray(all_corr_max_max).flatten(), np.asarray(all_corr_min_max).flatten(),alternative='greater')
    print(f't-test between all_corr_min_min and all_corr_max_min: t={t}, p={p}')
    add_significance_info(ax, 2 - gap, 2 + gap, max(np.max(all_corr_min_max[0]), np.max(all_corr_max_max[1])), .05, p,added_text='ds_max>ds_min')
    # drop spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0.5,2.5])
    # plot a line at zero
    ax.axhline(0, color='black', linestyle='dashed', linewidth=1,zorder=0)
    ax.set_ylabel(f'Correlation with sentences \n with estimated perceptual frequency (n={2*k_sample})')

    # make a second axis and plot estimated frequences ,
    ax = plt.axes((.55, .2, .125, .55))
    ax.boxplot(all_perceived_freq_ds_min_flat, positions=[1 - .125], showmeans=True, meanline=True, showfliers=False, widths=0.25,
               boxprops=dict(color=colors[0]), zorder=2)
    ax.boxplot(all_perceived_freq_ds_max_flat, positions=[1 + .125], showmeans=True, meanline=True, showfliers=False, widths=0.25,
               boxprops=dict(color=colors[2]), zorder=2)
    ax.set_xlim([0.5, 1.5])
    ax.set_ylim([0, 7])
    ax.set_ylabel(f'Estimated perceptual frequency \n for Sampled sentences n={2*k_sample}')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    t, p = ttest_ind(np.asarray(all_perceived_freq_ds_min_flat).flatten(), np.asarray(all_perceived_freq_ds_max_flat).flatten(), alternative='greater')
    add_significance_info(ax, 1 - gap, 1 + gap, max(np.max(ds_min_pred), np.max(ds_max_pred)) + .1, 0.05, p,
                          added_text='ds_min>ds_max')

    ax=plt.axes((.8, .2, .125, .55))
    # plot estimate frequency for ds_min and ds_max as a box plot
    ax.boxplot(ds_min_pred, positions=[1 - .125], showmeans=True,meanline=True, showfliers=False, widths=0.25, boxprops=dict(color=colors[0]), zorder=2)
    ax.boxplot(ds_max_pred, positions=[1 + .125], showmeans=True, meanline=True, showfliers=False, widths=0.25,
               boxprops=dict(color=colors[2]), zorder=2)
    ax.set_xlim([0.5, 1.5])
    ax.set_ylim([0, 7])
    ax.set_ylabel(f'Estimated perceptual frequency \n for ds_parametric sentences')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    t, p = ttest_ind(np.asarray(ds_min_pred).flatten(), np.asarray(ds_max_pred).flatten(),alternative='greater')
    add_significance_info(ax, 1 - gap, 1 + gap, max(np.max(ds_min_pred), np.max(ds_max_pred))+.1, 0.05, p,
                          added_text='ds_min>ds_max')

    fig.show()
    # save figure
    fig.savefig(os.path.join(ANALYZE_DIR, f'correlation_between_ds_parametric_and_estim_max_model_{model_id}_layer_{layer_id}_n_{2*k_sample}.png'), dpi=300)
    # save eps
    fig.savefig(os.path.join(ANALYZE_DIR, f'correlation_between_ds_parametric_and_estim_max_model_{model_id}_layer_{layer_id}_n_{2*k_sample}.pdf'))
    # save ds_min and ds_max sentences as a pickle
    unique_sent=list(set(all_sentences_ds_min_flat))
    unique_sent_id=np.unique([all_sentences_id_min_flat[all_sentences_ds_min_flat.index(x)] for x in unique_sent])
    # filter input data  dataframe based on sampler_obj
    input_data_filter=all_input_data_flat[all_input_data_flat['sent_id'].isin(unique_sent_id)]
    # save the sentences in COCA_PREPROCESSED_DIR as a pickle
    save_file_name=os.path.join(COCA_PREPROCESSED_DIR,f'coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_{2*k_sample}.pkl')
    input_data_filter.to_pickle(save_file_name)

    unique_sent=list(set(all_sentences_ds_max_flat))

    unique_sent_id=np.unique([all_sentences_id_max_flat[all_sentences_ds_max_flat.index(x)] for x in unique_sent])
    # filter input data  dataframe based on sampler_obj
    input_data_filter=all_input_data_flat[all_input_data_flat['sent_id'].isin(unique_sent_id)]
    # save the sentences in COCA_PREPROCESSED_DIR as a pickle
    save_file_name=os.path.join(COCA_PREPROCESSED_DIR,f'coca_preprocessed_all_clean_100K_sample_1_2_ds_max_est_n_{2*k_sample}.pkl')
    input_data_filter.to_pickle(save_file_name)


