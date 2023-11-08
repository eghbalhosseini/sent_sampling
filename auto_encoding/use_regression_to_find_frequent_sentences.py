import numpy as np
import pandas as pd
import getpass
import os
import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from pathlib import Path
from utils.data_utils import load_obj, construct_stimuli_set, BENCHMARK_CONFIG, save_obj, SAVE_DIR,ANALYZE_DIR
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
from utils import extract_pool
from utils.data_utils import COCA_PREPROCESSED_DIR
# import PCA
from sklearn.decomposition import PCA
#suppress warnings
warnings.filterwarnings('ignore')
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])

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
    k_sample=2000
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
    dataset_id='coca_preprocessed_all_clean_100K_sample_1'
    sample_extractor_id = f'group={model_id}_layers-dataset={dataset_id}_{stim_type}-activation-bench=None-ave=False'
    sampler_obj = extract_pool[sample_extractor_id]()
    model_activation_name = f"{sampler_obj.dataset}_{sampler_obj.stim_type}_{sampler_obj.model_spec[layer_id]}_layer_{layer_id}_{sampler_obj.extract_name}_ave_{sampler_obj.average_sentence}.pkl"
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
    fig, ax = plt.subplots()
    # plot both prediction and actual frequency and normalize them
    ax.hist(y_sample_pred, bins=100, alpha=0.5, label='predicted', density=True)
    ax.hist(rating_frequency_mean, bins=100, alpha=0.5, label='actual', density=True)

    fig.show()
    # find the 5000 sentences with highest prediction
    index_ds_min = np.argsort(y_sample_pred)[-k_sample:]
    # find the 5000 sentences with lowest prediction
    index_ds_max = np.argsort(y_sample_pred)[:k_sample]
    # sentences with highest prediction
    sentences_ds_min = [sample_sent[x] for x in index_ds_min]
    sentence_act_min = [X_sample_pca[x] for x in index_ds_min]
    sentence_id_min = [sample_sent_id[x] for x in index_ds_min]
    # sentences with lowest prediction
    sentences_ds_max = [sample_sent[x] for x in index_ds_max]
    sentence_act_max = [X_sample_pca[x] for x in index_ds_max]
    sentence_id_max = [sample_sent_id[x] for x in index_ds_max]
##
    # save the sentences as text files
    with open(os.path.join(SAVE_DIR,'analysis', f'estimated_ds_min_sentences_model_{model_id}_layer_{layer_id}.txt'), 'w') as f:
        for item in sentences_ds_min:
            f.write("%s\n" % item)

    with open(os.path.join(SAVE_DIR,'analysis', f'estimated_ds_max_sentences_model_{model_id}_layer_{layer_id}.txt'), 'w') as f:
        for item in sentences_ds_max:
            f.write("%s\n" % item)

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
        corr_min = [np.corrcoef(x, y)[0, 1] for y in sentence_act_min]
        corr_max = [np.corrcoef(x, y)[0, 1] for y in sentence_act_max]
        all_corr_min_min.append(corr_min)
        all_corr_min_max.append(corr_max)

    # do the same for act_max
    all_corr_max_min=[]
    all_corr_max_max=[]
    for x in tqdm(ds_max_act_pca):
        # compute correlation between x and each row of act_est_ds_min
        corr_min = [np.corrcoef(x, y)[0, 1] for y in sentence_act_min]
        corr_max = [np.corrcoef(x, y)[0, 1] for y in sentence_act_max]
        all_corr_max_min.append(corr_min)
        all_corr_max_max.append(corr_max)

    # do the same for act_rand

    # plot the histogram of correlations for each group in a figure with 3 subplots
    fig, ax = plt.subplots(3, 1)
    ax[0].hist(np.asarray(all_corr_min_min).flatten(), bins=100, alpha=0.5, label='ds_min with min', density=True)
    ax[0].hist(np.asarray(all_corr_max_min).flatten(), bins=100, alpha=0.5, label='ds_max with min', density=True)
    # show the means
    ax[0].axvline(np.asarray(all_corr_min_min).flatten().mean(), color='b', linestyle='solid', linewidth=1.5)
    ax[0].axvline(np.asarray(all_corr_max_min).flatten().mean(), color='r', linestyle='solid', linewidth=1.5)
    ax[0].legend()
    ax[1].hist(np.asarray(all_corr_min_max).flatten(), bins=100, alpha=0.5, label='ds_min with max', density=True)
    ax[1].hist(np.asarray(all_corr_max_max).flatten(), bins=100, alpha=0.5, label='ds_max with max', density=True)
    ax[1].axvline(np.asarray(all_corr_min_max).flatten().mean(), color='b', linestyle='solid', linewidth=1.5)
    ax[1].axvline(np.asarray(all_corr_max_max).flatten().mean(), color='r', linestyle='solid', linewidth=1.5)
    ax[1].legend()
    ax[2].hist(np.asarray(ds_min_pred).flatten(), bins=100, alpha=0.5, label='predicted ds min', density=True)
    ax[2].hist(np.asarray(ds_max_pred).flatten(), bins=100, alpha=0.5, label='predicted ds max', density=True)
    ax[2].axvline(np.asarray(ds_min_pred).flatten().mean(), color='b', linestyle='solid', linewidth=1.5)
    ax[2].axvline(np.asarray(ds_max_pred).flatten().mean(), color='r', linestyle='solid', linewidth=1.5)
    ax[2].legend()
    fig.show()
    # save figure
    fig.savefig(os.path.join(ANALYZE_DIR, f'correlation_between_ds_parametric_and_estim_max_model_{model_id}_layer_{layer_id}_.png'), dpi=300)
    # save eps
    fig.savefig(os.path.join(ANALYZE_DIR, f'correlation_between_ds_parametric_and_estim_max_model_{model_id}_layer_{layer_id}_.eps'))

    # find repetition in the selected sentence for ds_min
    # step for removing duplicates:

    unique_sent=list(set(sentences_ds_min))
    #
    unique_sent_id=[sentence_id_min[sentences_ds_min.index(x)] for x in unique_sent]
    sampler_obj.load_dataset()
    input_data=sampler_obj.data_
    # filter input data  dataframe based on sampler_obj
    input_data_filter=input_data[input_data['sent_id'].isin(unique_sent_id)]
    # save the sentences in COCA_PREPROCESSED_DIR as a pickle
    save_file_name=os.path.join(COCA_PREPROCESSED_DIR,f'coca_preprocessed_all_clean_100K_sample_1_estim_ds_min.pkl')
    input_data_filter.to_pickle(save_file_name)

    # go into
    # c=sentences.groupby('sentence')
    # len(list(a.groupby('sent_id').apply(lambda x: ' '.join(x.word_form))))