import numpy as np
import pandas as pd
import getpass
import os
import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from pathlib import Path
from utils.data_utils import load_obj, construct_stimuli_set, BENCHMARK_CONFIG, save_obj, SAVE_DIR
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
    layer_id=24
    # samples from the model
    k_sample=1000
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
    # sentences with lowest prediction
    sentences_ds_max = [sample_sent[x] for x in index_ds_max]

    # save the sentences as text files
    with open(os.path.join(SAVE_DIR,'analysis', f'estimated_ds_min_sentences_model_{model_id}_layer_{layer_id}.txt'), 'w') as f:
        for item in sentences_ds_min:
            f.write("%s\n" % item)

    with open(os.path.join(SAVE_DIR,'analysis', f'estimated_ds_max_sentences_model_{model_id}_layer_{layer_id}.txt'), 'w') as f:
        for item in sentences_ds_max:
            f.write("%s\n" % item)