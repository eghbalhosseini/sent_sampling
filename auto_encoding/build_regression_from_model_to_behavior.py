import numpy as np
import pandas as pd
import getpass
import os
import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
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
# import PCA
from sklearn.decomposition import PCA
#suppress warnings
warnings.filterwarnings('ignore')
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])

if getpass.getuser() == 'eghbalhosseini':
    data_path = '/Users/eghbalhosseini/MyData/sent_sampling/auto_encoder/'
else:
    data_path = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/auto_encoder/'

import matplotlib
matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 10,'font.weight':'bold'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
# "roberta-base xlnet-large-cased bert-large-uncased-whole-word-masking xlm-mlm-en-2048 gpt2-xl albert-xxlarge-v2 ctrl"

if __name__ == '__main__':
    model_id = 'albert-xxlarge-v2'
    dataset_id='neural_ctrl_stim'
    stim_type='textNoPeriod'
    predict='rating_frequency_mean'
    pca_type='varaince_explaned'
    p = Path(data_path, 'beta-control-neural_stimset_D-S_light_freq.csv')
    df = pd.read_csv(p.__str__())
    # get the sentences
    sentences = df['sentence'].values
    extractor_id = f'group={model_id}_layers-dataset={dataset_id}_{stim_type}-activation-bench=None-ave=False'

    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset(splits=20)
    extractor_obj()
    # get rating_frequency_mean for sentences
    rating_frequency_mean=df['rating_frequency_mean'].values
    # get rating_conversation_mean for sentences
    rating_conversation_mean=df['rating_conversational_mean'].values
    rating_surprisal_sum=df['surprisal-gpt2-xl_sum'].values
    rating_surprisal_mean=df['surprisal-gpt2-xl_mean'].values

    if predict=='rating_conversation_mean':
        y = rating_frequency_mean
    elif predict=='rating_conversation_mean':
        y = rating_conversation_mean
    elif predict=='rating_surprisal_sum':
        y = rating_surprisal_sum
    elif predict=='rating_surprisal_mean':
        y = rating_surprisal_mean

    for pca_type in ['varaince_explaned','n_components']:
        layer_mse = []
        layer_r2 = []
        layer_r2_full = []
        layer_mae = []
        x_shape = []
        # create a linear regression model
        model = LinearRegression()
        # do a ridge regression
        #model = Ridge(alpha=10)
        k_folds = 5
        kf = KFold(n_splits=k_folds)
        if pca_type=='varaince_explaned':
            pca_operator = PCA(n_components=0.8)
        elif pca_type=='n_components':
            pca_operator = PCA(n_components=50)
        for layers in tqdm(extractor_obj.model_group_act):
            sentence_from_ext = [x[1] for x in layers['activations']]
            activations_from_ext = [x[0] for x in layers['activations']]
            X = np.array(activations_from_ext)
            # do a pca and reduce the dimensionality of the data so that 90% of the variance is explained
            # do a pca that select 10 components

            X=pca_operator.fit_transform(X)
            # print number of dimension
            print(X.shape)
            x_shape.append(X.shape[1])
            # print the variance explained
            np.all([x == y for x, y in zip(sentences, sentence_from_ext)])
            # Define the number of folds
            mse_scores = []
            r2_scores = []
            mae_scores=[]
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Create and fit a linear regression model
                model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = model.predict(X_test)
                # Calculate the mean squared error
                mse = mean_squared_error(y_test, y_pred)
                mse_scores.append(mse)

                # Calculate R-squared
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)
                # calcaute MAE
                mae = mean_absolute_error(y_test, y_pred)
                mae_scores.append(mae)
            # do a regression on the whole dataset and get the R2

            model.fit(X, y)
            y_pred = model.predict(X)
            r2_full = r2_score(y, y_pred)
            layer_r2_full.append(r2_full)

            layer_mae.append(mae_scores)
            layer_mse.append(mse_scores)
            layer_r2.append(r2_scores)

        # plot the mean and standard deviation of the mse across layers
        layer_mse_mean=np.mean(layer_mse,axis=1)
        layer_mse_std=np.std(layer_mse,axis=1)
        layer_r2_mean=np.mean(layer_r2,axis=1)
        layer_r2_std=np.std(layer_r2,axis=1)
        layer_mae_mean=np.mean(layer_mae,axis=1)
        layer_mae_std=np.std(layer_mae,axis=1)
        # create a figure with 3 subplots

        fig, axs = plt.subplots(2, 2,figsize=(11, 8))
        # flatten the axs
        axs=axs.flatten()

        axs[0].errorbar(np.arange(len(layer_mse_mean)),layer_mse_mean,layer_mse_std,marker='o',linestyle='None',capsize=5)

        axs[0].set_xlabel('Layer')
        axs[0].set_ylabel('MSE')
        axs[0].set_ylim([0,5])
        # turn off spines
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].set_title(f'{model_id}, {dataset_id} \n {stim_type}, {predict} \n {pca_type}')
        # plot the r2
        axs[1].errorbar(np.arange(len(layer_r2_mean)),layer_r2_mean,layer_r2_std,marker='o',linestyle='None',capsize=5)

        axs[1].set_xlabel('Layer')
        axs[1].set_ylabel('R2')
        #set y lim to [-1,1]
        axs[1].set_ylim([-1,1])
        # plot a line at zero
        axs[1].axhline(y=0, color='r', linestyle='-')
        # also plot the full R2
        axs[1].plot(np.arange(len(layer_r2_mean)),layer_r2_full,marker='o',color='k',label='full R2')
        # plot the mae
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)

        axs[2].errorbar(np.arange(len(layer_mae_mean)),layer_mae_mean,layer_mae_std,marker='o',linestyle='None',capsize=5)

        axs[2].set_xlabel('Layer')
        axs[2].set_ylabel('MAE')
        axs[2].set_ylim([0, 5])
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)

        ax=axs[3]
        ax.plot(np.arange(len(layer_mse_mean)), layer_r2_mean, marker='o', label='R2')
        ax.set_xlabel('Layer')
        ax.set_ylabel('R2', color='blue')
        ax.set_ylim([-1, 1])
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(layer_mse_mean)), x_shape, marker='o', color='red', label='dimensions')
        ax2.set_ylabel('Dimensions', color='red')

        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        fig.show()

        fig.savefig(os.path.join(ANALYZE_DIR, f'regression_{predict}_activation_{dataset_id}_{model_id}_{pca_type}.png'), dpi=300)
        # save eps
        fig.savefig(os.path.join(ANALYZE_DIR, f'regression_{predict}_activation_{dataset_id}_{model_id}_{pca_type}.pdf'))
        # make a figure and plot the mean R2 vs x_shape