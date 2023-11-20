import numpy as np
import pandas as pd
import getpass
import os
import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from pathlib import Path

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

if __name__ == '__main__':
    model_id = 'roberta-base'
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
    # get rating_conversation_mean for sentences
    rating_conversation_mean=df['rating_conversational_mean'].values
    rating_surprisal_sum=df['surprisal-gpt2-xl_sum'].values
    rating_surprisal_mean=df['surprisal-gpt2-xl_mean'].values
    layer_mse=[]
    layer_r2=[]
    layer_mae=[]
    x_shape=[]
    y = rating_surprisal_mean
    model = LinearRegression()
    # do a ridge regression
    #model = Ridge(alpha=10)
    k_folds = 5
    kf = KFold(n_splits=k_folds)
    pca_operator = PCA(n_components=0.8)
    #pca_operator = PCA(n_components=50)
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
    fig, axs = plt.subplots(1, 3,figsize=(15,5))
    # plot the mse
    axs[0].errorbar(np.arange(len(layer_mse_mean)),layer_mse_mean,layer_mse_std,marker='o',linestyle='None',capsize=5)
    axs[0].set_title('MSE')
    axs[0].set_xlabel('Layer')
    axs[0].set_ylabel('MSE')
    axs[0].set_ylim([0,5])

    # plot the r2
    axs[1].errorbar(np.arange(len(layer_r2_mean)),layer_r2_mean,layer_r2_std,marker='o',linestyle='None',capsize=5)
    axs[1].set_title('R2')
    axs[1].set_xlabel('Layer')
    axs[1].set_ylabel('R2')
    #set y lim to [-1,1]
    axs[1].set_ylim([-1,1])
    # plot a line at zero
    axs[1].axhline(y=0, color='r', linestyle='-')
    # plot the mae
    axs[2].errorbar(np.arange(len(layer_mae_mean)),layer_mae_mean,layer_mae_std,marker='o',linestyle='None',capsize=5)
    axs[2].set_title('MAE')
    axs[2].set_xlabel('Layer')
    axs[2].set_ylabel('MAE')
    axs[2].set_ylim([0, 5])
    fig.show()

    # make a figure and plot the mean R2 vs x_shape
    fig,ax=plt.subplots()

    ax.plot(np.arange(len(layer_mse_mean)),layer_r2_mean,marker='o',label='R2')
    ax.plot(np.arange(len(layer_mse_mean)),x_shape/np.max(x_shape),marker='o',color='red',label='dimensions')
    ax.set_xlabel('Layer')
    ax.set_ylabel('R2')
    ax.set_ylim([-1, 1])
    ax.legend()
    fig.show()



