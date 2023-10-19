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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
from utils import extract_pool
#suppress warnings
warnings.filterwarnings('ignore')
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])

if getpass.getuser() == 'eghbalhosseini':
    data_path = '/Users/eghbalhosseini/MyData/sent_sampling/auto_encoder/'
else:
    data_path = '/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/auto_encoder/'

if __name__ == '__main__':
    model_id = 'xlnet-large-cased'
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
    layer_mse=[]
    layer_r2=[]
    layer_mae=[]
    for layers in tqdm(extractor_obj.model_group_act):
        sentence_from_ext = [x[1] for x in layers['activations']]
        activations_from_ext = [x[0] for x in layers['activations']]
        X = np.array(activations_from_ext)
        np.all([x == y for x, y in zip(sentences, sentence_from_ext)])
        # Define the number of folds
        k_folds = 5
        kf = KFold(n_splits=k_folds)
        mse_scores = []
        r2_scores = []
        mae_scores=[]
        y=rating_frequency_mean
        # Split the data into training and testing sets
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Create and fit a linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)
            y_pred_constrained =y_pred#  1 + 6 * (1 / (1 + np.exp(-y_pred)))

            # Calculate the mean squared error
            mse = mean_squared_error(y_test, y_pred_constrained)
            mse_scores.append(mse)

            # Calculate R-squared
            r2 = r2_score(y_test, y_pred_constrained)
            r2_scores.append(r2)
            # calcaute MAE
            mae = mean_absolute_error(y_test, y_pred_constrained)
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

    # plot the r2
    axs[1].errorbar(np.arange(len(layer_r2_mean)),layer_r2_mean,layer_r2_std,marker='o',linestyle='None',capsize=5)
    axs[1].set_title('R2')
    axs[1].set_xlabel('Layer')
    axs[1].set_ylabel('R2')
    # set y lim to [-1,1]
    axs[1].set_ylim([-5,1])
    # plot the mae
    axs[2].errorbar(np.arange(len(layer_mae_mean)),layer_mae_mean,layer_mae_std,marker='o',linestyle='None',capsize=5)
    axs[2].set_title('MAE')
    axs[2].set_xlabel('Layer')
    axs[2].set_ylabel('MAE')

    fig.show()


