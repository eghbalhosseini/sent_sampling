import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from utils import extract_pool
from utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import torch
from scipy.spatial.distance import pdist, squareform
# check if gpu is available
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import umap
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.colors as mcolors
import torch
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


class Encoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_bottleneck):
        super(Encoder, self).__init__()
        #self.weight_matrices = torch.stack([torch.randn(650, 256) for _ in range(7)])
        self.fc1_hidden = nn.linear(n_features=n_features,n_hidden=n_hidden)
        self.fc2_botlneck = nn.Linear(n_hidden, n_bottleneck)
    def forward(self, input_data):
        x = self.fc1_hidden(input_data)
        encoded = (self.fc2_botlneck(x))
        return encoded

class Decoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden):
        super(Decoder, self).__init__()
        self.fc1 = np.linear(n_features=n_features,n_hidden=n_hidden)
        # add a dropout layer
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, encoded):
        x=self.dropout(encoded)
        x=self.fc1(x)
        return x

class JointEmbedding(nn.Module):
    def __init__(self, input_size, encoder_h, bottleneck_size):
        super(JointEmbedding, self).__init__()
        self.encoder = Encoder(input_size,encoder_h,bottleneck_size)
        self.decoder = Decoder(bottleneck_size,input_size)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':

    #%% compute the pca
    n_components=650
    extract_id = 'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False'
    model_pca_path=Path(ANALYZE_DIR,f'{extract_id}_pca_n_comp_{n_components}.pkl')
    train_pca_loads=pd.read_pickle(model_pca_path.__str__())

    # create a plot with the number of models
    # get model names from ext_obj.model_group_act
    input_size = n_components
    hidden_size = 256
    bottleneck_size = 32
    num_epochs = 1000
    learning_rate = 0.003260644
    batch_size = 128
    alpha_activation=1e-8
    input_data=[torch.tensor(x['act']) for x in train_pca_loads]
    input_data_shape=[x.shape[1] for x in input_data]
    train_data = torch.stack(input_data, dim=2)
    train_size = int(0.9 * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
    print("Train set size:", len(train_dataset))
    print("Test set size:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimilarityAutoencoder(input_size, hidden_size, bottleneck_size).to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    mseloss = torch.nn.MSELoss(reduction='mean')
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        # Training
        model.train()
        for inputs in train_loader:
            inputs = inputs.to(device)
            encoded,decoded=model(inputs)
            # Zero the gradients
            similarities = similarity_loss(inputs, decoded)
            mse_val=mseloss(decoded,inputs)
            XY_loss = torch.sum(similarities) + torch.var(similarities)
            #activation_loss = alpha_activation * torch.norm(encoded, p=2)
            #loss = activation_loss+ XY_loss            # Backward pass and optimization
            #loss= XY_loss
            loss = mse_val
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Test
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs in test_loader:
                inputs = inputs.to(device)
                #
                encoded, decoded = model(inputs)
                test_similarites=similarity_loss(inputs, decoded)
                activation_loss = alpha_activation * torch.norm(encoded, p=2)
                XY_loss = torch.sum(test_similarites)# + torch.var(test_similarites)
                batch_loss = XY_loss+ XY_loss
                # Compute the loss
                test_loss += batch_loss.item()

        # Print epoch loss
        print("Epoch [{}/{}], Loss: {:.4f}, Test Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss, test_loss))
    model_pca_loads=load_obj(os.path.join(ANALYZE_DIR,'model_pca_n_comp_650.pkl'))
    modle_names=[x['model_name'] for x in model_pca_loads]
    model_sh = ['RoBERTa', 'XLNet-L', 'BERT-L', 'XLM', 'GPT2-XL', 'ALBERT-XXL', 'CTRL']
    input_data_min = torch.stack([torch.tensor(x['act_min']) for x in model_pca_loads])
    input_data_max = torch.stack([torch.tensor(x['act_max']) for x in model_pca_loads])
    input_data_rand = torch.stack([torch.tensor(x['act_rand']) for x in model_pca_loads])
    # reshape input_data_min to have the shape n_samples x n_features x n_models
    input_data_min = input_data_min.permute(1, 2, 0).to(device)
    input_data_max = input_data_max.permute(1, 2, 0).to(device)
    input_data_rand=input_data_rand.permute(1, 2, 0).to(device)

    XX_min=corr_coeff(input_data_min)
    pairs = torch.combinations(torch.arange(XX_min.shape[-1]), with_replacement=False)
    XX_min_vec = XX_min[:, pairs[:, 0], pairs[:, 1]]
    XX_min_corr= 1-torch.corrcoef(XX_min_vec)
    pairs = torch.combinations(torch.arange(XX_min_corr.shape[-1]), with_replacement=False)
    XX_min_corr_vec=XX_min_corr[ pairs[:, 0], pairs[:, 1]]
    XX_min_corr_vec.mean()
    # print the mean of the correlation
    print(f'mean of the correlation of the min: {XX_min_corr_vec.mean()}')

    XX_min = corr_coeff(input_data_max)
    pairs = torch.combinations(torch.arange(XX_min.shape[-1]), with_replacement=False)
    XX_min_vec = XX_min[:, pairs[:, 0], pairs[:, 1]]
    XX_min_corr = 1 - torch.corrcoef(XX_min_vec)
    pairs = torch.combinations(torch.arange(XX_min_corr.shape[-1]), with_replacement=False)
    XX_min_corr_vec = XX_min_corr[pairs[:, 0], pairs[:, 1]]
    XX_min_corr_vec.mean()
    print(f'mean of the correlation of the max: {XX_min_corr_vec.mean()}')

    XX_min = corr_coeff(input_data_rand)
    pairs = torch.combinations(torch.arange(XX_min.shape[-1]), with_replacement=False)
    XX_min_vec = XX_min[:, pairs[:, 0], pairs[:, 1]]
    XX_min_corr = 1 - torch.corrcoef(XX_min_vec)
    pairs = torch.combinations(torch.arange(XX_min_corr.shape[-1]), with_replacement=False)
    XX_min_corr_vec = XX_min_corr[pairs[:, 0], pairs[:, 1]]
    XX_min_corr_vec.mean()
    print(f'mean of the correlation of the rand: {XX_min_corr_vec.mean()}')

    with torch.no_grad():
        encoded_min,decoded_min = model(input_data_min)
        encoded_max,decoded_max = model(input_data_max)
        encoded_rand, decoded_rand = model(input_data_rand)

    XX_min = corr_coeff(encoded_min)
    pairs = torch.combinations(torch.arange(XX_min.shape[-1]), with_replacement=False)
    XX_min_vec = XX_min[:, pairs[:, 0], pairs[:, 1]]
    XX_min_corr = 1 - torch.corrcoef(XX_min_vec)
    pairs = torch.combinations(torch.arange(XX_min_corr.shape[-1]), with_replacement=False)
    XX_min_corr_vec = XX_min_corr[pairs[:, 0], pairs[:, 1]]
    XX_min_corr_vec.mean()

    XX_min = corr_coeff(encoded_max)
    pairs = torch.combinations(torch.arange(XX_min.shape[-1]), with_replacement=False)
    XX_min_vec = XX_min[:, pairs[:, 0], pairs[:, 1]]
    XX_min_corr = 1 - torch.corrcoef(XX_min_vec)
    pairs = torch.combinations(torch.arange(XX_min_corr.shape[-1]), with_replacement=False)
    XX_min_corr_vec = XX_min_corr[pairs[:, 0], pairs[:, 1]]
    XX_min_corr_vec.mean()



    with torch.no_grad():
        encoded_train= model(train_data.to(device))

    # plot the encoded min in the first 2 feature for each model in a different color

    # fig,ax=plt.subplots()
    # for i in range(5):
    #     #
    #     x=encoded_min[:,:,i].detach().cpu().numpy()
    #     # do a pca on x
    #     pca=PCA(n_components=2)
    #     x_pca=pca.fit_transform(x)
    #     pca.explained_variance_ratio_
    #     # plot the pca
    #     ax.scatter(x_pca[:,0],x_pca[:,1],label=i)
    # ax.legend()
    # # set xlim and y lim to -400 to 400
    # #ax.set_xlim([-400,400])
    # #ax.set_ylim([-400, 400])
    #
    # fig.show()
    #
    #
    # fig,ax=plt.subplots()
    # for i in range(5):
    #     x = encoded_max[:, :, i].detach().cpu().numpy()
    #     pca = PCA(n_components=2)
    #     x_pca = pca.fit_transform(x)
    #     pca.explained_variance_ratio_
    #     # plot the pca
    #     ax.scatter(x_pca[:, 0], x_pca[:, 1], label=i)
    # ax.legend()
    # #ax.set_xlim([-400,400])
    # #ax.set_ylim([-400, 400])
    # plt.show()


    for i in range(7):
        fig, ax = plt.subplots()
        x = encoded_max[:, :, i].detach().cpu().numpy()
        pca = PCA(n_components=5)
        x_pca = pca.fit_transform(x)
        # plot the pca
        ax.scatter(x_pca[:, 0], x_pca[:, 1],s=3,c='r', label=i)
        x = encoded_min[:, :, i].detach().cpu().numpy()
        pca = PCA(n_components=5)
        x_pca = pca.fit_transform(x)
        # plot the pca
        ax.scatter(x_pca[:, 0], x_pca[:, 1],s=2,c='b', label=i)
    #ax.set_xlim([-400,400])
    #ax.set_ylim([-400, 400])
        fig.show()

    sns.set_theme(style="ticks")
    for i in range(7):
        #fig, ax = plt.subplots()
        x = encoded_max[:, :, i].detach().cpu().numpy()
        pca = PCA(n_components=5)
        x_pca_max = pca.fit_transform(x)[:,:2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_pca_min= pca.fit_transform(x)[:,:2]
        # make a
        # plot the pca
        # concat x_pca_max and x_pca_min

        x_pca=np.concatenate((x_pca_min,x_pca_max),axis=0)
        # create labels max and min
        labels=np.concatenate((np.repeat('min',x_pca_min.shape[0]),np.repeat('max',x_pca_max.shape[0])),axis=0)
        # create a df with x_pca and labels
        df=pd.DataFrame(x_pca,columns=['x','y'])
        df['labels']=labels
        g = sns.jointplot(
            data=df,
            x="x", y="y", hue="labels",
            kind="scatter",
        ax=fig)
        # add title
        plt.title(f'{model_sh[i]}')
        plt.show()


    sns.set_theme(style="ticks")
    for i in range(7):
        #fig, ax = plt.subplots()
        x = encoded_max[:, :, i].detach().cpu().numpy()

        x_pca_max = pca.fit_transform(x)[:,:2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_pca_min= pca.fit_transform(x)[:,:2]
        # make a
        # plot the pca
        # concat x_pca_max and x_pca_min

        x_pca=np.concatenate((x_pca_min,x_pca_max),axis=0)
        # create labels max and min
        labels=np.concatenate((np.repeat('min',x_pca_min.shape[0]),np.repeat('max',x_pca_max.shape[0])),axis=0)
        # create a df with x_pca and labels
        df=pd.DataFrame(x_pca,columns=['x','y'])
        df['labels']=labels
        g = sns.jointplot(
            data=df,
            x="x", y="y", hue="labels",
            kind="scatter",
        ax=fig)
        plt.title(f'{model_sh[i]}')
        plt.show()

    sns.set_theme(style="ticks")
    for i in range(7):
        #fig, ax = plt.subplots()
        x_train=encoded_train[0][:, :, i]
        pca = PCA(n_components=5)
        x_pca_train = pca.fit(x_train.detach().cpu().numpy())
        x = encoded_max[:, :, i].detach().cpu().numpy()

        x_pca_max = pca.transform(x)[:,:2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_pca_min= pca.transform(x)[:,:2]
        # make a
        # plot the pca
        # concat x_pca_max and x_pca_min

        x_pca=np.concatenate((x_pca_min,x_pca_max),axis=0)
        # create labels max and min
        labels=np.concatenate((np.repeat('min',x_pca_min.shape[0]),np.repeat('max',x_pca_max.shape[0])),axis=0)
        # create a df with x_pca and labels
        df=pd.DataFrame(x_pca,columns=['x','y'])
        df['labels']=labels
        g = sns.jointplot(
            data=df,
            x="x", y="y", hue="labels",
            kind="scatter",
        ax=fig)
        plt.title(f'{model_sh[i]}')
        plt.show()


    # compute the the norm of encoded_train, encoded_min, encoded_max for each model
    encoded_train_norm=torch.norm(encoded_train[0],dim=1)
    encoded_min_norm=torch.norm(encoded_min,dim=1)
    encoded_max_norm=torch.norm(encoded_max,dim=1)
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]
    # create a figure with 7 subplots and plot the norm of each model as a distribution
    fig,ax=plt.subplots(4,2)
    # flatten ax
    ax=ax.flatten()
    for i in range(7):
        #ax[i].hist(encoded_train_norm[:,i].detach().cpu().numpy(),bins=100,color=colors[1])

        ax[i].hist(encoded_max_norm[:,i].detach().cpu().numpy(),20,color=colors[2],alpha=0.5)
        ax[i].hist(encoded_min_norm[:, i].detach().cpu().numpy(), 20, color=colors[0],alpha=0.5)
        ax[i].set_title(f'{model_sh[i]}')
    plt.tight_layout()
    fig.show()
    #

    # create a figure with 7 subplots and plot the norm of each model as a distribution
    fig,ax=plt.subplots(4,2)
    # flatten ax
    ax=ax.flatten()
    for i in range(7):
        #ax[i].hist(encoded_train_norm[:,i].detach().cpu().numpy(),bins=100,color=colors[1])
        # compute the pca based on encoded_train
        x_train=encoded_train[0][:, :, i]
        pca = PCA(n_components=5)
        x_pca_train = pca.fit_transform(x_train.detach().cpu().numpy())
        x = encoded_max[:, :, i].detach().cpu().numpy()
        x_pca_max = pca.transform(x)[:,:2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_pca_min= pca.transform(x)[:,:2]
        # create the same division for histogram x_pca_max and x_pca_min for histogram based on x_pca_train
        hist_div=np.linspace(x_pca_train.min(),x_pca_train.max(),100)



        # compute the norms of x_pca_max and x_pca_min
        x_pca_max_norm=torch.norm(torch.tensor(x_pca_max),dim=1)
        x_pca_min_norm=torch.norm(torch.tensor(x_pca_min),dim=1)
        ax[i].hist(np.asarray(x_pca_min_norm),hist_div,color=colors[0],zorder=3,alpha=0.5)
        ax[i].hist(np.asarray(x_pca_max_norm),hist_div,color=colors[2],zorder=2,alpha=0.5)
        ax[i].set_title(f'{model_sh[i]}')
        # limit the axis to the range of x_pca_min and x_pca_max
        ax[i].set_xlim([np.min([x_pca_min.min(),x_pca_max.min()]),np.max([x_pca_min.max(),x_pca_max.max()])])


    plt.tight_layout()
    fig.show()