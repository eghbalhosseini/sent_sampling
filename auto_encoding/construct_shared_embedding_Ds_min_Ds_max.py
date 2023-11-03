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


class MultiChannelLayer(nn.Module):
    def __init__(self):
        super(MultiChannelLayer, self).__init__()

        # Define the channels with different sizes
        self.channel1 = nn.Linear(256, 768)
        self.channel2 = nn.Linear(256, 1024)
        self.channel3 = nn.Linear(256, 1024)
        self.channel4 = nn.Linear(256, 2048)
        self.channel5 = nn.Linear(256, 1600)
        self.channel6 = nn.Linear(256, 4096)
        self.channel7 = nn.Linear(256, 1280)

    def forward(self, x):
        # Apply each channel separately
        out1 = self.channel1(x[:, 0])
        out2 = self.channel2(x[:, 1])
        out3 = self.channel3(x[:, 2])
        out4 = self.channel4(x[:, 3])
        out5 = self.channel5(x[:, 4])
        out6 = self.channel6(x[:, 5])
        out7 = self.channel7(x[:, 6])

        # Concatenate the outputs along the feature dimension
        out = torch.cat((out1, out2, out3, out4, out5, out6, out7), dim=1)

        return out

class CustomLayer(nn.Module):
    def __init__(self,n_channels=7,n_features=650,n_hidden=256):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(n_channels, n_features, n_hidden))
        self.bias = nn.Parameter(torch.randn(n_channels, n_hidden))
    def forward(self, input_data):
        #output=torch.einsum('bic,cio->bco', input_data, self.weight)+self.bias
        output = torch.einsum('bic,cio->bco', input_data, self.weight) + self.bias
        # reshape output to be batch x features x channels
        #output = output.permute(0, 2, 1)
        return output

class Encoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_bottleneck):
        super(Encoder, self).__init__()
        #self.weight_matrices = torch.stack([torch.randn(650, 256) for _ in range(7)])
        self.fc1 = CustomLayer(n_channels=7,n_features=n_features,n_hidden=n_hidden)
        self.fc2_shared = nn.Linear(n_hidden, n_bottleneck)
    def forward(self, input_data):
        x = self.fc1(input_data)
        encoded = (self.fc2_shared(x))
        # reorganize the output so its batch x features x channels
        encoded = encoded.permute(0, 2, 1)
        return encoded

class Decoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden):
        super(Decoder, self).__init__()
        self.fc1 = CustomLayer(n_channels=7,n_features=n_features,n_hidden=n_hidden)
        #self.fc2 = CustomLayer(n_channels=7, n_features=256, n_hidden=650)
    def forward(self, encoded):
        x=self.fc1(encoded)
        x = x.permute(0, 2, 1)
        #x = (self.fc2(x))
        #x = x.permute(0, 2, 1)
        return x

class SimilarityAutoencoder(nn.Module):
    def __init__(self, input_size, encoder_h, bottleneck_size):
        super(SimilarityAutoencoder, self).__init__()
        self.encoder = Encoder(input_size,encoder_h,bottleneck_size)
        self.decoder = Decoder(bottleneck_size,input_size)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded

def corr_coeff(X):
    X_m = (X - X.mean(dim=(0, 1), keepdim=True))
    X_m = torch.nn.functional.normalize(X_m)
    XX = 1 - torch.bmm(torch.permute(X_m, (2, 0, 1)), torch.permute(X_m, (2, 1, 0)))
    return XX

def normalize(X):
    norm = torch.norm(X, p=2, dim=1)
    X = X / norm.unsqueeze(1)
    return X


def similarity_loss(X, Y):
    XX=corr_coeff(X)
    #XX= torch.clamp(XX, 0.0, np.inf)
    YY=corr_coeff(Y)
    # get upper diagonal
    n1 = XX.shape[1]
    pairs = torch.combinations(torch.arange(n1), with_replacement=False)
    XX_vec=XX[:,pairs[:, 0], pairs[:, 1]]
    YY_vec=YY[:,pairs[:,0],pairs[:,1]]
    # compute cosine similarity
    XX_vec=normalize(XX_vec)
    YY_vec=normalize(YY_vec)
    #
    similarites=1-torch.diag(XX_vec @ YY_vec.T)
    #XY_loss=torch.sum(similarites)+torch.var(similarites)
    return similarites

if __name__ == '__main__':

    #%% compute the pca
    n_components=650
    extract_id = 'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False'
    model_pca_path=Path(ANALYZE_DIR,f'{extract_id}_pca_n_comp_{n_components}.pkl')
    train_pca_loads=pd.read_pickle(model_pca_path.__str__())

    # create a plot with the number of models
    # get model names from ext_obj.model_group_act
    input_size = n_components
    hidden_size = 128
    bottleneck_size = 16
    num_epochs = 400
    learning_rate = 0.001
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
            XY_loss = torch.sum(similarities) + torch.var(similarities)
            activation_loss = alpha_activation * torch.norm(encoded, p=2)
            loss = activation_loss+ XY_loss            # Backward pass and optimization
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
                XY_loss = torch.sum(test_similarites) + torch.var(test_similarites)
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

    XX_min = corr_coeff(input_data_max)
    pairs = torch.combinations(torch.arange(XX_min.shape[-1]), with_replacement=False)
    XX_min_vec = XX_min[:, pairs[:, 0], pairs[:, 1]]
    XX_min_corr = 1 - torch.corrcoef(XX_min_vec)
    pairs = torch.combinations(torch.arange(XX_min_corr.shape[-1]), with_replacement=False)
    XX_min_corr_vec = XX_min_corr[pairs[:, 0], pairs[:, 1]]
    XX_min_corr_vec.mean()

    XX_min = corr_coeff(input_data_rand)
    pairs = torch.combinations(torch.arange(XX_min.shape[-1]), with_replacement=False)
    XX_min_vec = XX_min[:, pairs[:, 0], pairs[:, 1]]
    XX_min_corr = 1 - torch.corrcoef(XX_min_vec)
    pairs = torch.combinations(torch.arange(XX_min_corr.shape[-1]), with_replacement=False)
    XX_min_corr_vec = XX_min_corr[pairs[:, 0], pairs[:, 1]]
    XX_min_corr_vec.mean()

    with torch.no_grad():
        encoded_min,decoded_min = model(input_data_min)
        encoded_max,decoded_max = model(input_data_max)
        encoded_rand, decoded_rand = model(input_data_rand)

    similarity_loss(input_data_min,decoded_min)
    similarity_loss(input_data_max, decoded_max)


    with torch.no_grad():
        encoded_train= model(train_data.to(device))

    # plot the encoded min in the first 2 feature for each model in a different color

    fig,ax=plt.subplots()
    for i in range(5):
        #
        x=encoded_min[:,:,i].detach().cpu().numpy()
        # do a pca on x
        pca=PCA(n_components=2)
        x_pca=pca.fit_transform(x)
        pca.explained_variance_ratio_
        # plot the pca
        ax.scatter(x_pca[:,0],x_pca[:,1],label=i)
    ax.legend()
    # set xlim and y lim to -400 to 400
    #ax.set_xlim([-400,400])
    #ax.set_ylim([-400, 400])

    plt.show()


    fig,ax=plt.subplots()
    for i in range(5):
        x = encoded_max[:, :, i].detach().cpu().numpy()
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x)
        pca.explained_variance_ratio_
        # plot the pca
        ax.scatter(x_pca[:, 0], x_pca[:, 1], label=i)
    ax.legend()
    #ax.set_xlim([-400,400])
    #ax.set_ylim([-400, 400])
    plt.show()


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

        # ax.scatter(x_pca[:, 0], x_pca[:, 1],s=3,c='r', label=i)
        # x = encoded_min[:, :, i].detach().cpu().numpy()
        # pca = PCA(n_components=5)
        # x_pca = pca.fit_transform(x)
        # # plot the pca
        # ax.scatter(x_pca[:, 0], x_pca[:, 1],s=2,c='b', label=i)
        #ax.set_xlim([-400,400])
        #ax.set_ylim([-400, 400])
        # add title
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

        # ax.scatter(x_pca[:, 0], x_pca[:, 1],s=3,c='r', label=i)
        # x = encoded_min[:, :, i].detach().cpu().numpy()
        # pca = PCA(n_components=5)
        # x_pca = pca.fit_transform(x)
        # # plot the pca
        # ax.scatter(x_pca[:, 0], x_pca[:, 1],s=2,c='b', label=i)
        #ax.set_xlim([-400,400])
        #ax.set_ylim([-400, 400])
        plt.show()
