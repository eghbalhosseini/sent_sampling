import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer
from auto_encoding.encoding_utils import corr_coeff, similarity_loss, normalize, CustomLayer
from auto_encoding.sweep_shared_embedding_for_frequent_set import build_dataset, build_optimizer, build_network
import math
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
        # add a dropout layer
        self.dropout = nn.Dropout(p=0.2)
        #self.fc2 = CustomLayer(n_channels=7, n_features=256, n_hidden=650)
    def forward(self, encoded):

        x=self.dropout(encoded)
        x=self.fc1(x)
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

if __name__ == '__main__':
    dataset_id='coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K'
    config = {
                   "epochs": 200,
                   "batch_size": 128,
                   "lr":0.0027754682602258566,
                   "hidden_size":512,
                   "bottleneck_size":16,
                   "optimizer":"adam",
                   'alpha_r': 0.00712404624193811,
                   "loss_mode":'MSE',
                   'extract_id':f'group=best_performing_pereira_1-dataset={dataset_id}_textNoPeriod-activation-bench=None-ave=False',
                   'activation_loss':False,
                    'normalize':True,
               }
    #%% compute the pca
    n_components=650
    extract_id = f'group=best_performing_pereira_1-dataset={dataset_id}_textNoPeriod-activation-bench=None-ave=False'
    train_loader, test_loader=build_dataset(extract_id=config['extract_id'],n_components=n_components,batch_size=config['batch_size'],normalize=config['normalize'])
    model=build_network(n_components,hidden_size=config['hidden_size'],bottleneck_size=config['bottleneck_size'])
    optimizer=build_optimizer(model,config['optimizer'],learning_rate=config['lr'])
    mseloss = torch.nn.MSELoss(reduction='mean')
    for epoch in range(config['epochs']):
        epoch_loss = 0
        example_ct = 0
        step_ct = 0
        model.train()
        n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config['batch_size'])
        for step, input in enumerate(train_loader):
            # Move data to the device (e.g., GPU) if available
            inputs = input.to(device)
            encoded, decoded = model(inputs)
            # Zero the gradients
            if config['loss_mode'] == 'MSE':
                loss_act = mseloss(inputs, decoded)
            elif config['loss_mode'] == 'SIM':
                similarities = similarity_loss(inputs, decoded)
                loss_act = torch.mean(similarities)  # + torch.sqrt(torch.var(similarities))
            if config['activation_loss']:
                activation_loss = (1 / config['bottleneck_size']) * config['alpha_r'] * torch.norm(encoded, p=2)
            else:
                activation_loss = 0
            loss = activation_loss + loss_act

            # Backward pass and optimization
            optimizer.zero_grad()
            max_grad_norm = 0.1  # Set your maximum threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            metrics = {"train/train_loss": loss,
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                       "train/example_ct": example_ct}
            step_ct += 1
            example_ct += len(inputs)
            # Test
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                for inputs in test_loader:
                    inputs = inputs.to(device)
                    #
                    encoded, decoded = model(inputs)
                    if config['loss_mode'] == 'MSE':
                        loss_act = mseloss(inputs, decoded)
                    elif config['loss_mode'] == 'SIM':
                        similarities = similarity_loss(inputs, decoded)
                        loss_act = torch.mean(similarities) + torch.sqrt(torch.var(similarities))
                    if config['activation_loss']:
                        activation_loss = (1 / config['bottleneck_size']) * config['alpha_r'] * torch.norm(encoded, p=2)
                    else:
                        activation_loss = 0

                    loss = activation_loss + loss_act

                    # XY_loss = torch.sum(test_similarites) + torch.sqrt(torch.var(test_similarites))
                    batch_loss = loss
                    # Compute the loss
                    test_loss += batch_loss.item()
                    # 🐝 Log train and validation metrics to wandb
            val_metrics = {"val/val_loss": test_loss}
            print("Epoch [{}/{}], Loss: {:.4f}, Test Loss: {:.4f}".format(epoch + 1, config['epochs'], epoch_loss, test_loss))
        # Test



    model_pca_loads=load_obj(os.path.join(ANALYZE_DIR,'model_pca_n_comp_650.pkl'))

    modle_names=[x['model_name'] for x in model_pca_loads]
    model_sh = ['RoBERTa', 'XLNet-L', 'BERT-L', 'XLM', 'GPT2-XL', 'ALBERT-XXL', 'CTRL']
    standard_scaler = StandardScaler()
    input_data_min = [torch.tensor(x['act_min']) for x in model_pca_loads]
    input_data_max = [torch.tensor(x['act_max']) for x in model_pca_loads]
    input_data_rand = [torch.tensor(x['act_rand']) for x in model_pca_loads]
    if config['normalize']:
        input_data_min = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data_min]
        input_data_max = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data_max]
        input_data_rand = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data_rand]
    input_data_min = torch.stack(input_data_min).to(device).permute(1, 2, 0).to(torch.float32)
    input_data_max = torch.stack(input_data_max).to(device).permute(1, 2, 0).to(torch.float32)
    input_data_rand = torch.stack(input_data_rand).to(device).permute(1, 2, 0).to(torch.float32)
    extract_id=config['extract_id']
    train_pca_path = Path(ANALYZE_DIR, f'{extract_id}_pca_n_comp_{n_components}.pkl')
    train_pca_loads = pd.read_pickle(train_pca_path.__str__())
    train_data = [torch.tensor(x['act']) for x in train_pca_loads]
    if config['normalize']:
        train_data = [torch.tensor(standard_scaler.fit_transform(X)) for X in train_data]
    train_data = torch.stack(train_data, dim=2).to(torch.float32).to(device)

    for data,id in [(input_data_min,'min'),(input_data_rand,'rand'),(input_data_max,'max')]:
        XX=corr_coeff(data)
        pairs = torch.combinations(torch.arange(XX.shape[-1]), with_replacement=False)
        XX_vec = XX[:, pairs[:, 0], pairs[:, 1]]
        XX_corr= 1-torch.corrcoef(XX_vec)
        pairs = torch.combinations(torch.arange(XX_corr.shape[-1]), with_replacement=False)
        XX_corr_vec=XX_corr[ pairs[:, 0], pairs[:, 1]]
        print(f'mean of the correlation of the {id}: {XX_corr_vec.mean()}')



    with torch.no_grad():
        encoded_train = model(train_data)
        encoded_min,decoded_min = model(input_data_min)
        encoded_max,decoded_max = model(input_data_max)
        encoded_rand, decoded_rand = model(input_data_rand)

    for data,id in [(encoded_min,'min'),(encoded_rand,'rand'),(encoded_max,'max')]:
        XX=corr_coeff(data)
        pairs = torch.combinations(torch.arange(XX.shape[-1]), with_replacement=False)
        XX_vec = XX[:, pairs[:, 0], pairs[:, 1]]
        XX_corr= 1-torch.corrcoef(XX_vec)
        pairs = torch.combinations(torch.arange(XX_corr.shape[-1]), with_replacement=False)
        XX_corr_vec=XX_corr[ pairs[:, 0], pairs[:, 1]]
        print(f'mean of the correlation of the {id}: {XX_corr_vec.mean()}')



    sns.set_theme(style="ticks")
    for i in range(7):
        #fig, ax = plt.subplots()
        x = input_data_max[:, :, i].detach().cpu().numpy()
        pca = PCA(n_components=5)
        x_pca_max = pca.fit_transform(x)[:,:2]
        x = input_data_min[:, :, i].detach().cpu().numpy()
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
            kind="scatter")
        # add title
        plt.title(f'{model_sh[i]}')
        plt.show()
        #g.savefig(os.path.join(ANALYZE_DIR,f'{model_sh[i]}_ds_min_ds_max_shared_embedding_pca.png'))
        # save eps
        #g.savefig(os.path.join(ANALYZE_DIR,f'{model_sh[i]}_ds_min_ds_max_shared_embedding_pca.eps'), format='eps')



    sns.set_theme(style="ticks")
    for i in range(7):
        #fig, ax = plt.subplots()
        x_train=encoded_train[0][:, :, i]
        pca = PCA(n_components=2)
        x_pca_train = pca.fit(x_train.detach().cpu().numpy())
        x = encoded_max[:, :, i].detach().cpu().numpy()
        x_max = pca.transform(x)[:,:2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_min= pca.transform(x)[:,:2]
        x_pca=np.concatenate((x_min,x_max),axis=0)
        # create labels max and min
        labels=np.concatenate((np.repeat('min',x_min.shape[0]),np.repeat('max',x_max.shape[0])),axis=0)
        # create a df with x_pca and labels
        df=pd.DataFrame(x_pca,columns=['x','y'])
        df['labels']=labels
        g = sns.jointplot(
            data=df,
            x="x", y="y", hue="labels",
            kind="scatter",)
        ax=g.ax_joint
        # ax.axvline(x_min.mean(axis=0)[0],color='b',linestyle='--',zorder=1)
        # ax.axvline(x_max.mean(axis=0)[0],color='r',linestyle='--',zorder=1)
        # ax.axhline(x_min.mean(axis=0)[1],color='b',linestyle='--',zorder=1)
        # ax.axhline(x_max.mean(axis=0)[1],color='r',linestyle='--',zorder=1)
        plt.title(f'{model_sh[i]}\n pca train')
        plt.show()
        g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_shared_embedding_train_pca.png'))
        # save eps
        g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_shared_embedding_train_pca.eps'), format='eps')

    sns.set_theme(style="ticks")
    for i in range(7):
        x = encoded_max[:, :, i].detach().cpu().numpy()
        x_max = x[:,:2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_min = x[:,:2]
        x_pca = np.concatenate((x_min, x_max), axis=0)
        # create labels max and min
        labels = np.concatenate((np.repeat('min', x_min.shape[0]), np.repeat('max', x_max.shape[0])), axis=0)
        # create a df with x_pca and labels
        df = pd.DataFrame(x_pca, columns=['x', 'y'])
        df['labels'] = labels
        g = sns.jointplot(
            data=df,
            x="x", y="y", hue="labels",
            kind="scatter", )
        ax = g.ax_joint
        # ax.axvline(x_min.mean(axis=0)[0], color='b', linestyle='--', zorder=1)
        # ax.axvline(x_max.mean(axis=0)[0], color='r', linestyle='--', zorder=1)
        # ax.axhline(x_min.mean(axis=0)[1], color='b', linestyle='--', zorder=1)
        # ax.axhline(x_max.mean(axis=0)[1], color='r', linestyle='--', zorder=1)
        plt.title(f'{model_sh[i]}\n')
        g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_shared_embedding.png'))
        # save eps
        g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_shared_embedding.eps'),
                  format='eps')
        plt.show()
