import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

import umap
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import math
import wandb
wandb.login()
import pprint


class CustomLayer(nn.Module):
    def __init__(self,n_channels=7,n_features=650,n_hidden=256):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(n_channels, n_features, n_hidden))
        self.bias = nn.Parameter(torch.randn(n_channels, n_hidden))
    def forward(self, input_data):
        output = torch.einsum('bic,cio->bco', input_data, self.weight) + self.bias
        return output

class GPT2_XL_Encoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_bottleneck):
        super(GPT2_XL_Encoder, self).__init__()
        #self.weight_matrices = torch.stack([torch.randn(650, 256) for _ in range(7)])
        self.fc1 = nn.Linear(n_features,n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_bottleneck)
    def forward(self, input_data):
        x = self.fc1(input_data)
        encoded = (self.fc2(x))
        # reorganize the output so its batch x features x channels
        return encoded

class Decoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden_1,n_hidden_2):
        super(Decoder, self).__init__()
        self.fc1 = CustomLayer(n_channels=6,n_features=n_features,n_hidden=n_hidden_1)
        self.fc2 = CustomLayer(n_channels=6, n_features=n_hidden_1, n_hidden=n_hidden_2)
    def forward(self, encoded):
        x=self.fc1(encoded)
        x = x.permute(0, 2, 1)
        x = (self.fc2(x))
        x = x.permute(0, 2, 1)
        return x

class GPTAutoencoder(nn.Module):
    def __init__(self, input_size, encoder_h, bottleneck_size,decoder_h):
        super(GPTAutoencoder, self).__init__()
        self.encoder = GPT2_XL_Encoder(input_size,encoder_h,bottleneck_size)
        self.decoder = Decoder(bottleneck_size,decoder_h,input_size)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        encoded_matrix = encoded.unsqueeze(-1).repeat(1, 1, 6)
        decoded = self.decoder(encoded_matrix)
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

def build_dataset(model_id,n_components,batch_size):
    model_pca_path = Path(ANALYZE_DIR, f'model_pca_n_comp_{n_components}.pkl')
    model_pca_loads = pd.read_pickle(model_pca_path.__str__())
    input_model_id = [model_id in x['model_name'] for x in model_pca_loads]
    output_model_id = [not (model_id in x['model_name']) for x in model_pca_loads]
    input_model = model_pca_loads[int(np.argwhere(input_model_id))]
    output_model = [model_pca_loads[int(x)] for x in np.argwhere(output_model_id)]
    input_data = torch.tensor(input_model['act'])
    output_data = torch.stack([torch.tensor(x['act']) for x in output_model], dim=2)
    dataset = TensorDataset(input_data, output_data)
    train_ratio = 0.95
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def build_network(input_size,hidden_size,bottleneck_size,decoder_h):
    model = GPTAutoencoder(input_size, hidden_size, bottleneck_size, decoder_h)
    return model

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train_epoch(network, train_loader,test_loader, optimizer,config,epoch):
    epoch_loss = 0
    example_ct = 0
    step_ct = 0
    mseloss = torch.nn.MSELoss(reduction='mean')
    network.train()

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config.batch_size)
    for step, in_tr in enumerate(train_loader):

        # Move data to the device (e.g., GPU) if available
        inputs = in_tr[0].to(device)
        targets = in_tr[1].to(device)

        encoded, decoded = network(inputs)
        # Zero the gradients
        if config.loss_mode=='MSE':
            loss_act = mseloss(targets, decoded)
        else:
            similarities = similarity_loss(targets, decoded)
            loss_act = torch.sum(similarities) + torch.sqrt(torch.var(similarities))
        activation_loss = (1/config.bottleneck_size)*config.alpha_r * torch.norm(encoded, p=2)
        loss = activation_loss + loss_act

        # Backward pass and optimization
        optimizer.zero_grad()
        max_grad_norm = 0.1  # Set your maximum threshold
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
        optimizer.step()
        epoch_loss += loss.item()
        metrics = {"train/train_loss": loss,
                   "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                   "train/example_ct": example_ct}
        if step + 1 < n_steps_per_epoch:
            # 🐝 Log train metrics to wandb
            wandb.log(metrics)
        step_ct += 1
        example_ct += len(inputs)
    # Test
    network.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            #
            encoded, decoded = network(inputs)
            if config.loss_mode == 'MSE':
                loss_act = mseloss(targets, decoded)
            else:
                similarities = similarity_loss(targets, decoded)
                loss_act = torch.sum(similarities) + torch.sqrt(torch.var(similarities))
            #test_similarites = similarity_loss(targets, decoded)
            activation_loss = (1 / config.bottleneck_size) * config.alpha_r * torch.norm(encoded, p=2)
            loss = 0*activation_loss + loss_act  # Backward pass and optimization
            #XY_loss = torch.sum(test_similarites) + torch.sqrt(torch.var(test_similarites))
            batch_loss = loss
            # Compute the loss
            test_loss += batch_loss.item()
            # 🐝 Log train and validation metrics to wandb
    val_metrics = {"val/val_loss": test_loss}
    wandb.log({**metrics, **val_metrics})
    return epoch_loss / len(train_loader), test_loss

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        train_loader,test_loader = build_dataset(config.model_id,650,config.batch_size)
        network = build_network(650,config.hidden_size, config.bottleneck_size,config.decoder_h)
        network=network.to(device)
        optimizer = build_optimizer(network, config.optimizer, config.lr)

        for epoch in range(config.epochs):
            avg_loss,val_loss = train_epoch(network, train_loader,test_loader, optimizer,config,epoch)
            wandb.log({"loss": avg_loss,'valid_loss':val_loss, "epoch": epoch})
        return network

#%%
if __name__ == '__main__':
    # %%
    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'valid_loss',
        'goal': 'minimize'
    }
    parameters_dict = {
        'optimizer': {
            #'values': ['adam', 'sgd']
            'value':  'sgd'
        },
        'hidden_size': {
            'values': [128]
        },
        'bottleneck_size': {
            'values': [16]
        },
        'decoder_h': {
            'values': [ 512]
        }

 #        ,'model_id':{'values':['roberta-base',
 # 'xlnet-large-cased',
 # 'bert-large-uncased-whole-word-masking',
 # 'xlm-mlm-en-2048',
 # 'gpt2-xl',
 # 'albert-xxlarge-v2',
 # 'ctrl']}
        , 'model_id': {'values': ['xlnet-large-cased',
                       'gpt2-xl']}

    }
    parameters_dict.update({
        'epochs': {
            'value': 50},
        'loss_mode': {
            'value': 'MSE'},

        'lr': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.0001},
    'batch_size': {
        'value': 256},
    'alpha_r': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.001}
    })
    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="mseEncoder_sweep")
    wandb.agent(sweep_id, train, count=20)
    wandb.finish()



    # %% train a model on selected hyperparameters
    wandb.finish()

    wandb.init(project="xlnet_autoencoder",
        config={
            "epochs": 10000,
            "batch_size": 256,
            "lr": 0.00009,
            "hidden_size":128,
            "bottleneck_size":16,
            "decoder_h":512,
            "optimizer":"sgd",
            "model_id":'xlnet-large-cased',
            #'alpha_r':0.0008,
            'alpha_r': 0.0000,
            "loss_mode":'MSE'

        })
    # wandb.init(project="xlnet_autoencoder",
    #     config={
    #         "epochs": 300,
    #         "batch_size": 256,
    #         "lr": 0.27,
    #         "hidden_size":256,
    #         "bottleneck_size":64,
    #         "decoder_h":512,
    #         "optimizer":"sgd",
    #         "model_id":'gpt2-xl',
    #         'alpha_r':0.0021,
    #
    #     })

    config=wandb.config
    xlnet_model=train(config=config)
    wandb.finish()
    alpha=str(config.alpha_r).replace('.','')
    # save model
    model_save_path=Path(ANALYZE_DIR,f'autoencoder_{config.model_id}_{config.loss_mode}_bn_{config.bottleneck_size}_h_{config.hidden_size}_dh_{config.decoder_h}_ep_{config.epochs}_alpha_{alpha}.pt')
    torch.save(xlnet_model.state_dict(),model_save_path.__str__())
    #%% load model
    xlnet_model=build_network(650,config.hidden_size, config.bottleneck_size,config.decoder_h)
    xlnet_model.load_state_dict(torch.load(model_save_path.__str__()))
    xlnet_model=xlnet_model.to(device)
    #%% compute the pca
    n_components=650
    model_pca_path=Path(ANALYZE_DIR,f'model_pca_n_comp_{n_components}.pkl')
    model_pca_loads=pd.read_pickle(model_pca_path.__str__())
    input_model_id=[config.model_id in x['model_name'] for x in model_pca_loads]
    output_model_id=[ not(config.model_id in  x['model_name']) for x in model_pca_loads]
    input_model=model_pca_loads[int(np.argwhere(input_model_id))]

    input_data_min = torch.tensor(input_model['act_min']).to(device)

    input_data_max = torch.tensor(input_model['act_max']).to(device)
    input_data_rand = torch.tensor(input_model['act_rand']).to(device)
    input_data=torch.tensor(input_model['act']).to(device)

    with torch.no_grad():
        encoded_min,decoded_min = xlnet_model(input_data_min)
        encoded_max,decoded_max = xlnet_model(input_data_max)
        encoded_rand, decoded_rand = xlnet_model(input_data_rand)
        encoded_all,decoded_all=xlnet_model(input_data)



    fig, ax = plt.subplots()
    grays = (.8, .8, .8, .5)
    x = encoded_all.detach().cpu().numpy()
    pca = PCA(n_components=10)
    x_pca = pca.fit_transform(x)
    k=0
    p=1

    data = pd.DataFrame({'X': x_pca[:, k], 'Y':x_pca[:,p]})
    sns.scatterplot(data=data, x='X', y='Y', color=grays, s=5, linewidth=0,ax=ax)
    # min
    x_min = encoded_min.detach().cpu().numpy()
    x_points = x_min#scaler.transform(x_min)
    x_points = x_points - pca.mean_
    x_points_pca = np.dot(x_points, pca.components_.T)
    data = pd.DataFrame({'X': x_points_pca[:, k], 'Y': x_points_pca[:, p]})
    sns.scatterplot(data=data, x='X', y='Y', color='b', s=5, linewidth=0,ax=ax)
    # max
    x_max = encoded_max.detach().cpu().numpy()
    x_points = x_max  # scaler.transform(x_min)
    x_points = x_points - pca.mean_
    x_points_pca = np.dot(x_points, pca.components_.T)
    data = pd.DataFrame({'X': x_points_pca[:, k], 'Y': x_points_pca[:, p]})
    sns.scatterplot(data=data, x='X', y='Y', color='r', s=5, linewidth=0,ax=ax)

    fig.show()
    plt.show()

 #%%
    min_dist = .1
    n_neighbors = 50
    metric = 'euclidean'

    model_umap_loads = []
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    act_umap = umap_obj.fit_transform(encoded_all.cpu())
    act_min_umap = umap_obj.transform(encoded_min.cpu())
    act_max_umap = umap_obj.transform(encoded_max.cpu())
    act_rand_umap = umap_obj.transform(encoded_rand.cpu())

    fig, ax = plt.subplots()
    grays = (.8, .8, .8, .5)

    data = pd.DataFrame({'X': act_umap[:, 0], 'Y': act_umap[:, 1]})
    sns.scatterplot(data=data, x='X', y='Y', color=grays, s=5, linewidth=0, ax=ax)

    data = pd.DataFrame({'X': act_min_umap[:, 0], 'Y': act_min_umap[:, 1]})
    sns.scatterplot(data=data, x='X', y='Y', color='b', s=5, linewidth=0, ax=ax)

    data = pd.DataFrame({'X': act_max_umap[:, 0], 'Y': act_max_umap[:, 1]})
    sns.scatterplot(data=data, x='X', y='Y', color='r', s=5, linewidth=0, ax=ax)

    fig.show()
