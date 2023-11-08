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
import pprint
from auto_encoding.encoding_utils import corr_coeff, similarity_loss, normalize, CustomLayer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer
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

def build_dataset(extract_id,n_components,batch_size,normalize=False):
    standard_scaler = StandardScaler()

    #group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False_pca_n_comp_650.pkl
    model_pca_path = Path(ANALYZE_DIR, f'{str(extract_id)}_pca_n_comp_{n_components}.pkl')
    train_pca_loads = pd.read_pickle(model_pca_path.__str__())
    input_data = [torch.tensor(x['act']) for x in train_pca_loads]
    if normalize:
        input_data = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data]
    train_data = torch.stack(input_data, dim=2).to(torch.float32).to(device)
    train_size = int(0.9 * len(train_data))
    test_size = len(train_data) - train_size
    train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
    print("Train set size:", len(train_dataset))
    print("Test set size:", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def build_network(input_size,hidden_size,bottleneck_size):
    #input_size, encoder_h, bottleneck_size, output_size, theta_size
    model = SimilarityAutoencoder(input_size, hidden_size, bottleneck_size).to(device)
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
    for step, input in enumerate(train_loader):

        # Move data to the device (e.g., GPU) if available
        inputs = input.to(device)

        encoded, decoded = network(inputs)
        # Zero the gradients
        if config.loss_mode=='MSE':
            loss_act = mseloss(inputs, decoded)
        elif config.loss_mode == 'SIM':
            similarities = similarity_loss(inputs, decoded)
            loss_act = torch.mean(similarities) #+ torch.sqrt(torch.var(similarities))

        if config.activation_loss:
            activation_loss = (1/config.bottleneck_size)*config.alpha_r * torch.norm(encoded, p=2)
        else:
            activation_loss=0
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
        for inputs in test_loader:
            inputs = inputs.to(device)
            #
            encoded, decoded = network(inputs)
            if config.loss_mode == 'MSE':
                loss_act = mseloss(inputs, decoded)
            elif config.loss_mode == 'SIM':
                similarities = similarity_loss(inputs, decoded)
                loss_act = torch.mean(similarities) + torch.sqrt(torch.var(similarities))
            if config.activation_loss:
                activation_loss = (1 / config.bottleneck_size) * config.alpha_r * torch.norm(encoded, p=2)
            else:
                activation_loss = 0

            loss = activation_loss + loss_act

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
        train_loader,test_loader = build_dataset( config.extract_id,650,config.batch_size,config.normalize)
        network = build_network(650, config.hidden_size,config.bottleneck_size)
        network=network.to(device)
        optimizer = build_optimizer(network, config.optimizer, config.lr)
        # save the length of dataset into wandb
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
            'value': 'adam'
            #'value':  'sgd'
        },
        'activation_loss': {
            'values': [True, False]
            #'value':  True
        },
        'normalize': {
            'values': [True, False]
            #'value':  True
        },
        'hidden_size': {
            'values': [128,256, 512]
        },
        'bottleneck_size': {
            'values': [16,32,64]
        },

        'extract_id':{'value':'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False'}
    }
    parameters_dict.update({
        'epochs': {
            'values': [200,500,1000,2000]},
        'loss_mode': {
            'value': 'SIM'},

        'lr': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0.00001,
        'max': 0.005},
    'batch_size': {
        'values': [64,128]},
    'alpha_r': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.01}
    })
    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=f"SharedEmbedding_sweep_SIM")
    wandb.agent(sweep_id, train, count=200)
    wandb.finish()
    # %% train a model on selected hyperparameters

    # wandb.init(project="none",
    #        config={
    #            "epochs": 200,
    #            "batch_size": 128,
    #            "lr":0.003260644,
    #            "hidden_size":128,
    #            "bottleneck_size":64,
    #            "optimizer":"adam",
    #            'alpha_r': 0,
    #            "loss_mode":'SIM',
    #            'extract_id':'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False',
    #            'activation_loss':False,
    #             'normalize':False,
    #        })
    #
    # config=wandb.config
    # auto_model=train(config=config)