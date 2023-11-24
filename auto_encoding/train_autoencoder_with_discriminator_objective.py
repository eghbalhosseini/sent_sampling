import os
import numpy as np
import sys
from pathlib import Path
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, low_dim_project
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from auto_encoding.sweep_shared_embedding_for_frequent_set import build_dataset, build_optimizer, build_network
from torch.optim import AdamW
import math
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.image as mpimg
class Encoder(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_bottleneck):
        super(Encoder, self).__init__()
        #self.weight_matrices = torch.stack([torch.randn(650, 256) for _ in range(7)])
        self.fc1_hidden = nn.Linear(n_input,n_hidden)
        self.fc2_hidden = nn.Linear(n_hidden, n_hidden)
        self.fc2_botlneck = nn.Linear(n_hidden, n_bottleneck)
    def forward(self, input_data):
        x = nn.functional.relu(self.fc1_hidden(input_data))
        x = nn.functional.relu(self.fc2_hidden(x))
        encoded = (self.fc2_botlneck(x))
        return encoded

class Decoder(torch.nn.Module):
    def __init__(self,n_bottleneck,n_hidden,n_output):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_bottleneck,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_output)
        # add a dropout layer
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, encoded):
        x=self.dropout(encoded)
        x=nn.functional.relu(self.fc1(x))
        decoded = self.fc2(x)
        return decoded

class ModelAutoencoder(nn.Module):
    def __init__(self, input_size, n_hidden, bottleneck_size):
        super(ModelAutoencoder, self).__init__()
        self.encoder = Encoder(input_size,n_hidden,bottleneck_size)
        self.decoder = Decoder(bottleneck_size,n_hidden,input_size)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded
class ModelDataset(Dataset):
    def __init__(self, model_1_data, model_2_data):
        """
        Initialize the dataset with contexts, targets, and attention masks.

        :param contexts: A tensor of shape (num_samples, max_sequence_length)
        :param targets: A tensor of shape (num_samples, max_sequence_length)
        :param attention_masks: A binary tensor of shape (num_samples, max_sequence_length)
        """
        self.data_1 = model_1_data
        self.data_2 = model_2_data


    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data_1)

    def __getitem__(self, idx):
        """
        Return the context, target, and attention mask tensors for a given index.
        """
        return {
            'model_1': self.data_1[idx],
            'model_2': self.data_2[idx],

        }

def discriminative_loss(latent1, latent2):
    return nn.MSELoss()(latent1, latent2)

if __name__ == '__main__':
    dataset_id = 'coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K'
    extract_id = f'group=best_performing_pereira_1-dataset={dataset_id}_textNoPeriod-activation-bench=None-ave=False'
    optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    learning_rate = 1e-4
    batch_size = 128
    config = {"lr": learning_rate, "num_epochs": 20, "batch_size": batch_size, 'optimizer': 'Adam',
                    'alpha': 1, 'beta': 0.00}

    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset(splits=20)
    ext_obj()
    optim_obj = optim_pool[optim_id]()
    optim_obj.load_extractor(ext_obj)

    ext_obj.model_spec
    act_1=torch.tensor([x[0] for x in optim_obj.activations[1]['activations']])
    act_2=torch.tensor([x[0] for x in optim_obj.activations[0]['activations']])

    # make an ecnoder for act_1
    model_dataset = ModelDataset(act_1,act_2)
    # create train and test splits
    dataset_size = len(model_dataset)
    train_size = int(dataset_size * 0.9)  # Let's say we want 80% for training
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(model_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # create the first autoencoder
    model_1_autoencoder = ModelAutoencoder(act_1.shape[1], 256, 16).to(device)
    model_2_autoencoder = ModelAutoencoder(act_2.shape[1], 256, 16).to(device)
    # create the second autoencoder
    optimizer1=optim.Adam(list(model_1_autoencoder.parameters())+list(model_2_autoencoder.parameters()),lr=learning_rate)
    #optimizer2= optim.Adam(model_2_autoencoder.parameters(),lr=learning_rate)
    mseloss = torch.nn.MSELoss(reduction='mean')
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        example_ct = 0
        step_ct = 0
        model_1_autoencoder.train()
        model_2_autoencoder.train()
        n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config['batch_size'])
        for step, inputs in tqdm(enumerate(train_loader)):
            # Move data to the device (e.g., GPU) if available
            inputs_m1 = inputs['model_1'].to(device)
            inputs_m2 = inputs['model_2'].to(device)

            latent1, reconstructed1 = model_1_autoencoder(inputs_m1)
            latent2, reconstructed2 = model_2_autoencoder(inputs_m2)

            loss_1 = mseloss(inputs_m1, reconstructed1)
            loss_2 = mseloss(inputs_m2, reconstructed2)
            disc_loss = discriminative_loss(latent1, latent2)

            total_loss = loss_1 + loss_2 + disc_loss

            # Perform a single backward pass
            optimizer1.zero_grad()
            #optimizer2.zero_grad()
            total_loss.backward()
            optimizer1.step()
            #optimizer2.step()

        model_1_autoencoder.eval()
        model_2_autoencoder.eval()
        with torch.no_grad():
            test_loss = 0.0
            total_disc_loss= 0.0
            model_1_loss = 0.0
            model_2_loss = 0.0
            all_encoded1 = []
            all_encoded2 = []
            for inputs in test_loader:
                inputs_m1 = inputs['model_1'].to(device)
                inputs_m2 = inputs['model_2'].to(device)
                #
                encoded1, decoded1 = model_1_autoencoder(inputs_m1)
                encoded2, decoded2 = model_2_autoencoder(inputs_m2)

                loss_1 = mseloss(inputs_m1, decoded1)
                loss_2 = mseloss(inputs_m2, decoded2)

                disc_loss = discriminative_loss(encoded1, encoded2)
                loss = loss_1 + loss_2 #+ disc_loss

                # XY_loss = torch.sum(test_similarites) + torch.sqrt(torch.var(test_similarites))
                batch_loss = loss
                # Compute the loss
                test_loss += batch_loss.item()
                model_1_loss += loss_1.item()
                model_2_loss += loss_2.item()
                total_disc_loss += disc_loss.item()
                # 🐝 Log train and validation metrics to wandb
                all_encoded1.append(encoded1)
                all_encoded2.append(encoded2)
        val_metrics = {"val/val_loss": test_loss}
        print(
            "Epoch [{}/{}], Loss: {:.2f}, Test Loss: {:.2f}, model 1 Loss {:.2f}, Model 2 loss {:.2f} disc loss {:.2f}".format(epoch + 1, config['num_epochs'], epoch_loss, test_loss,model_1_loss,model_2_loss,total_disc_loss))
        pca = PCA(n_components=5)
        plt.clf()
        all_encoded1 = torch.cat(all_encoded1, dim=0)
        all_encoded2 = torch.cat(all_encoded2, dim=0)
        pca.fit(all_encoded1.cpu().numpy())
        pca1 = pca.transform(all_encoded1.cpu().numpy())
        pca2 = pca.transform(all_encoded2.cpu().numpy())
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        axs.scatter(pca1[:, 0], pca1[:, 1], label='model 1')
        axs.scatter(pca2[:, 0], pca2[:, 1], label='model 2')
        axs.legend()
        fig.show()
        #fig.savefig('/om/user/ehoseini/sent_sampling/auto_encoding/test.png',dpi=300)


    # plot latent space for model 1 and model 2
    model_1_autoencoder.eval()
    model_2_autoencoder.eval()
    all_encoded1= []
    all_encoded2 = []
    with torch.no_grad():
        test_loss = 0.0

        for inputs in test_loader:
            inputs_m1 = inputs['model_1'].to(device)
            inputs_m2 = inputs['model_2'].to(device)
            #
            encoded1, decoded1 = model_1_autoencoder(inputs_m1)
            encoded2, decoded2 = model_2_autoencoder(inputs_m2)
            # plot the latent space
            all_encoded1.append(encoded1)
            all_encoded2.append(encoded2)

    all_encoded1=torch.cat(all_encoded1,dim=0)
    all_encoded2=torch.cat(all_encoded2,dim=0)
    # plot the latent space
    pca = PCA(n_components=5)
    pca.fit(all_encoded1.cpu().numpy())
    pca1=pca.transform(all_encoded1.cpu().numpy())
    pca2=pca.transform(all_encoded2.cpu().numpy())
    plt.clf()
    plt.scatter(pca1[:,0],pca1[:,1],label='model 1')
    plt.scatter(pca2[:,0],pca2[:,1],label='model 2')
    plt.legend()
    plt.show()