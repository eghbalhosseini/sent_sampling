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
import torch.nn.functional as F
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
from accelerate import Accelerator
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import wandb
import uuid
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
class SingleModelDataset(Dataset):
    def __init__(self, model_data):
        """
        Initialize the dataset with contexts, targets, and attention masks.

        :param contexts: A tensor of shape (num_samples, max_sequence_length)
        :param targets: A tensor of shape (num_samples, max_sequence_length)
        :param attention_masks: A binary tensor of shape (num_samples, max_sequence_length)
        """
        self.data = model_data



    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the context, target, and attention mask tensors for a given index.
        """
        return {
            'model': self.data[idx],


        }

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', type=str, default='roberta-base', help='model name')
parser.add_argument('--bottleneck_size', type=int, default=16, help='bottleneck size')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
parser.add_argument('--alpha_r', type=float, default=1e-9, help='alpha_r')
args = parser.parse_args()

if __name__ == '__main__':
    # get the variabels from args
    model_name=args.model_name
    bottleneck_size=int(args.bottleneck_size)
    hidden_size=int(args.hidden_size)
    alpha_r=float(args.alpha_r)
# set the config
    learning_rate = 1e-4
    batch_size = 128

    config = {"lr": learning_rate, "num_epochs": 50, "batch_size": batch_size, 'optimizer': 'Adam',
                    'alpha': 1, 'beta': 0.00,"model_name":model_name,'bottleneck_size':bottleneck_size,'hidden_size':hidden_size,
              'alpha_r':alpha_r}

    gradient_accumulation_steps=1
    dataset_id = 'coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K'
    extract_id = f'group=best_performing_pereira_1-dataset={dataset_id}_textNoPeriod-activation-bench=None-ave=False'
    optim_id = 'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=100-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'

    # ('roberta-base',
    #  'xlnet-large-cased',
    #  'bert-large-uncased-whole-word-masking',
    #  'xlm-mlm-en-2048',
    #  'gpt2-xl',
    #  'albert-xxlarge-v2',
    #  'ctrl')

    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset(splits=20)
    ext_obj()
    optim_obj = optim_pool[optim_id]()
    optim_obj.load_extractor(ext_obj)


    model_idx=int(''.join(([str(idx) if ext_obj.model_spec[idx]==config['model_name'] else '' for idx in range(len(ext_obj.model_spec))])))
    act=torch.tensor([x[0] for x in optim_obj.activations[model_idx]['activations']])
    # make an ecnoder for act_1
    model_dataset = SingleModelDataset(act)
    # create train and test splits
    dataset_size = len(model_dataset)
    train_size = int(dataset_size * 0.9)  # Let's say we want 80% for training
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(model_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # create the first autoencoder
    model_autoencoder = ModelAutoencoder(act.shape[1], hidden_size, bottleneck_size).to(device)

    # create the second autoencoder
    optimizer=optim.Adam(model_autoencoder.parameters(),lr=learning_rate)
    #optimizer2= optim.Adam(model_2_autoencoder.parameters(),lr=learning_rate)
    mseloss = torch.nn.MSELoss(reduction='mean')
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * config['num_epochs']) // gradient_accumulation_steps)
    batch_count = 0
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config['batch_size'])
    accelerate = Accelerator(log_with="wandb",gradient_accumulation_steps=gradient_accumulation_steps)
    accelerate.init_trackers("autoencoding_recon", config=config,init_kwargs={"wandb": {
            "notes": "reconstruction loss",
            "entity": "evlab",
        }})
    random_idx=uuid.uuid4().hex
    accelerate.trackers[0].run.name = f'{config["model_name"]}_{bottleneck_size}_{hidden_size}_{alpha_r}_{random_idx[:5]}'
    model_autoencoder, optimizer, train_loader, test_loader,lr_scheduler,mseloss = accelerate.prepare(
        model_autoencoder, optimizer, train_loader, test_loader,lr_scheduler,mseloss)
    abs_step = 1
    wandb_tracker = accelerate.get_tracker("wandb")
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        example_ct = 0
        step_ct = 0
        model_autoencoder.train()
        torch.cuda.empty_cache()
        for step, inputs in tqdm(enumerate(train_loader)):
            # Move data to the device (e.g., GPU) if available
            inputs_m1 = inputs['model'].to(device)
            with accelerate.accumulate(model_autoencoder):
                latent1, reconstructed1 = model_autoencoder(inputs_m1)
                activation_loss = config['alpha_r'] * torch.mean(torch.norm(latent1,dim=1, p=1))
                loss = mseloss(inputs_m1, reconstructed1)
                loss = loss + activation_loss
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            batch_count += len(inputs)
            accelerate.log({"training_loss": loss}, step=abs_step)
            accelerate.log({"train/train_loss": loss}, step=abs_step)
            accelerate.log({"train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch},
                            step=abs_step)
            accelerate.log({"train/batch_count": batch_count}, step=abs_step)
            # save learning rate
            accelerate.log({"train/lr": optimizer.param_groups[0]['lr']}, step=abs_step)
            if abs_step % 100 == 0:
                model_autoencoder.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    test_loss = 0.0
                    embeddings=[]
                    for inputs in test_loader:
                        inputs_m1 = inputs['model'].to(device)
                        latent1, reconstructed1 = model_autoencoder(inputs_m1)
                        embeddings.append(latent1)
                        loss = mseloss(inputs_m1, reconstructed1)
                        activation_loss = config['alpha_r'] * torch.mean(torch.norm(latent1, dim=1, p=2))
                        # Compute the loss
                        loss = loss + activation_loss
                        test_loss += loss.item()
                accelerate.log({"validation/valid_loss": test_loss}, step=abs_step)
                # do a 2 dimensional PCA
                embeddings= torch.concat(embeddings)
                U,S,V = torch.pca_lowrank(embeddings)
                # make U a list of vectors with 2 dimension
                # split it into 1000 lists
                U=U[:,:2]
                U = torch.split(U, U.shape[0])[0]
                # make it a list of lists
                U = [x.tolist() for x in U]
                wandb_tracker.log({"embeddings": wandb.Table(columns=["pc1", "pc2"], data=U)})
                model_autoencoder.train()
                torch.cuda.empty_cache()
            abs_step += 1

        #print("Epoch [{}/{}], Loss: {:.2f}, Test Loss: {:.2f}".format(epoch + 1, config['num_epochs'], epoch_loss, test_loss))
    # save model
    random_idx=uuid.uuid4().hex

    save_id=f'autoencoder_recon_{dataset_id}_model_{config["model_name"]}_b_{config["bottleneck_size"]}_h_{config["hidden_size"]}_alpha_r_{config["alpha_r"]}_id_{random_idx}'
    save_dir = Path(SAVE_DIR,'auto_encoding', save_id)
    # if path does not exist, create it
    save_dir.mkdir(parents=True, exist_ok=True)
    # save the model using accelerate
    accelerate.save({'model': model_autoencoder.state_dict(), 'optimizer': optimizer.state_dict()}, Path(save_dir, 'model.pt'))
    # save the config
    save_obj(config, Path(save_dir, 'config.pkl'))

    accelerate.end_training()
    torch.cuda.empty_cache()