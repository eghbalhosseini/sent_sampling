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
from torch.utils.data import Subset
import torch.nn.functional as F
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torchvision.transforms.functional import to_pil_image as to_pil_image
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
class MultiModelDataset(Dataset):
    def __init__(self, model_data_list):
        """
        Initialize the dataset with contexts, targets, and attention masks.

        :param contexts: A tensor of shape (num_samples, max_sequence_length)
        :param targets: A tensor of shape (num_samples, max_sequence_length)
        :param attention_masks: A binary tensor of shape (num_samples, max_sequence_length)
        """
        self.data_list = model_data_list



    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data_list[0])

    def __getitem__(self, idx):
        """
        Return the context, target, and attention mask tensors for a given index.
        """
        return {
            'idx': idx,
            'model_list': [data[idx] for data in self.data_list],
        }

def pairwise_cosine_similarity(X):
        """
        Compute the pairwise cosine similarity for a matrix.

        Args:
        X (torch.Tensor): A matrix of size [n_samples, n_features].

        Returns:
        torch.Tensor: A matrix of size [n_samples, n_samples] containing
                      the pairwise cosine similarities.
        """
        # Normalize each row (sample) in X to be unit vector
        X_normalized = F.normalize(X, p=2, dim=1)

        # Compute pairwise cosine similarity
        similarity_matrix = torch.mm(X_normalized, X_normalized.transpose(0, 1))

        return similarity_matrix

def nt_xent_loss_multi(embeddings, temperature):
    num_embeddings = len(embeddings)
    batch_size = embeddings[0].size(0)

    combined = torch.cat(embeddings, dim=0)  # Combine all embeddings
    sim_matrix = F.cosine_similarity(combined.unsqueeze(1), combined.unsqueeze(0), dim=2) / temperature
    sim_matrix_exp = torch.exp(sim_matrix)

    # Create a mask to exclude self-comparisons
    mask = torch.eye(num_embeddings * batch_size, device=combined.device).bool()
    #sim_matrix_exp.masked_fill_(mask, 0)
    sim_matrix_exp = sim_matrix_exp.masked_fill(mask, 0)
    # Sum of exp similarities for normalization
    sum_sim_exp = sim_matrix_exp.sum(dim=1, keepdim=True)

    # Loss calculation for each positive pair
    loss = 0
    for i in range(num_embeddings):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        positive_exp = sim_matrix_exp[start_idx:end_idx, start_idx:end_idx]
        positive_sum = positive_exp.sum(dim=1)
        loss += -torch.log(positive_sum / sum_sim_exp[start_idx:end_idx]).sum()

    return loss / (num_embeddings * batch_size)


def contrastive_difference(embeddings, temperature):
    total_loss = 0

    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j:
                # Normalize the embeddings
                left_normed = embeddings[i] / embeddings[i].norm(dim=-1, keepdim=True)
                right_normed = embeddings[j] / embeddings[j].norm(dim=-1, keepdim=True)

                # Compute the logits for both directions
                logits_left = torch.matmul(left_normed, right_normed.t()) * torch.exp(temperature)
                logits_right = torch.matmul(right_normed, left_normed.t()) * torch.exp(temperature)

                # Compute softmax probabilities
                prob_left = F.softmax(logits_left, dim=-1)
                prob_right = F.softmax(logits_right, dim=-1)

                # Create labels (identity matrix for matching pairs)
                labels = torch.eye(prob_left.size(0)).to(prob_left.device)

                # Compute the losses for both directions
                loss_left = F.cross_entropy(prob_left, labels, reduction='sum')
                loss_right = F.cross_entropy(prob_right, labels, reduction='sum')

                # Aggregate the losses
                total_loss += (loss_left + loss_right) / 2

    # Average the total loss
    average_loss = total_loss / (len(embeddings) * (len(embeddings) - 1))
    return average_loss / len(embeddings)


if __name__ == '__main__':
    # get the variabels from args
    source_model='gpt2-xl'
    model_dirs={'albert-xxlarge-v2':'autoencoder_recon_coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K_model_albert-xxlarge-v2_b_32_h_256_alpha_r_1e-05_id_7ea031b7d03c461abc8eb97e39395905',
                'bert-large-uncased-whole-word-masking':'autoencoder_recon_coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K_model_bert-large-uncased-whole-word-masking_b_32_h_256_alpha_r_1e-05_id_87eb37890d9b4d658ce923d138b229a7',
                'ctrl':'autoencoder_recon_coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K_model_ctrl_b_32_h_256_alpha_r_1e-05_id_33de19dbce2546f4a9f9b24982f778f1',
                'gpt2-xl':'autoencoder_recon_coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K_model_gpt2-xl_b_32_h_256_alpha_r_1e-05_id_254df48f0a014e78b15e3194dba70827',
                'roberta-base':'autoencoder_recon_coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K_model_roberta-base_b_32_h_256_alpha_r_1e-05_id_2003c233a1e54d9b85c973c1eb001bce',
                'xlm-mlm-en-2048':'autoencoder_recon_coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K_model_xlm-mlm-en-2048_b_32_h_256_alpha_r_1e-05_id_d55bfffcf9684664bd16a008ef9912a4',
                'xlnet-large-cased':'autoencoder_recon_coca_preprocessed_all_clean_100K_sample_1_2_ds_min_est_n_10K_model_xlnet-large-cased_b_32_h_256_alpha_r_1e-05_id_629206d7761049408a4550ce12c11496'}
    bottleneck_size=32
    hidden_size=256
    alpha_r=1e-5
# set the config
    learning_rate =  1e-4
    batch_size = 1024

    config = {"lr": learning_rate, "num_epochs": 1000, "batch_size": batch_size, 'optimizer': 'Adam',
                    'alpha': 1, 'beta': 0.00,"source_model":source_model,'bottleneck_size':bottleneck_size,'hidden_size':hidden_size,
              'alpha_r':alpha_r,'temperature':1,"lambda_recon":.3}

    gradient_accumulation_steps=1
    dataset_id = 'coca_preprocessed_all_clean_100K_sample_1_2_ds_max_est_n_10K'
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


    source_model_idx=int(''.join(([str(idx) if ext_obj.model_spec[idx]==config['source_model'] else '' for idx in range(len(ext_obj.model_spec))])))
    target_model_idx=list(set(range(len(ext_obj.model_spec)))-set([source_model_idx]))
    target_model_names=[ext_obj.model_spec[idx] for idx in target_model_idx]
    act_source=torch.tensor([x[0] for x in optim_obj.activations[source_model_idx]['activations']])
    act_target=[torch.tensor(np.stack([x[0] for x in optim_obj.activations[idx]['activations']])) for idx in target_model_idx]
    # make a list of act_source and act_target
    acts=[act_source]+act_target
    # make an ecnoder for act_1
    models_dataset = MultiModelDataset(acts)

    # make a list containing all the datasets
    # create train and test splits
    dataset_size = len(act_source)
    train_size = int(dataset_size * 0.9)  # Let's say we want 80% for training
    test_size = dataset_size - train_size
    train_indices, test_indices = random_split(range(dataset_size), [train_size, test_size])
    train_dataset=Subset(models_dataset,train_indices)
    test_dataset=Subset(models_dataset,test_indices)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # create the source autoencoder
    source_autoencoder = ModelAutoencoder(act_source.shape[1], hidden_size, bottleneck_size).to(device)
    # load the weights for source autoencoder
    source_autoencoder.load_state_dict(torch.load(Path(SAVE_DIR,'auto_encoding',model_dirs[config['source_model']],'model.pt'))['model'])
    # create the target autoencoder
    target_autoencoders = [ModelAutoencoder(act_target[idx].shape[1], hidden_size, bottleneck_size).to(device) for idx in range(len(act_target))]
    # load the weights for target autoencoder
    for idx in range(len(target_model_names)):
        target_autoencoders[idx].load_state_dict(torch.load(Path(SAVE_DIR,'auto_encoding',model_dirs[target_model_names[idx]],'model.pt'))['model'])
    # create a list of autoencoders paramets
    all_target_autoencoders=[auto_encoder.parameters() for auto_encoder in target_autoencoders]
    # create the second autoencoder
    optimizers_targets=[optim.Adam(auto_encoder,lr=learning_rate) for auto_encoder in all_target_autoencoders]
    optimizer_source = optim.Adam(source_autoencoder.parameters(), lr=learning_rate)
    #optimizer2= optim.Adam(model_2_autoencoder.parameters(),lr=learning_rate)
    mseloss = torch.nn.MSELoss(reduction='mean')
    lr_schedulers_targets =[ get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * config['num_epochs']) // gradient_accumulation_steps) for optimizer in optimizers_targets]
    lr_scheduler_source = get_cosine_schedule_with_warmup(optimizer=optimizer_source,
                                                            num_warmup_steps=0,
                                                            num_training_steps=(len(train_loader) * config['num_epochs']) // gradient_accumulation_steps)

    batch_count = 0
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config['batch_size'])
    accelerate = Accelerator(log_with="wandb",gradient_accumulation_steps=gradient_accumulation_steps)
    accelerate.init_trackers("autoencoding_contrastive", config=config,init_kwargs={"wandb": {
            "notes": "reconstruction loss",
            "entity": "evlab",
        }})
    random_idx=uuid.uuid4().hex
    accelerate.trackers[0].run.name = f'{config["source_model"]}_{bottleneck_size}_{hidden_size}_{alpha_r}_{random_idx[:5]}'
    source_autoencoder,target_autoencoders, optimizer_source,optimizers_targets, train_loader, test_loader,lr_scheduler_source,lr_schedulers_targets,mseloss = accelerate.prepare(
        source_autoencoder,target_autoencoders, optimizer_source,optimizers_targets, train_loader, test_loader,lr_scheduler_source,lr_schedulers_targets,mseloss)
    abs_step = 1
    wandb_tracker = accelerate.get_tracker("wandb")
    target_model_idx = 0
    #target_model_idx = np.random.choice(range(len(target_model_names)))
    optimizer_target = optimizers_targets[target_model_idx]
    target_autoencoder = target_autoencoders[target_model_idx]
    target_model_name = target_model_names[target_model_idx]

    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        example_ct = 0
        step_ct = 0
        target_model_idx = np.random.choice(range(len(target_model_names)))
        optimizer_target = optimizers_targets[target_model_idx]
        target_autoencoder = target_autoencoders[target_model_idx]
        target_model_name = target_model_names[target_model_idx]
        source_autoencoder.train()
        target_autoencoder.train()
        torch.cuda.empty_cache()
        for step, inputs in tqdm(enumerate(train_loader)):
            # Move data to the device (e.g., GPU) if available

            inputs_m_source = inputs['model_list'][0].to(device)
            # random select a target model
            #target_model_idx = np.random.choice(range(len(target_model_names)))

            inputs_m_target = inputs['model_list'][target_model_idx+1].to(device)

            latent_source, reconstructed_source = source_autoencoder(inputs_m_source)
            latent_target, reconstructed_target = target_autoencoder(inputs_m_target)
            activation_loss = config['alpha_r'] * torch.mean(torch.norm(latent_source,dim=1, p=1))

            contrastive = nt_xent_loss_multi([latent_source, latent_target], temperature=torch.tensor(config['temperature']))
            loss_1 = mseloss(inputs_m_source, reconstructed_source)
            loss_2 = mseloss(inputs_m_target, reconstructed_target)
            total_loss = config['lambda_recon']*(loss_1  + loss_2) + contrastive
            #loss = loss + activation_loss
            epoch_loss += total_loss.item()
            total_loss.backward()
            #optimizers[target_model_idx].step()
            #optimizers[0].step()

            #optimizers[target_model_idx].zero_grad()
            #optimizers[0].zero_grad()
            optimizer_source.step()
            optimizer_target.step()

            lr_scheduler_source.step()
            lr_schedulers_targets[target_model_idx].step()
            batch_count += len(inputs)
            accelerate.log({"training_loss": total_loss}, step=abs_step)
            accelerate.log({"loss_source": loss_1}, step=abs_step)
            accelerate.log({"loss_target": loss_2}, step=abs_step)
            accelerate.log({"train/train_loss": total_loss}, step=abs_step)
            accelerate.log({"train/contrastive_loss": contrastive}, step=abs_step)
            accelerate.log({"train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch},
                            step=abs_step)
            accelerate.log({"train/batch_count": batch_count}, step=abs_step)
            # save learning rate
            accelerate.log({"train/lr": optimizer_source.param_groups[0]['lr']}, step=abs_step)
            accelerate.log({"train/lr_target": optimizer_target.param_groups[0]['lr']}, step=abs_step)
            if abs_step % 100 == 0:
                source_autoencoder.eval()
                target_autoencoder.eval()
                torch.cuda.empty_cache()
                # random select a target model
               # target_model_idx = np.random.choice(range(len(target_model_names)))
                all_encoded1 = []
                all_encoded2 = []
                with torch.no_grad():
                    test_loss = 0.0
                    test_loss_target=0.0
                    test_loss_source=0.0
                    test_contrastive=0.0
                    embeddings=[]
                    for inputs in test_loader:
                        inputs_m_source = inputs['model_list'][0].to(device)
                        inputs_m_target = inputs['model_list'][target_model_idx + 1].to(device)
                        latent_source, reconstructed_source = source_autoencoder(inputs_m_source)
                        latent_target, reconstructed_target = target_autoencoder(inputs_m_target)

                        all_encoded1.append(latent_source)
                        all_encoded2.append(latent_target)
                        contrastive = nt_xent_loss_multi([latent_source, latent_target],
                                                         temperature=torch.tensor(config['temperature']))
                        loss_1 = mseloss(inputs_m_source, reconstructed_source)
                        loss_2 = mseloss(inputs_m_target, reconstructed_target)
                        total_loss = config['lambda_recon']*(loss_1+loss_2) + contrastive
                        test_loss += total_loss.item()
                        test_contrastive+=contrastive.item()
                        test_loss_source+=loss_1.item()
                        test_loss_target+=loss_2.item()
                        # Compute the loss

                accelerate.log({"validation/valid_loss": test_loss}, step=abs_step)
                accelerate.log({"validation/contrastive_loss": test_contrastive}, step=abs_step)
                accelerate.log({"validation/loss_source": test_loss_source}, step=abs_step)
                accelerate.log({"validation/loss_target": test_loss_target}, step=abs_step)
                # do a 2 dimensional PCA
                embeddings_source= torch.concat(all_encoded1)
                # compute pairwise cosine similarity between points in the latent space
                # pairwise_source=pairwise_cosine_similarity(embeddings_source)
                # pairwise_target=pairwise_cosine_similarity(torch.concat(all_encoded2))
                # pil_image = to_pil_image(pairwise_source, mode='L')
                # image = wandb.Image(pil_image, caption=f'{config["source_model"]}')
                # accelerate.log({"encoding_source": image}, step=abs_step)
                # accelerate.log({"sum_target": torch.mean(pairwise_target)}, step=abs_step)
                # accelerate.log({"sum_source": torch.mean(pairwise_source)}, step=abs_step)
                #
                # pil_image = to_pil_image(pairwise_target, mode='L')
                # image = wandb.Image(pil_image, caption=f"{target_model_name}")
                # accelerate.log({"encoding_target": image}, step=abs_step)
                #
                U,S,V = torch.pca_lowrank(embeddings_source)
                U=U[:,:2]
                U = torch.split(U, U.shape[0])[0]
                # make it a list of lists
                U = [x.tolist() for x in U]
                wandb_tracker.log({"embeddings_source": wandb.Table(columns=["pc1", "pc2"], data=U)},step=abs_step)

                embeddings_target = torch.concat(all_encoded2)
                U, S, V = torch.pca_lowrank(embeddings_target)
                U = U[:, :2]
                U = torch.split(U, U.shape[0])[0]
                # make it a list of lists
                U = [x.tolist() for x in U]
                wandb_tracker.log({f"embeddings_{target_model_name}": wandb.Table(columns=["pc1", "pc2"], data=U)},step=abs_step)


                source_autoencoder.train()
                target_autoencoder.train()
                torch.cuda.empty_cache()
            abs_step += 1

        #print("Epoch [{}/{}], Loss: {:.2f}, Test Loss: {:.2f}".format(epoch + 1, config['num_epochs'], epoch_loss, test_loss))
    # save model
    random_idx=uuid.uuid4().hex

    # save_id=f'autoencoder_recon_{dataset_id}_model_{config["model_name"]}_b_{config["bottleneck_size"]}_h_{config["hidden_size"]}_alpha_r_{config["alpha_r"]}_id_{random_idx}'
    # save_dir = Path(SAVE_DIR,'auto_encoding', save_id)
    # # if path does not exist, create it
    # save_dir.mkdir(parents=True, exist_ok=True)
    # # save the model using accelerate
    # accelerate.save({'model': model_autoencoder.state_dict(), 'optimizer': optimizer.state_dict()}, Path(save_dir, 'model.pt'))
    # # save the config
    # save_obj(config, Path(save_dir, 'config.pkl'))




    #%% test how ds_min and ds_max are projected in the latent space
    dataset_id = 'ds_parametric'
    stim_type= 'textNoPeriod'
    layer_id=44
    sample_extractor_id = f'group={config["source_model"]}_layers-dataset={dataset_id}_{stim_type}-activation-bench=None-ave=False'
    ds_param_obj = extract_pool[sample_extractor_id]()
    ds_param_obj.load_dataset()

    model_activation_name = f"{ds_param_obj.dataset}_{ds_param_obj.stim_type}_{ds_param_obj.model_spec[layer_id]}_layer_{layer_id}_{ds_param_obj.extract_name}_ave_{ds_param_obj.average_sentence}.pkl"
    ds_parametric_layer = load_obj(os.path.join(SAVE_DIR, model_activation_name))
    ds_stim=ds_param_obj.data_
    # select part of the data framne that have sentence_type =='ds_min'
    ds_min_loc = [idx for idx, x in enumerate(ds_stim['sent_type']) if x == 'ds_min']
    ds_max_loc=[idx for idx, x in enumerate(ds_stim['sent_type']) if x == 'ds_max']
    # get the sentence id
    ds_min_id = np.unique([ds_stim['sent_id'][x] for x in ds_min_loc])
    ds_max_id = np.unique([ds_stim['sent_id'][x] for x in ds_max_loc])

    # go through the ds_parametric layer and select elements that have the same sentence id
    ds_min_loc_layer = [x for idx, x in enumerate(ds_parametric_layer) if x[2] in ds_min_id]
    ds_max_loc_layer = [x for idx, x in enumerate(ds_parametric_layer) if x[2] in ds_max_id]
    # do the same thin
    # get the activations
    ds_min_act = [x[0] for x in ds_min_loc_layer]
    ds_max_act = [x[0] for x in ds_max_loc_layer]

    # get encoded activations in eval mode
    source_autoencoder.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        ds_min_encoded = source_autoencoder.encoder(torch.tensor(ds_min_act).to(device))
        ds_max_encoded = source_autoencoder.encoder(torch.tensor(ds_max_act).to(device))
        #ds_max_encoded_target = target_autoencoder.encoder(torch.tensor(ds_max_act).to(device))
        #ds_min_encoded_target = target_autoencoder.encoder(torch.tensor(ds_min_act).to(device))
        # get encoder for act_source
        act_source_encoded = source_autoencoder.encoder(act_source.to(device))
        # do the same for act_target

    U, S, V = torch.pca_lowrank(act_source_encoded)
    U = U[:, :2]
    # mutiply ds_min_encoded and ds_max_encoded by by V[:,:2]
    ds_min_pca = torch.matmul(ds_min_encoded, V[:, :2])
    ds_max_pca = torch.matmul(ds_max_encoded, V[:, :2])
    # devide them by S[:2]
    ds_min_pca = ds_min_pca / S[:2]
    ds_max_pca = ds_max_pca / S[:2]
    # combine ds_min_pca and ds_max_pca using torch
    x_pca = torch.cat((ds_min_pca, ds_max_pca), dim=0)

    x_pca = torch.split(x_pca, x_pca.shape[0])[0]
    # make it a list of lists
    x_pca = [x.tolist() for x in x_pca]
    labels = torch.concat((torch.tensor(np.repeat(0, ds_min_pca.shape[0]))
                           , torch.tensor(np.repeat(1, ds_max_pca.shape[0]))), axis=0).tolist()
    x_pca = [item1 + [item2] for item1, item2 in zip(x_pca, labels)]

    wandb_tracker.log({"embedding_ds": wandb.Table(columns=["pc1", "pc2","label"], data=x_pca)}, step=abs_step)

    accelerate.end_training()
    torch.cuda.empty_cache()
    # do a pca on act_source_encoded
    # pca_src = PCA(n_components=2,whiten=False)
    # scaler = StandardScaler()
    # source_data_standardized = scaler.fit_transform(act_source_encoded.detach().cpu().numpy())
    # ds_min_standardized = scaler.transform(ds_min_encoded.detach().cpu().numpy())
    # ds_max_standardized = scaler.transform(ds_max_encoded.detach().cpu().numpy())
    # pca_src.fit(source_data_standardized)
    # # transform ds_min_encoded and ds_max_encoded
    # ds_min_encoded_pca = pca_src.fit_transform(ds_min_standardized)
    # ds_max_encoded_pca = pca_src.fit_transform(ds_max_standardized)
    # source_encoded_pca=pca_src.fit_transform(source_data_standardized)
    #
    #
    # x_pca = np.concatenate((ds_min_pca.detach().cpu().numpy(), ds_max_pca.detach().cpu().numpy()), axis=0)
    #     # create labels max and min
    #     # create a df with x_pca and labels
    # df = pd.DataFrame(x_pca, columns=['x', 'y'])
    # df['labels'] = labels
    # g = sns.jointplot(
    #         data=df,
    #         x="x", y="y", hue="labels",
    #         kind="scatter",s =5 )
    # ax = g.ax_joint
    # # plot x_pca_source in the background as a scatter using sns
    # df = pd.DataFrame(source_encoded_pca, columns=['x', 'y'])
    # df['labels'] = 'source'
    # sns.scatterplot(data=df, x="x", y="y", hue="labels", ax=ax, alpha=0.1, s=5)
    # plt.show()
    #
