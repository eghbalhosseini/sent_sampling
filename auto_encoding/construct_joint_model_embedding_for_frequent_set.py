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
import pandas as pd
import torch
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import math
import torch
import torch.nn as nn
from auto_encoding.encoding_utils import corr_coeff, similarity_loss, normalize, CustomLayer
from auto_encoding.sweep_joint_model_embedding_for_frequent_set import build_dataset,build_optimizer,build_network
class Encoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_bottleneck):
        super(Encoder, self).__init__()
        #self.weight_matrices = torch.stack([torch.randn(650, 256) for _ in range(7)])
        self.fc1_hidden = nn.Linear(n_features,n_hidden)
        self.fc2_botlneck = nn.Linear(n_hidden, n_bottleneck)
    def forward(self, input_data):
        x = self.fc1_hidden(input_data)
        encoded = (self.fc2_botlneck(x))
        return encoded

class Decoder(torch.nn.Module):
    def __init__(self,n_features,n_hidden):
        super(Decoder, self).__init__()
        self.fc1 = CustomLayer(n_channels=7,n_features=n_features,n_hidden=n_hidden)
        # add a dropout layer
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, encoded):
        x=self.dropout(encoded)
        x=self.fc1(x)
        return x

class JointEmbedding(nn.Module):
    def __init__(self, input_size, encoder_h, bottleneck_size,output_size,theta_size=32):
        super(JointEmbedding, self).__init__()
        self.encoder = Encoder(input_size,encoder_h,bottleneck_size)
        self.decoder = Decoder(bottleneck_size,output_size)
        self.theta = torch.nn.Parameter(torch.zeros( 7,theta_size), requires_grad=True)
    def forward(self, inputs):
        # concatenate theta to inputs
        theta_mod = self.theta.repeat(inputs.shape[0], 1, 1).to(device)
        inputs = torch.cat((inputs, theta_mod), dim=2)
        encoded = self.encoder(inputs)
        encoded = encoded.permute(0, 2, 1)
        decoded = self.decoder(encoded)
        return encoded.permute(0,2,1), decoded


if __name__ == '__main__':

    config = {
                   "epochs": 1000,
                   "batch_size": 128,
                   "lr":0.0002603876064984116,
                   "hidden_size":512,
                   "bottleneck_size":64,
                   "optimizer":"adam",
                   'alpha_r': 0.00015641566751182945,
                   "loss_mode":'SIM',
                    'p_dropout':0.1,
                   'extract_id':'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False',
                   'activation_loss':False,
                    'normalize':True,
                    'theta_size':0
               }

    #%% compute the pca
    n_components=650
    train_loader, test_loader = build_dataset(extract_id=config['extract_id'], n_components=n_components,
                                              batch_size=config['batch_size'], normalize=config['normalize'])
    model = build_network(650, config['hidden_size'], config['bottleneck_size'], 650, config['theta_size'], p_dropout=config['p_dropout'])

    optimizer = build_optimizer(model, config['optimizer'], learning_rate=config['lr'])

    mseloss = torch.nn.MSELoss(reduction='mean')
    # Training loop
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
            print("Epoch [{}/{}], Loss: {:.4f}, Test Loss: {:.4f}".format(epoch + 1, config['epochs'], epoch_loss,
                                                                          test_loss))
        # Test

    model_pca_loads=load_obj(os.path.join(ANALYZE_DIR,'model_pca_n_comp_650.pkl'))
    modle_names=[x['model_name'] for x in model_pca_loads]
    model_sh = ['RoBERTa', 'XLNet-L', 'BERT-L', 'XLM', 'GPT2-XL', 'ALBERT-XXL', 'CTRL']
    input_data_min=[torch.tensor(x['act_min']) for x in model_pca_loads]
    input_data_max=[torch.tensor(x['act_max']) for x in model_pca_loads]
    input_data_rand=[torch.tensor(x['act_rand']) for x in model_pca_loads]
    # normalize the data
    input_data_min = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data_min]
    input_data_max = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data_max]
    input_data_rand = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data_rand]
    input_data_min = torch.stack(input_data_min).to(device).permute(1,0,2).to(torch.float32)
    input_data_max = torch.stack(input_data_max).to(device).permute(1,0,2).to(torch.float32)
    input_data_rand = torch.stack(input_data_rand).to(device).permute(1,0,2).to(torch.float32)
    # reshape input_data_min to have the shape n_samples x n_features x n_models




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

    theta_corr=torch.corrcoef(model.theta.data)
    # plot it as an image
    plt.imshow(theta_corr.detach().cpu().numpy())
    plt.colorbar()
    plt.show()
    # put model id as x and y ticks
    plt.imshow(theta_corr.detach().cpu().numpy())
    plt.colorbar()
    plt.xticks(np.arange(7),model_sh,rotation=45)
    plt.yticks(np.arange(7),model_sh,rotation=45)
    plt.show()

    with torch.no_grad():
        encoded_train= model(train_data.to(device).to(torch.float32))

    sns.set_theme(style="ticks")
    for i in range(7):
        # fig, ax = plt.subplots()
        x = input_data_max[:, :, i].detach().cpu().numpy()
        pca = PCA(n_components=5)
        x_pca_max = pca.fit_transform(x)[:, :2]
        x = input_data_min[:, :, i].detach().cpu().numpy()
        x_pca_min = pca.fit_transform(x)[:, :2]
        # make a
        # plot the pca
        # concat x_pca_max and x_pca_min

        x_pca = np.concatenate((x_pca_min, x_pca_max), axis=0)
        # create labels max and min
        labels = np.concatenate((np.repeat('min', x_pca_min.shape[0]), np.repeat('max', x_pca_max.shape[0])), axis=0)
        # create a df with x_pca and labels
        df = pd.DataFrame(x_pca, columns=['x', 'y'])
        df['labels'] = labels
        g = sns.jointplot(
            data=df,
            x="x", y="y", hue="labels",
            kind="scatter")
        # add title
        plt.title(f'{model_sh[i]}')
        plt.show()
        g.savefig(os.path.join(ANALYZE_DIR,f'{model_sh[i]}_ds_min_ds_max_joint_embedding_pca.png'))
        #save eps
        g.savefig(os.path.join(ANALYZE_DIR,f'{model_sh[i]}_ds_min_ds_max_joint_embedding_pca.eps'), format='eps')

    sns.set_theme(style="ticks")
    for i in range(7):
        # fig, ax = plt.subplots()
        x_train = encoded_train[0][:, :, i]
        pca = PCA(n_components=2)
        x_pca_train = pca.fit(x_train.detach().cpu().numpy())
        x = encoded_max[:, :, i].detach().cpu().numpy()
        x_max = pca.transform(x)[:, :2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_min = pca.transform(x)[:, :2]
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
        # ax.axvline(x_min.mean(axis=0)[0],color='b',linestyle='--',zorder=1)
        # ax.axvline(x_max.mean(axis=0)[0],color='r',linestyle='--',zorder=1)
        # ax.axhline(x_min.mean(axis=0)[1],color='b',linestyle='--',zorder=1)
        # ax.axhline(x_max.mean(axis=0)[1],color='r',linestyle='--',zorder=1)
        plt.title(f'{model_sh[i]}\n pca train')
        plt.show()
        # g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_shared_embedding_train_pca.png'))
        # # save eps
        # g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_shared_embedding_train_pca.eps'),
        #           format='eps')

    sns.set_theme(style="ticks")
    for i in range(7):
        x = encoded_max[:, :, i].detach().cpu().numpy()
        x_max = x[:, :2]
        x = encoded_min[:, :, i].detach().cpu().numpy()
        x_min = x[:, :2]
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
        #g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_joint_embedding.png'))
        # save eps
        #g.savefig(os.path.join(ANALYZE_DIR, f'{model_sh[i]}_ds_min_ds_max_joint_embedding.eps'),
        #          format='eps')
        plt.show()