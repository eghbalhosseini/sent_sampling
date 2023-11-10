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
from auto_encoding.encoding_utils import corr_coeff, similarity_loss, normalize, CustomLayer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer
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

    #%% compute the pca
    n_components=650
    extract_id = 'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False'
    model_pca_path=Path(ANALYZE_DIR,f'{extract_id}_pca_n_comp_{n_components}.pkl')
    train_pca_loads=pd.read_pickle(model_pca_path.__str__())
    #
    standard_scaler = StandardScaler()

    # create a plot with the number of models
    # get model names from ext_obj.model_group_act
    input_size = n_components
    output_size=n_components
    hidden_size = 512
    bottleneck_size = 64
    theta_size=0
    num_epochs = 1000
    learning_rate = 0.00047670875475033873
    batch_size = 128
    alpha_activation=0.000909658634735287
    input_data=[torch.tensor(x['act']) for x in train_pca_loads]
    input_data = [torch.tensor(standard_scaler.fit_transform(X)) for X in input_data]
    input_data_shape=[x.shape[1] for x in input_data]
    train_data = torch.stack(input_data, dim=1).to(torch.double)
    # switch the dims to be batch x number of models
    train_size = int(0.9 * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
    print("Train set size:", len(train_dataset))
    print("Test set size:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = JointEmbedding(input_size+theta_size, hidden_size, bottleneck_size,output_size,theta_size=theta_size).to(device)


    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    mseloss = torch.nn.MSELoss(reduction='mean')
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        # Training
        model.train()
        for inputs in train_loader:
            inputs = inputs.to(device)
            # make it double
            # make it a float tensor
            inputs = inputs.to(torch.float32)
            encoded,decoded=model(inputs)
            # Zero the gradients
            similarities = similarity_loss(inputs, decoded)
            mse_val=mseloss(decoded,inputs)
            XY_loss = torch.sum(similarities) + torch.var(similarities)
            activation_loss = alpha_activation * torch.norm(encoded, p=2)
            loss = activation_loss+ XY_loss            # Backward pass and optimization
            #loss= XY_loss
            #loss = mse_val
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
                inputs = inputs.to(torch.float32)
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
    #normalize the data



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