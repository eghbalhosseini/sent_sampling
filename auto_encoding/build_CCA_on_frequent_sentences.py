import numpy as np
from scipy.linalg import block_diag, eigh
from pathlib import Path
import pandas as pd
import os
import numpy as np
import sys
from pathlib import Path
import torch
import seaborn as sns
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
import matplotlib.pyplot as plt
def multiview_cca(data_views, num_components):
    """
    Perform multiview CCA on the given data views.

    :param data_views: A list of data matrices, each representing a view.
                       All matrices must have the same number of rows (samples).
    :param num_components: The number of components to compute for each view.
    :return: A list of projection matrices, one for each view.
    """
    num_views = len(data_views)
    num_samples = data_views[0].shape[0]

    # Center the data for each view
    centered_views = [X - torch.mean(X, axis=0) for X in data_views]
    total_covariance = np.cov(np.hstack(centered_views).T)

    # Sw would be a block diagonal matrix with each block being the covariance matrix of a view
    Sw = block_diag(*[np.cov(X.T) for X in centered_views])

    Sb = total_covariance - Sw

    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(Sb, Sw)

    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Split the eigenvectors to get the projection matrices for each view
    projection_matrices = np.split(eigenvectors, num_views)

    # Take only the top 'num_components' components for each view
    projection_matrices = [W[:, :num_components] for W in projection_matrices]
    projected_views = [np.dot(X, W) for X, W in zip(centered_views, projection_matrices)]

    return projection_matrices, projected_views,centered_views


if __name__ == '__main__':
    n_components = 650
    extract_id = 'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_100K_sample_1_estim_ds_min_textNoPeriod-activation-bench=None-ave=False'
    model_pca_path = Path(ANALYZE_DIR, f'{extract_id}_pca_n_comp_{n_components}.pkl')
    train_pca_loads = pd.read_pickle(model_pca_path.__str__())
    input_data = [torch.tensor(x['act']) for x in train_pca_loads]
    projection_matrices, projected_views, centered_views = multiview_cca(input_data, 2)
    # start a figure
    model_pca_loads = load_obj(os.path.join(ANALYZE_DIR, 'model_pca_n_comp_650.pkl'))
    model_sh = ['RoBERTa', 'XLNet-L', 'BERT-L', 'XLM', 'GPT2-XL', 'ALBERT-XXL', 'CTRL']
    input_data_min = [torch.tensor(x['act_min']) for x in model_pca_loads]
    input_data_max = [torch.tensor(x['act_max']) for x in model_pca_loads]
    input_data_rand = [torch.tensor(x['act_rand']) for x in model_pca_loads]

    data_min_centr= [x - torch.mean(input_data[idx], axis=0)  for idx,x in enumerate(input_data_min)]
    data_max_centr= [x - torch.mean(input_data[idx], axis=0)  for idx,x in enumerate(input_data_max)]
    data_rand_centr= [x - torch.mean(input_data[idx], axis=0)  for idx,x in enumerate(input_data_rand)]

    # project the data using projection matrices
    projected_views_min = [np.dot(x, W) for x, W in zip(data_min_centr, projection_matrices)]
    projected_views_max = [np.dot(x, W) for x, W in zip(data_max_centr, projection_matrices)]
    projected_views_rand = [np.dot(x, W) for x, W in zip(data_rand_centr, projection_matrices)]



    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]

    # create plot with 3 rows and 3 columns
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # plot the data
    for idx, ax in enumerate(axs.flat):
        if idx < 7:
            # plot the original data
            ax.scatter(projected_views[idx][:, 0], projected_views[idx][:, 1], color=colors[1], alpha=0.5,s=5)
            ax.scatter(projected_views_min[idx][:, 0], projected_views_min[idx][:, 1], color=colors[0],edgecolor='k', alpha=1,s=8)
            ax.scatter(projected_views_max[idx][:, 0], projected_views_max[idx][:, 1], color=colors[2],edgecolor='k', alpha=1,s=8)
            #ax.scatter(projected_views_rand[idx][:, 0], projected_views_rand[idx][:, 1], color=colors[2], alpha=0.5)
            ax.set_title(model_sh[idx])
        else:
            ax.axis('off')

    fig.show()