
import os
import pandas as pd
from tqdm import tqdm
from sent_sampling.utils.data_utils import RESULTS_DIR,ANALYZE_DIR,LEX_PATH_SET
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool, pt_create_corr_rdm_short
import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, ks_2samp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gpu_object_function(precomputed_matrices, S, device):
    S = torch.tensor(S, dtype=torch.long, device=device)  # Ensure S is a tensor with gradient support
    pairs = torch.combinations(S, with_replacement=False)

    # Use list comprehension to gather the XY_corr values
    X = [XY_corr[pairs[:, 0], pairs[:, 1]].to(device) for XY_corr in precomputed_matrices]

    # Stack the list into a tensor and transpose
    X = torch.stack(X).to(device)
    X = torch.transpose(X, 1, 0)

    # Ensure the shape for further processing
    if X.shape[1] < X.shape[0]:
        X = torch.transpose(X, 1, 0)

    assert X.shape[1] > X.shape[0]

    # Center the data by subtracting the mean
    X = X - X.mean(dim=1, keepdim=True)

    # Normalize the data
    X = F.normalize(X, dim=1)

    # Create an identity matrix
    identity = torch.eye(X.shape[0], device=device)

    # Compute the correlation matrix and adjust it
    XY_corr = identity - torch.mm(X, X.T)
    XY_corr = torch.triu(XY_corr, diagonal=1)

    # Clamp the values to ensure non-negative distances
    d_mat = torch.clamp(XY_corr, 0.0, float('inf'))

    # Compute the correction factor and the mean distance
    n1 = d_mat.shape[1]
    correction = n1 * n1 / (n1 * (n1 - 1) / 2)
    d_val = correction * d_mat.mean()

    return d_val




if __name__ == '__main__':
    extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_minus_ev_sentences_textNoPeriod-activation-bench=None-ave=False'

    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    ext_obj()
    # extract ev sentences
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))
    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]
    sentences = [x[1] for x in ext_obj.model_group_act[0]['activations']]
    # find location of ev sentences in sentences
    ev_sentence_ids = []
    for ev_sent in ev_sentences:
        # remove the period
        ev_sent = ev_sent[:-1]
        ev_sentence_ids.append(sentences.index(ev_sent))

    optim_id='coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples=200-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    optim_obj = optim_pool[optim_id]()

    optim_obj.load_extractor(ext_obj)

    low_resolution= False
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=False, preload=False,
                                                 save_results=False)



    S=ev_sentence_ids
    # put the XY_corr on device
    for idx in range(len(optim_obj.XY_corr_list)):
        optim_obj.XY_corr_list[idx] = optim_obj.XY_corr_list[idx].to(device)

    precomputed_matrices = optim_obj.XY_corr_list
    gpu_object_function(optim_obj.XY_corr_list,S,device)
    optim_obj.gpu_object_function_debug(S)

    probs = torch.nn.Parameter(torch.ones(optim_obj.N_S) / torch.ones(optim_obj.N_S), requires_grad=True)
    optimizer = torch.optim.Adam([probs], lr=0.01)
    indices=S
    for _ in tqdm(range(1000)):  # Number of optimization steps
        optimizer.zero_grad()

        # Sample 20 indices according to the probabilities
        S = torch.multinomial(probs, optim_obj.N_s, replacement=False)

        # Compute the objective
        loss = gpu_object_function(optim_obj.XY_corr_list,S,device)

        # Ensure the loss requires grad
        # Compute gradients and update probabilities
        loss.backward()
        optimizer.step()

        # Ensure probabilities remain valid
        with torch.no_grad():
            probs.clamp_(0, 1)
            probs.div_(probs.sum())

    # Select final set of 20 samples
    final_indices = torch.multinomial(probs, 20, replacement=False)
    final_samples = final_indices  # These are the indices of the selected samples

    print("Selected sample indices:", final_samples)