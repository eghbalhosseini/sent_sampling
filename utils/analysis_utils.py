import numpy as np
import pickle


def flatten_rdm(rdm_mat):
    A=np.triu(np.ones(np.shape(rdm_mat)),k=1)
    up_tria_mat=np.where(A==0,np.nan,A)
    rdm_tria_up=np.multiply(rdm_mat,up_tria_mat)
    rdm_tria_up_flat=rdm_tria_up.flatten()
    rdm_tria_up_flat=rdm_tria_up_flat[np.logical_not(np.isnan(rdm_tria_up_flat))]
    return rdm_tria_up_flat




