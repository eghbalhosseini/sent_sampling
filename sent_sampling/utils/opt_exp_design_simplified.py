import time
import numpy as np
import logging
import torch
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import uuid
import os
n_parallel_swaps=10
LOG_BASE = np.e
EPSILON = 1e-16
PRECISION = 1e-200
SAVE_DIR='/'

'''###################################################
Functionality for sampling by swapping  
###################################################'''

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
ch.setFormatter(formatter)


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger


def create_rdm(patterns, vec=False, distance='correlation'):
    rdm_vec = pdist(patterns, distance)
    if not vec:
        return squareform(rdm_vec)
    return rdm_vec


def second_order_rdm(patterns_list, vec=False, distance='correlation'):
    rdm_vecs = np.array([create_rdm(patterns, True, distance) for patterns in patterns_list])
    rdm2_vec = pdist(rdm_vecs, distance)
    if not vec:
        return squareform(rdm2_vec)
    return rdm2_vec

def get_machine_identifier():
    # Get the MAC address as a UUID
    return uuid.UUID(int=uuid.getnode())



LOGGER = get_logger('OPT-EXP-DSGN')
def swap(X, x_remove, x_add):
    X_swap = X + [x_add]
    X_swap.remove(x_remove)
    return X_swap


def swapi(A, B, a_ind, b_ind):
    a, b = A[a_ind], B[b_ind]
    A[a_ind] = b
    B[b_ind] = a



def coordinate_ascent_eh(N, n, objective_function=None, n_init=1, n_iter=200,S_init=None):
    """
    A generic coordinate ascent algorithm for optimizing sentence selection with respect to a given objective function.
    The default objective function is the model discriminability D which is based on the 2nd order RDMs.
    :param N: Total number of sentences
    :param n: Stimulus set size (n < N)
    :param objective_function: a function f(S), where S subset of range(N) with size n. The algorithm searches for
    S^* = argmax f(S), and will converge at a local optimum.
    :param n_init: Number of times to run the algorithm with different random initialization. The final results will be
    the best S of the n_init runs.
    :param n_iter: Maximal number of iterations in each run. The algorithm will stop after n_iter even if the solution
    did not converge to a local optimum.
    :return:
    S_best, the best performing set
    f_best, the score of the best performing set, that is f_best = f(S_best)
    """
    n_out = N - n
    S_full = set(np.arange(N))
    S_best = []
    f_best = 0
    for t_init in range(n_init):
        if S_init is not None :
            S=S_init
        else:
            S = list(np.random.choice(np.arange(N), n, replace=False))

        S_out = list(S_full.difference(set(S)))
        fS = objective_function(S)
        fS_loop_start = fS
        changed = True
        t = 0
        LOGGER.info('===> Starting init %d, initial f(S) = %.5f' % (t_init, fS))
        while t < n_iter and changed:
            changed = False
            t += 1
            time_start = time.perf_counter()
            # start with a random selection from s
            si_list=np.random.choice(S, size=n, replace=False)
            # go one by one through element in S and replace them with so
            for si_idx,si in enumerate(si_list):
                so_list=np.random.choice(S_out, size=n_out, replace=False)
                so_idx=0
                keep_swapping=True
                while keep_swapping:
                    S_test=swap(S,si,so_list[so_idx])
                    #time_start = time.perf_counter()
                    f_swap = objective_function(S_test)
                    #time_elapsed = (time.perf_counter() - time_start)
                    if f_swap > fS:
                        fS = f_swap
                        fS_loop = fS
                        S = S_test
                        S_out = swap(S_out, so_list[so_idx], si)

                        LOGGER.info('[%d/%d] [t = %d] id = %d, %d to %d after %d swaps,  f(S) = %.5f' % (t_init + 1, n_init, t,si_idx,si,so_list[so_idx],so_idx, fS))
                        keep_swapping=False
                        changed=True
                    else:
                        keep_swapping = True
                        so_idx += 1
                    if so_idx==len(so_list):
                        LOGGER.info('[%d/%d] [t = %d] id = %d f(s) %d didnt change after all %d swaps,  f(S) = %.5f' % (t_init + 1, n_init, t,si_idx,si, so_idx, fS))
                        keep_swapping=False
            if not changed:
                LOGGER.info('[%d/%d] [t = %d] converged to a local optimum,  f(S) = %.5f' % (t_init + 1, n_init, t, fS))
            time_elapsed = (time.perf_counter() - time_start)
            LOGGER.info('loop %d done, loop total time %f, f(S) = %.5f' % (t, time_elapsed, fS))
            if t == n_iter:
                LOGGER.info('[%d/%d] max iteration %d reached, f(S) = %.5f' % (t_init + 1, n_init, n_iter, fS))
        if fS > f_best:
            S_best = S
            f_best = fS
    LOGGER.info('Done, opt f(S) = %.5f' % f_best)
    return S_best, f_best

'''###################################################
Functionality for optimzation over n models 
###################################################'''
def Distance():
    """ds"""
    NotImplementedError

def minus_Distance():
    """2-ds"""
    NotImplementedError

def corrcoef_metric(act):
    """corrcoef"""
    metric_val = 1 - torch.corrcoef(act)
    return metric_val

@torch.no_grad()
def pt_create_corr_rdm_short(X,Y=None,vec=False,device=None):
    # note currently it is compeletely ignoring Y
    X=(X-X.mean(axis=1,keepdim=True))
    X=torch.nn.functional.normalize(X)
    if Y is not None:
            Y=(Y-Y.mean(axis=1,keepdim=True))
            Y=torch.nn.functional.normalize(Y)
    else:
        Y=X
    XY_corr=torch.tensor(1,device=X.device,dtype = float,requires_grad=False)-torch.mm(X,torch.transpose(Y,1,0))
    XY_corr=torch.triu(XY_corr,diagonal=1)
    if vec:
        return torch.clamp(torch.reshape(XY_corr,(1,-1)), 0.0, np.inf)
    return torch.clamp(XY_corr, 0.0, np.inf)
class optim:
    def __init__(self, n_init=3, n_iter=300,N_s=50, objective_function=Distance,
                 optim_algorithm=None,N_S=None,activations=None,run_gpu=False,device=None):
        self.n_iter=n_iter
        self.n_init=n_init
        self.N_S = N_S
        self.N_s=N_s
        self.activations=activations
        self.objective_function=objective_function
        self.optim_algorithm=optim_algorithm
        self.device=device
        self.run_gpu=run_gpu
        self.s_init=None

    def precompute_corr_rdm_on_gpu(self,dtype=torch.float32):
        self.XY_corr_list=[]
        for idx, act_ in tqdm(enumerate(self.activations)):
                act = torch.tensor(act_, dtype=dtype, device=self.device,requires_grad=False)
                XY_corr=corrcoef_metric(act)
                self.XY_corr_list.append(XY_corr.to(self.device))
                del act
                del act_
                del XY_corr
                torch.cuda.empty_cache()

        # double check target device allocation.
        self.XY_corr_list=[x.to(self.device) for x in self.XY_corr_list]
        if self.run_gpu:
            # delete activations from gpu if it exists in self
            self.activations = None
            torch.cuda.empty_cache()
    def gpu_object_function_ds(self,S):
        samples=torch.tensor(S, dtype = torch.long, device = self.device)
        pairs = torch.combinations(samples, with_replacement=False).to('cpu')
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]].to(self.device) for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample).to(self.device)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        d_val_mean=d_val.cpu().numpy().mean()
        # do a version with std reductions too
        mdl_pairs = torch.combinations(torch.tensor(np.arange(d_mat.shape[0])), with_replacement=False)
        d_val_std=torch.std(d_mat[mdl_pairs[:,0],mdl_pairs[:,1]]).cpu().numpy()
        d_optim=d_val_mean #-.2*d_val_std
        return d_optim

    def gpu_object_function_minus_ds(self,S):
        samples=torch.tensor(S, dtype = torch.long, device = self.device)
        pairs = torch.combinations(samples, with_replacement=False).to('cpu')
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]].to(self.device) for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample).to(self.device)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        d_val_mean=d_val.cpu().numpy().mean()
        # do a version with std reductions too
        mdl_pairs = torch.combinations(torch.tensor(np.arange(d_mat.shape[0])), with_replacement=False)
        d_val_std=torch.std(d_mat[mdl_pairs[:,0],mdl_pairs[:,1]]).cpu().numpy()
        d_optim=2-d_val_mean #-.2*d_val_std
        return d_optim

    def __call__(self,*args, **kwargs):
        if self.objective_function.__doc__ == 'ds':
            self.objective = self.gpu_object_function_ds
        elif self.objective_function.__doc__ == '2-ds':
            self.objective = self.gpu_object_function_minus_ds
        S_opt_d, DS_opt_d = self.optim_algorithm(N=self.N_S, n=self.N_s, objective_function=self.objective, n_init=self.n_init,
                                                     n_iter=self.n_iter,S_init=self.s_init)

        self.S_opt_d=S_opt_d
        self.DS_opt_d=DS_opt_d

        return S_opt_d, DS_opt_d

if __name__ == '__main__':
    # create a set of random matrxi with size 100 by 500 and random
    # seed 1234
    torch.manual_seed(1234)
    activations = [torch.rand(100, 500) for _ in range(5)]
    # create a optim object
    opt = optim(n_init=1, n_iter=300,N_S=100, N_s=20, objective_function=Distance, optim_algorithm=coordinate_ascent_eh,activations=activations,run_gpu=True,device='cpu')
    opt.precompute_corr_rdm_on_gpu()
    S_opt, DS_opt = opt()