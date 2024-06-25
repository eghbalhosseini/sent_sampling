import numpy as np
#import tools
from tdict import Tdict
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
from scipy.stats import entropy
import warnings
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import pdist, squareform
import multiprocessing as mp
from joblib import Parallel, delayed
n_parallel_swaps=10
from pathos.multiprocessing import ProcessingPool as Pool
import time

import logging
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tdict import Tdict
import uuid
LOG_BASE = np.e
EPSILON = 1e-16
PRECISION = 1e-200

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
ch.setFormatter(formatter)


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger



def save_obj(f_name, obj):
    with open(f_name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)

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


# Stats
def log(x):
    return np.log(x + PRECISION) / np.log(LOG_BASE)


def normalize(freq):
    return freq / freq.sum()


def normalize_rows(A, p0=None):
    Z = A.sum(axis=1)[:, None]
    p0 = p0 if p0 is not None else 1/A.shape[1]
    return np.where(Z > 0, A / A.sum(axis=1)[:, None], p0)


def gibs(v, beta=1):
    return normalize(np.exp(beta * v))


def bayes(pY_X, pX):
    pXY = pY_X * pX if len(pX.shape) == 2 else pY_X * pX[:, None]
    pY = pXY.sum(axis=0)[:, None]
    return np.where(pY > EPSILON, pXY.T / pY, pX)


def split_dist(pXY):
    pX = pXY.sum(axis=1)[None].T
    assert (np.all(pX > 0))
    pY_X = pXY / pX
    return pX, pY_X


def conditional(pXY):
    pX = pXY.sum(axis=1)[:, None]
    return pXY / pX


def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > EPSILON, v * log(v), 0)


def H(p, axis=None):
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


def VI(pXY):
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - 2 * MI(pXY)


def DKL(p, q, axis=None):
    return (xlogx(p) - np.where(p > EPSILON, p * log(q), 0)).sum(axis=axis)


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


def coordinate_ascent(N, n, objective_function=None, n_init=1, n_iter=200,S_init=None):
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
        S = list(np.random.choice(np.arange(N), n, replace=False))
        S_out = list(S_full.difference(set(S)))
        fS = objective_function(S)
        changed = True
        t = 0
        LOGGER.info('===> Starting init %d, initial f(S) = %.5f' % (t_init, fS))
        while t < n_iter and changed:
            changed = False
            t += 1
            for si in np.random.choice(S, size=n, replace=False):

                for so in np.random.choice(S_out, size=n_out, replace=False):
                    S_swap = swap(S, si, so)
                    f_swap = objective_function(S_swap)
                    if f_swap > fS:
                        fS = f_swap
                        S = S_swap
                        S_out = swap(S_out, so, si)
                        changed = True
                        LOGGER.info('[%d/%d] [t = %d] swapped,  f(S) = %.5f' % (t_init + 1, n_init, t, fS))
                        break
                if changed:
                    break
            if not changed:
                LOGGER.info('[%d/%d] [t = %d] converged to a local optimum,  f(S) = %.5f' % (t_init + 1, n_init, t, fS))
            if t == n_iter:
                LOGGER.info('[%d/%d] max iteration %d reached, f(S) = %.5f' % (t_init + 1, n_init, n_iter, fS))
        if fS > f_best:
            S_best = S
            f_best = fS
    LOGGER.info('Done, opt f(S) = %.5f' % f_best)
    return S_best, f_best


def coordinate_ascent_v2(N, n, objective_function=None, n_init=1, n_iter=200,S_init=None):
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
        S = list(np.random.choice(np.arange(N), n, replace=False))
        S_out = list(S_full.difference(set(S)))
        fS = objective_function(S)
        changed = True
        t = 0
        LOGGER.info('===> Starting init %d, initial f(S) = %.5f' % (t_init, fS))
        while t < n_iter and changed:
            changed = False
            t += 1
            for i_in in np.random.permutation(n):
                si = S[i_in]
                for i_out in np.random.permutation(n_out):
                    so = S_out[i_out]
                    S_swap = swap(S, si, so)
                    f_swap = objective_function(S_swap)
                    if f_swap > fS:
                        fS = f_swap
                        swapi(S, S_out, i_in, i_out)
                        changed = True
                        LOGGER.info('[%d/%d] [t = %d] swapped,  f(S) = %.5f' % (t_init + 1, n_init, t, fS))
            if not changed:
                LOGGER.info('[%d/%d] [t = %d] converged to a local optimum,  f(S) = %.5f' % (t_init + 1, n_init, t, fS))
            if t == n_iter:
                LOGGER.info('[%d/%d] max iteration %d reached, f(S) = %.5f' % (t_init + 1, n_init, n_iter, fS))
        if fS > f_best:
            S_best = S
            f_best = fS
    LOGGER.info('Done, opt f(S) = %.5f' % f_best)
    return S_best, f_best


def coordinate_ascent_eh(N, n, objective_function=None, n_init=1, n_iter=200,early_stopping=False,stop_threshold=1e-3,S_init=None):
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
            if early_stopping:
                # check threshold crossing
                fs_diff=np.abs(fS_loop-fS_loop_start)
                thr_criteria=fs_diff < stop_threshold
                #update initial fS
                fS_loop_start=fS_loop
                if thr_criteria:
                    changed=False
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



def coordinate_ascent_parallel_eh(N, n, objective_function=None, n_init=1, n_iter=200,n_parallel_swaps=10,S_init=None):
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
    n_ps=n_parallel_swaps
    p_pool=Pool(n_ps)
    for t_init in range(n_init):
        if S_init is not None :
            S=S_init
        else:
            S = list(np.random.choice(np.arange(N), n, replace=False))
        #S = list(np.random.choice(np.arange(N), n, replace=False))
        S_out = list(S_full.difference(set(S)))
        fS = objective_function(S)
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
                so_idx = 0
                swap_groups=[so_list[i * n_ps:(i + 1) * n_ps] for i in range((len(so_list) + n_ps - 1) // n_ps)]
                keep_swapping=True
                while keep_swapping:
                    so_vals=swap_groups[so_idx]
                    new_swaps=[swap(S,si,x) for x in so_vals]
                    time_start = time.perf_counter()
                    f_swaps=p_pool.map(objective_function, new_swaps)
                    time_elapsed = (time.perf_counter() - time_start)
                    #print(f" elapsed time {time_elapsed}")
                    f_fS_comp=list(f_swaps > fS)
                    if any(f_fS_comp):
                        #new_id=f_fS_comp.index(True) # find first instance of true
                        new_id=np.argmax(f_swaps)
                        fS = f_swaps[new_id]
                        S = new_swaps[new_id]
                        S_out = swap(S_out, so_vals[new_id], si)
                        LOGGER.info('[%d/%d] [t = %d] id = %d, %d to %d after %d swaps,  f(S) = %.5f' % (
                        t_init + 1, n_init, t, si_idx, si, so_vals[new_id], so_idx+1, fS))
                        keep_swapping=False
                        changed=True
                    else:
                        keep_swapping = True
                        so_idx += 1
                    if so_idx==len(swap_groups):
                        LOGGER.info('[%d/%d] [t = %d] id = %d f(s) %d didnt change after all %d swaps,  f(S) = %.5f' % (t_init + 1, n_init, t,si_idx,si, so_idx, fS))
                        keep_swapping=False
            if not changed:
                LOGGER.info(
                            '[%d/%d] [t = %d] converged to a local optimum,  f(S) = %.5f' % (t_init + 1, n_init, t, fS))
            if t == n_iter:
                LOGGER.info('[%d/%d] max iteration %d reached, f(S) = %.5f' % (t_init + 1, n_init, n_iter, fS))
            time_elapsed = (time.perf_counter() - time_start)
            LOGGER.info('loop %d done, loop total time %f, f(S) = %.5f' % (t,time_elapsed, fS))
        if fS > f_best:
            S_best = S
            f_best = fS
    LOGGER.info('Done, opt f(S) = %.5f' % f_best)
    return S_best, f_best


