import torch
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.scipy.linalg import svd
from typing import Literal
from jax.config import config
import time

from jax.numpy import pad as jax_pad
config.update("jax_enable_x64", True)
def bures_dist(x,y):
    xxt = torch.mm(x, x.t())
    lam, U = torch.linalg.eigh(xxt)
    xxt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T
    # Compute YY^T
    yyt = torch.mm(y, y.t())
    xy_interim = torch.mm(torch.mm(xxt_sqrt, yyt), xxt_sqrt)
    lam, U = torch.linalg.eigh(xy_interim)
    xyt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T
    # compute bures,
    bures_ = torch.trace(xxt) + torch.trace(yyt) - 2 * torch.trace(xyt_sqrt)
    return bures_



def bures_dist_epsilon(x, y, epsilon=1e-8):
    # Compute XX^T
    xxt = torch.mm(x, x.t())
    lam, U = torch.linalg.eigh(xxt)
    xxt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T

    # Compute YY^T
    yyt = torch.mm(y, y.t())

    # Compute the interim matrix and add regularization
    xy_interim = torch.mm(torch.mm(xxt_sqrt, yyt), xxt_sqrt)
    xy_interim += torch.eye(xy_interim.size(0)).to(x.device) * epsilon

    try:
        lam, U = torch.linalg.eigh(xy_interim)
    except torch._C._LinAlgError as e:
        print("Encountered an error with linalg.eigh:", e)
        return None

    xyt_sqrt = U @ torch.diag(torch.sqrt(lam)) @ U.T

    # Compute Bures distance
    bures_ = torch.trace(xxt) + torch.trace(yyt) - 2 * torch.trace(xyt_sqrt)
    return bures_


def procrustes_dist(x,y):
    tr_xtx = torch.trace(torch.mm(x.T, x))
    tr_yty = torch.trace(torch.mm(y.T, y))
    #U, S, Vh = torch.linalg.svd(torch.mm(x.T, y))
    S=torch.linalg.svdvals(torch.mm(x.T, y),driver='gesvd')
    procrustes_ = tr_xtx + tr_yty - 2 * sum(S)
    return procrustes_


# Padding function in JAX
def pad_jax(array, max_pad):
    pad_width = ((0, 0), (0, max_pad - array.shape[-1]))
    return jnp.pad(array, pad_width, mode='constant', constant_values=0)

# Enable float64 precision
config.update("jax_enable_x64", True)

@jit
def compute_change(Xbar, X0):
    return jnp.linalg.norm(Xbar - X0) / jnp.sqrt(Xbar.size)

@jit
def update_barycenter(Xbar, XQ, n):
    return (n / (n + 1)) * Xbar + (1 / (n + 1)) * XQ


@jit
def jax_orthogonal_procrustes(A, B ):
    """Orthogonal Procrustes alignment of two matrices A and B.
    Both A and B need to be size (M, N) and JAX arrays.
    """
    # Compute xty
    #xty = jnp.dot(B.T, A).T
    xty = (B.T@ A).T
    # Choose SVD solver based on device type and solver preference
    U, w, Vt = svd((B.T@ A).T, full_matrices=True)
    # Compute R and scale
    R=U.dot(Vt) # this is equvalent to V @ U.T for B.T @ A becuase U is equal to V and Vt is equal to Ut, R is V @ U.T when we do svd of B.T @ A
    #R = jnp.dot(U, Vt)
    scale = jnp.sum(w)
    return R, scale



@jit
def jax_align(X: jnp.ndarray, Y: jnp.ndarray, group: int) -> jnp.ndarray:
    if group == 0:  # "orth"
        return jax_orthogonal_procrustes(X, Y)[0]
    elif group == 1:  # "perm"
        raise NotImplementedError("Permutation group alignment is not implemented.")
    elif group == 2:  # "identity"
        return jnp.eye(X.shape[1])
    else:
        raise ValueError(f"Specified group '{group}' not recognized.")

@jit
def _jax_euc_barycenter_streaming(Xs, group, random_state, tol, max_iter, warmstart, verbose, svd_solver):
    if group == 2:  # "identity"
        return jnp.mean(jnp.array(Xs), axis=0)

    # Stack Xs
    Xs = jnp.stack(Xs, axis=0)
    if Xs.ndim != 3:
        raise ValueError(
            "Expected 3d array with shape"
            "(n_datasets x n_observations x n_features), but "
            "got {}-d array with shape {}".format(Xs.ndim, Xs.shape))

    # If only one matrix is provided, the barycenter is trivial.
    if Xs.shape[0] == 1:
        return Xs[0]

    # Initialize random state and permutation indices
    if random_state is None:
        key = random.PRNGKey(int(time.time()))  # Create a key based on current time
    else:
        key = random.PRNGKey(random_state)
    key, subkey = random.split(key)
    indices = random.permutation(subkey, len(Xs))

    # Initialize barycenter.
    Xbar = Xs[indices[-1]] if (warmstart is None) else warmstart
    X0 = jnp.empty_like(Xbar)

    # Main loop
    itercount, n, chg = 0, 1, jnp.inf
    while (chg > tol) and (itercount < max_iter):
        # Save current barycenter for convergence checking.
        X0 = jnp.copy(Xbar)

        # Iterate over datasets.
        for i in indices:
            # Align i-th dataset to barycenter.
            XQ = jnp.dot(Xs[i], jax_align(Xs[i], X0, group=group))

            # Take a small step towards aligned representation.
            Xbar = update_barycenter(Xbar, XQ, n)
            n += 1

        # Detect convergence.
        chg = compute_change(Xbar, X0)

        # Display progress.
        if verbose:
            print(f"Iteration {itercount}, Change: {chg}")

        # Move to next iteration, with new random ordering over datasets.
        key, subkey = random.split(key)
        indices = random.permutation(subkey, len(Xs))
        itercount += 1

    return Xbar



def jax_frechet_mean(Xs, group="orth", random_state=None, tol=1e-3, max_iter=100, warmstart=None, verbose=False, method="streaming", return_aligned_Xs=False, svd_solver=None):
    group_dict = {"orth": 0, "perm": 1, "identity": 2}
    group_code = group_dict.get(group, -1)
    if group_code == -1:
        raise ValueError(f"Specified group '{group}' not recognized.")

    if group_code == 2:  # "identity"
        return jnp.mean(jnp.array(Xs), axis=0)

    if method == "streaming":
        Xbar = _jax_euc_barycenter_streaming(Xs, group_code, random_state, tol, max_iter, warmstart, verbose, svd_solver)
    elif method == "full_batch":
        raise NotImplementedError("Full batch method is not implemented yet.")

    if return_aligned_Xs:
        aligned_Xs = [x @ jax_align(x, Xbar, group=group_code) for x in Xs]

    return (Xbar, aligned_Xs) if return_aligned_Xs else Xbar

