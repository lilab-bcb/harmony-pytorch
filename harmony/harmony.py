import torch
import time

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from typing import Union
from .utils import one_hot_tensor

import logging

logger = logging.getLogger("harmony")


def harmonize(
    X: np.array,
    batch_mat: Union[pd.DataFrame, pd.Series],
    n_clusters: int = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 200,
    tol_harmony: float = 1e-4,
    tol_clustering: float = 1e-5,
    ridge_lambda: float = 1.0,
    sigma: float = 0.1,
    block_proportion: float = 0.05,
    theta: float = 2.0,
    tau: int = 0,
    correction_method: str = "fast",
    random_state: int = 0,
) -> np.array:
    """
    Integrate data using Harmony algorithm.

    Parameters
    ----------

    X: ``numpy.array``
        The input embedding with rows for cells (N) and columns for embedding coordinates (d).

    batch_mat: ``pandas.DataFrame`` or ``pandas.Series``
        The batch information of cells with rows for cells (N) and columns for batch factors.

    n_clusters: ``int``, optional, default: ``None``
        Number of clusters used in Harmony algorithm. If ``None``, choose the minimum of 100 and N / 30.

    max_iter_harmony: ``int``, optional, default: ``10``
        Maximum iterations on running Harmony if not converged.

    max_iter_clustering: ``int``, optional, default: ``200``
        Within each Harmony iteration, maximum iterations on the clustering step if not converged.

    tol_harmony: ``float``, optional, default: ``1e-4``
        Tolerance on justifying convergence of Harmony over objective function values.

    tol_clustering: ``float``, optional, default: ``1e-5``
        Tolerance on justifying convergence of the clustering step over objective function values within each Harmony iteration.

    ridge_lambda: ``float``, optional, default: ``1.0``
        Hyperparameter of ridge regression on the correction step.

    sigma: ``float``, optional, default: ``0.1``
        Weight of the entropy term in objective function.

    block_proportion: ``float``, optional, default: ``0.05``
        Proportion of block size in one update operation of clustering step.

    theta: ``float``, optional, default: ``2.0``
        Weight of the diversity penalty term in objective function.

    tau: ``int``, optional, default: ``0``
        Discounting factor on ``theta``. By default, there is no discounting.

    random_state: ``int``, optional, default: ``0``
        Random seed for reproducing results.

    Returns
    -------
    ``numpy.array``
        The integrated embedding by Harmony, of the same shape as the input embedding.

    Examples
    --------
    >>> adata = anndata.read_h5ad("filename.h5ad")
    >>> X_harmony = harmonize(adata.obsm['X_pca'], adata.obs['Channel'])
    """

    start = time.perf_counter()

    Z = torch.tensor(X, dtype=torch.float)
    Z_norm = normalize(Z, p=2, dim=1)
    n_cells = Z.shape[0]

    batch_codes = get_batch_codes(batch_mat)
    n_batches = batch_codes.nunique()
    N_b = torch.tensor(batch_codes.value_counts(sort=False).values, dtype=torch.float)
    Pr_b = N_b.view(-1, 1) / n_cells

    Phi = one_hot_tensor(batch_codes)

    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    theta = torch.tensor([theta], dtype=torch.float).expand(n_batches)

    if tau > 0:
        theta = theta * (1 - torch.exp(-N_b / (n_clusters * tau)) ** 2)

    theta = theta.view(1, -1)

    assert block_proportion >= 0 and block_proportion <= 1
    assert correction_method in ["fast", "original"]

    # Initialization
    R, E, O, objectives_harmony = initialize_centroids(
        Z_norm, n_clusters, sigma, Pr_b, Phi, theta, random_state
    )

    np.random.seed(random_state)
    rand_arr = np.random.randint(np.iinfo(np.int32).max, size=max_iter_harmony)

    for i in range(max_iter_harmony):
        start_iter = time.perf_counter()
        R = clustering(
            Z_norm,
            Pr_b,
            Phi,
            R,
            E,
            O,
            n_clusters,
            theta,
            tol_clustering,
            objectives_harmony,
            rand_arr[i],
            max_iter_clustering,
            sigma,
            block_proportion,
        )
        Z_hat = correction(Z, R, Phi, ridge_lambda, correction_method)
        end_iter = time.perf_counter()

        print(
            "\tCompleted {cur_iter} / {total_iter} in {duration:.2f}s.".format(
                cur_iter=i + 1,
                total_iter=max_iter_harmony,
                duration=end_iter - start_iter,
            )
        )

        if is_convergent_harmony(objectives_harmony, tol=tol_harmony):
            print("\tReach convergence after {} iteration(s).".format(i + 1))
            break

    end = time.perf_counter()
    logger.info(
        "Harmony integration is done. Time spent = {:.2f}s.".format(end - start)
    )

    return Z_hat.numpy()


def get_batch_codes(batch_mat):
    return batch_mat.astype("category").cat.codes.astype("category")


def initialize_centroids(
    Z_norm, n_clusters, sigma, Pr_b, Phi, theta, random_state, n_init=10
):
    n_cells = Z_norm.shape[0]

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
        n_jobs=-1,
    )
    kmeans.fit(Z_norm)

    Y = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    Y_norm = normalize(Y, p=2, dim=1)

    # Initialize R
    dist_mat = 2 * (1 - torch.matmul(Z_norm, Y_norm.t()))
    R = -dist_mat / sigma
    R = torch.add(R, -torch.max(R, dim=1, keepdim=True).values)
    R = torch.exp(R)
    R = torch.div(R, torch.sum(R, dim=1, keepdim=True))

    E = torch.matmul(Pr_b, torch.sum(R, dim=0, keepdim=True))
    O = torch.matmul(Phi.t(), R)

    objectives_harmony = []
    compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E, objectives_harmony)

    return R, E, O, objectives_harmony


def clustering(
    Z_norm,
    Pr_b,
    Phi,
    R,
    E,
    O,
    n_clusters,
    theta,
    tol,
    objectives_harmony,
    random_state,
    max_iter,
    sigma,
    block_proportion,
    n_init=10,
):

    # Initialize cluster centroids
    n_cells = Z_norm.shape[0]

    Y = torch.matmul(R.t(), Z_norm)
    Y_norm = normalize(Y, p=2, dim=1)

    # Compute initialized objective.
    objectives_clustering = []
    compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E, objectives_clustering)

    np.random.seed(random_state)

    for i in range(max_iter):
        idx_list = np.arange(n_cells)
        np.random.shuffle(idx_list)
        block_size = int(n_cells * block_proportion)
        pos = 0
        while pos < len(idx_list):
            idx_in = idx_list[pos : (pos + block_size)]
            R_in = R[
                idx_in,
            ]
            Phi_in = Phi[
                idx_in,
            ]

            # Compute O and E on left out data.
            O -= torch.matmul(Phi_in.t(), R_in)
            E -= torch.matmul(Pr_b, torch.sum(R_in, dim=0, keepdim=True))

            # Update and Normalize R
            R_in = torch.exp(
                -2 / sigma * (1 - torch.matmul(Z_norm[idx_in,], Y_norm.t()))
            )
            omega = torch.matmul(Phi_in, torch.pow(torch.div(E + 1, O + 1), theta.t()))
            R_in = R_in * omega
            R_in = normalize(R_in, p=1, dim=1)
            R[idx_in,] = R_in

            # Compute O and E with full data.
            O += torch.matmul(Phi_in.t(), R_in)
            E += torch.matmul(Pr_b, torch.sum(R_in, dim=0, keepdim=True))

            pos += block_size

        # Compute Cluster Centroids
        Y = torch.matmul(R.t(), Z_norm)
        Y_norm = normalize(Y, p=2, dim=1)

        compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E, objectives_clustering)

        if is_convergent_clustering(objectives_clustering, tol):
            objectives_harmony.append(objectives_clustering[-1])
            break

    return R


def correction(X, R, Phi, ridge_lambda, correction_method):
    if correction_method == "fast":
        return correction_fast(X, R, Phi, ridge_lambda)
    else:
        return correction_original(X, R, Phi, ridge_lambda)


# def correction(X, R, Phi, ridge_lambda, correction_method):
#    n_cells = X.shape[0]
#    n_clusters = R.shape[1]
#    n_batches = Phi.shape[1]
#    Phi_1 = torch.cat((torch.ones(n_cells, 1), Phi), dim = 1)
#
#    Z = X.clone()
#    N = torch.matmul(Phi.t(), R)
#    P = torch.eye(n_batches + 1, n_batches + 1)
#    for k in range(n_clusters):
#        Phi_t_diag_R = Phi_1.t() * R[:, k].view(1, -1)
#        inv_mat_1 = torch.inverse(torch.matmul(Phi_t_diag_R, Phi_1) + ridge_lambda * torch.eye(n_batches + 1, n_batches + 1))
#
#        N_k = torch.sum(R[:,k])
#        factor = 1 / (N[:, k] + ridge_lambda)
#        c = N_k + ridge_lambda + torch.sum(-factor * N[:, k]**2)
#        P[0, 1:] = -factor * N[:, k]
#        B = torch.cat((torch.tensor([[1/c]]), factor.view(1, -1)), dim = 1)
#        inv_mat_2 = torch.matmul(P.t() * B.view(1, -1), P)
#
#        if k == 0:
#            print("================")
#            print(inv_mat_1)
#            print(inv_mat_2)
#
#        inv_mat = inv_mat_1 if correction_method == 'original' else inv_mat_2
#
#        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
#        W[0, :] = 0
#        Z -= torch.matmul(Phi_t_diag_R.t(), W)


def correction_original(X, R, Phi, ridge_lambda):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1), Phi), dim=1)

    Z = X.clone()
    for k in range(n_clusters):
        Phi_t_diag_R = Phi_1.t() * R[:, k].view(1, -1)
        inv_mat = torch.inverse(
            torch.matmul(Phi_t_diag_R, Phi_1)
            + ridge_lambda * torch.eye(n_batches + 1, n_batches + 1)
        )
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0
        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def correction_fast(X, R, Phi, ridge_lambda):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1), Phi), dim=1)

    N = torch.matmul(Phi.t(), R)

    Z = X.clone()
    P = torch.eye(n_batches + 1, n_batches + 1)
    for k in range(n_clusters):
        N_k = torch.sum(R[:, k])

        factor = 1 / (N[:, k] + ridge_lambda)
        c = N_k + ridge_lambda + torch.sum(-factor * N[:, k] ** 2)
        c_inv = 1 / c

        P[0, 1:] = -factor * N[:, k]

        P_t_B_inv = torch.diag(
            torch.cat((torch.tensor([[c_inv]]), factor.view(1, -1)), dim=1).squeeze()
        )
        P_t_B_inv[1:, 0] = P[0, 1:] * c_inv
        inv_mat = torch.matmul(P_t_B_inv, P)

        Phi_t_diag_R = Phi_1.t() * R[:, k].view(1, -1)
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0

        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E, objective_arr):
    kmeans_error = torch.sum(R * 2 * (1 - torch.matmul(Z_norm, Y_norm.t())))
    entropy_term = sigma * torch.sum(R * torch.log(R))
    diversity_penalty = sigma * torch.sum(
        torch.matmul(theta, O * torch.log(torch.div(O + 1, E + 1)))
    )
    objective = kmeans_error + entropy_term + diversity_penalty

    objective_arr.append(objective)


def is_convergent_harmony(objectives_harmony, tol):
    if len(objectives_harmony) < 2:
        return False

    obj_old = objectives_harmony[-2]
    obj_new = objectives_harmony[-1]

    return np.abs(obj_old - obj_new) < tol * np.abs(obj_old)


def is_convergent_clustering(objectives_clustering, tol, window_size=3):
    if len(objectives_clustering) < window_size + 1:
        return False

    obj_old = 0
    obj_new = 0
    for i in range(window_size):
        obj_old += objectives_clustering[-2 - i]
        obj_new += objectives_clustering[-1 - i]

    return np.abs(obj_old - obj_new) < tol * np.abs(obj_old)
