import torch
import time

import numpy as np

from pandas import DataFrame
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from .utils import one_hot_tensor

import logging
logger = logging.getLogger("harmony")

def harmonize(
    X: np.array,
    batch_mat: DataFrame,
    n_clusters: int = None,
    theta: float = 2.0,
    tau: int = 0,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 200,
    tol_harmony: float = 1e-4,
    tol_clustering: float = 1e-5,
    ridge_lambda: float = 1.0,
    sigma: float = 0.1,
    block_proportion: float = 0.05,
    correction_method: str = "fast",
    random_state: int = 0,
) -> torch.Tensor:

    start = time.perf_counter()

    Z = torch.tensor(X, dtype = torch.float)
    n_cells = Z.shape[0]

    batch_codes = batch_mat.astype('category').cat.codes.astype('category')
    n_batches = batch_codes.nunique()
    N_b = torch.tensor(batch_codes.value_counts(sort = False).values, dtype = torch.float)
    Pr_b = N_b.view(-1, 1) / n_cells

    Phi = one_hot_tensor(batch_codes)

    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    R = torch.zeros(n_cells, n_clusters, dtype = torch.float)

    theta = torch.tensor([theta], dtype = torch.float).expand(n_batches)

    if tau > 0:
        theta = theta * (1 - torch.exp(- N_b / (n_clusters * tau)) ** 2)

    theta = theta.view(1, -1)
    
    assert block_proportion >= 0 and block_proportion <= 1
    assert correction_method in ["fast", "original"]

    # Initialization
    objectives_harmony = []

    np.random.seed(random_state)
    rand_arr = np.random.randint(np.iinfo(np.int32).max, size = max_iter_harmony)

    for i in range(max_iter_harmony):
        R = clustering(Z, Pr_b, Phi, R, n_clusters, theta, tol_clustering, objectives_harmony, rand_arr[i], sigma, max_iter_clustering, block_proportion)
        Z_hat = correction(Z, R, Phi, ridge_lambda, correction_method)
        
        print("\tcompleted  {cur_iter} / {total_iter}  iterations".format(cur_iter = i + 1, total_iter = max_iter_harmony))

        if is_convergent_harmony(objectives_harmony, tol = tol_harmony):
            break

    end = time.perf_counter()
    logger.info("Harmony integration is done. Time spent = {:.2f}s.".format(end - start))

    return Z_hat


def clustering(Z, Pr_b, Phi, R, n_clusters, theta, tol, objectives_harmony, random_state, sigma, max_iter, block_proportion, n_init = 10):
    
    # Initialize cluster centroids
    n_cells = Z.shape[0]
    Z_norm = normalize(Z, p = 2, dim = 1)

    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = n_init, random_state = random_state, n_jobs = -1)
    kmeans.fit(Z_norm)
    Y = torch.tensor(kmeans.cluster_centers_, dtype = torch.float)

    Y_norm = normalize(Y, p = 2, dim = 1)

    # Initialize R
    dist_mat = 2 * (1 - torch.matmul(Z_norm, Y_norm.t()))
    R = -dist_mat / sigma
    R = torch.add(R, -torch.max(R, dim = 1, keepdim = True).values)
    R = torch.exp(R)
    R = torch.div(R, torch.sum(R, dim = 1, keepdim = True))

    E = torch.matmul(Pr_b, torch.sum(R, dim = 0, keepdim = True))
    O = torch.matmul(Phi.t(), R)

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
            idx_in = idx_list[pos:(pos + block_size)]
            R_in = R[idx_in,]
            Phi_in = Phi[idx_in,]
    
            # Compute O and E on left out data.
            O -= torch.matmul(Phi_in.t(), R_in)
            E -= torch.matmul(Pr_b, torch.sum(R_in, dim = 0, keepdim = True))
    
            # Update and Normalize R
            R_in = torch.exp(- 2 / sigma * (1 - torch.matmul(Z[idx_in,], Y_norm.t())))
            omega = torch.matmul(Phi_in, torch.pow(torch.div(E + 1, O + 1), theta.view(-1, 1)))
            R_in = R_in * omega
            R_in = normalize(R_in, p = 1, dim = 1)
            R[idx_in,] = R_in
    
            # Compute O and E with full data.
            O += torch.matmul(Phi_in.t(), R_in)
            E += torch.matmul(Pr_b, torch.sum(R_in, dim = 0, keepdim = True))
    
            pos += block_size

        # Compute Cluster Centroids
        Y_new = torch.matmul(R.t(), Z)
        Y_new_norm = normalize(Y_new, p = 2, dim = 1)

        compute_objective(Y_new_norm, Z_norm, R, theta, sigma, O, E, objectives_clustering)

        if is_convergent_clustering(objectives_clustering, tol):
            objectives_harmony.append(objectives_clustering[-1])
            break
        else:
            Y_norm = Y_new_norm

    return R

def correction(X, R, Phi, ridge_lambda, correction_method):
    if correction_method == 'fast':
        return correction_fast(X, R, Phi, ridge_lambda)
    else:
        return correction_original(X, R, Phi, ridge_lambda)


def correction_original(X, R, Phi, ridge_lambda):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1), Phi), dim = 1)

    Z = X.clone()
    for k in range(n_clusters):
        Phi_t_diag_R = Phi_1.t() * R[:,k].view(1, -1)
        inv_mat = torch.inverse(torch.matmul(Phi_t_diag_R, Phi_1) + ridge_lambda * torch.eye(n_batches + 1, n_batches + 1))
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0
        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def correction_fast(X, R, Phi, ridge_lambda):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1), Phi), dim = 1)

    N = torch.matmul(Phi.t(), R)

    Z = X.clone()
    P = torch.eye(n_batches + 1, n_batches + 1)
    for k in range(n_clusters):
        N_k = torch.sum(R[:,k])

        factor = 1 / (N[:, k] + ridge_lambda)
        c = N_k + ridge_lambda + torch.sum(-factor * N[:, k]**2)
        
        P[0, 1:] = -factor * N[:, k]
        B_inv = torch.cat((torch.tensor([[1/c]]), factor.view(1, -1)), dim = 1)
        inv_mat = torch.matmul(P.t() * B_inv.view(1, -1), P)

        Phi_t_diag_R = Phi_1.t() * R[:,k].view(1, -1)
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0

        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E, objective_arr):
    kmeans_error = torch.sum(R * 2 * (1 - torch.matmul(Z_norm, Y_norm.t())))
    entropy_term = sigma * torch.sum(R * torch.log(R))
    diversity_penalty = sigma * torch.sum(torch.matmul(theta, O * torch.log(torch.div(O + 1, E + 1))))
    objective = kmeans_error + entropy_term + diversity_penalty

    objective_arr.append(objective)


def is_convergent_harmony(objectives_harmony, tol):
    if len(objectives_harmony) < 2:
        return False

    obj_old = objectives_harmony[-2]
    obj_new = objectives_harmony[-1]

    return np.abs(obj_old - obj_new) < tol * np.abs(obj_old)


def is_convergent_clustering(objectives_clustering, tol, window_size = 3):
    if len(objectives_clustering) < window_size + 1:
        return False

    obj_old = 0
    obj_new = 0
    for i in range(window_size):
        obj_old += objectives_clustering[-2 - i]
        obj_new += objectives_clustering[-1 - i]

    return np.abs(obj_old - obj_new) < tol * np.abs(obj_old)
    