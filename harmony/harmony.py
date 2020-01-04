import torch

import numpy as np

from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from .utils import one_hot_tensor

def harmonize(X, batch_mat, n_clusters = None, tau = 0, max_iter = 10, tol_harmony = 1e-4, tol_clustering = 1e-5, ridge_lambda = 1.0):
    X_tensor = torch.tensor(X, dtype = torch.float)
    Z = X_tensor.clone()
    n_cells = X_tensor.shape[0]

    batch_codes = batch_mat.astype('category').cat.codes.astype('category')
    n_batches = batch_codes.nunique()
    N_b = torch.tensor(batch_codes.value_counts(sort = False).values)
    Pr_b = N_b.float() / n_cells

    Phi = one_hot_tensor(batch_codes)

    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    R = torch.zeros(n_cells, n_clusters, dtype = torch.float)

    if tau <= 0:
        tau = np.random.randint(5, 21)

    theta = len(N_b) * (1 - torch.exp(- N_b.float() / (n_clusters * tau)) ** 2)
    
    # Initialization
    objectives_harmony = []

    for i in range(max_iter):
        R, Y = clustering(X_tensor, Z, Pr_b, Phi, R, n_clusters, theta, tol_clustering, objectives_harmony)
        Z_new = correction(X_tensor, R, Phi, ridge_lambda)
        
        if is_convergent_harmony(objectives_harmony, tol = tol_harmony):
            break
        else:
            Z = Z_new

    return Z


def clustering(X, Z, Pr_b, Phi, R, n_clusters, theta, tol, objectives_harmony, n_init = 10, random_state = 0, max_iter = 200, sigma = 0.1):
    
    # Initialize cluster centroids
    n_cells = Z.shape[0]

    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = n_init, random_state = random_state, n_jobs = -1)
    kmeans.fit(Z)
    Y = torch.tensor(kmeans.cluster_centers_, dtype = torch.float)

    Y_norm = normalize(Y, p = 2, dim = 1)
    Z_norm = normalize(Z, p = 2, dim = 1)

    # Initialize R
    dist_mat = 2 * (1 - torch.matmul(Z_norm, Y_norm.t()))
    R = -dist_mat / sigma
    R = torch.add(R, -torch.max(R, dim = 1).values.view(-1, 1))
    R = torch.exp(R)
    R = torch.div(R, torch.sum(R, dim = 1).view(-1, 1))

    E = torch.matmul(Pr_b.view(-1, 1), torch.sum(R, dim = 0).view(1, -1))
    O = torch.matmul(Phi.t(), R)

    # Compute initialized objective.
    objectives_clustering = []
    compute_objective(Y_norm, Z_norm, R, Phi, theta, sigma, O, E, objectives_clustering)

    for i in range(max_iter):
        idx_list = np.arange(n_cells)
        np.random.shuffle(idx_list)
        block_size = int(n_cells * 0.05)
        pos = 0
        while pos < len(idx_list):
            idx_in = idx_list[pos:(pos + block_size)]
            R_in = R[idx_in,]
            Phi_in = Phi[idx_in,]
    
            # Compute O and E on left out data.
            O -= torch.matmul(Phi_in.t(), R_in)
            E -= torch.matmul(Pr_b.view(-1, 1), torch.sum(R_in, dim = 0).view(1, -1))
    
            # Update and Normalize R
            R_in = torch.exp(- 2 / sigma * (1 - torch.matmul(Z[idx_in,], Y_norm.t())))
            diverse_penalty = torch.matmul(Phi_in, torch.pow(torch.div(E + 1, O + 1), theta.view(-1, 1).expand_as(E)))
            R_in = torch.mul(R_in, diverse_penalty)
            R_in = normalize(R_in, p = 1, dim = 1)
            R[idx_in,] = R_in
    
            # Compute O and E with full data.
            O += torch.matmul(Phi_in.t(), R_in)
            E += torch.matmul(Pr_b.view(-1, 1), torch.sum(R_in, dim = 0).view(1, -1))
    
            pos += block_size

        # Compute Cluster Centroids
        Y_new = torch.matmul(R.t(), X)
        Y_new_norm = normalize(Y_new, p = 2, dim = 1)

        compute_objective(Y_new_norm, Z_norm, R, Phi, theta, sigma, O, E, objectives_clustering)

        if is_convergent_clustering(objectives_clustering, tol):
            objectives_harmony.append(objectives_clustering[-1])
            break
        else:
            Y_norm = Y_new_norm

    return R, Y_norm


def correction(X, R, Phi, ridge_lambda):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1), Phi), dim = 1)

    Z = X.clone()
    for k in range(n_clusters):
        Phi_t_diag_R = torch.matmul(Phi_1.t(), torch.diag(R[:,k]))
        inv_mat = torch.inverse(torch.matmul(Phi_t_diag_R, Phi_1) + ridge_lambda * torch.eye(n_batches + 1, n_batches + 1))
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0
        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def compute_objective(Y_norm, Z_norm, R, Phi, theta, sigma, O, E, objective_arr):
    kmeans_error = torch.sum(R * 2 * (1 - torch.matmul(Z_norm, Y_norm.t())))
    entropy_term = sigma * torch.sum(R * torch.log(R))
    diverse_penalty = sigma * torch.sum(R * torch.matmul(Phi, theta.view(-1, 1).expand_as(E) * torch.log(torch.div(O + 1, E + 1))))
    objective = kmeans_error + entropy_term + diverse_penalty

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
    