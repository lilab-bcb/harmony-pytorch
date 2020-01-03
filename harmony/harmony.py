import torch
import numpy as np

from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from utils import one_hot_tensor

def harmonize(X, batch_mat, n_clusters = None, tau = 0, max_iter = 10, tol_harmony = 1e-4, tol_clustering = 1e-5, ...):
    Z = torch.tensor(X)
    n_cells = X.shape[0]

    batch_codes = batch_mat.cat.codes
    N_b = torch.tensor(batch_codes.value_counts(sort = False))
    Pr_b = N_b.float() / n_cells

    Phi = one_hot_tensor(batch_codes)

    # TODO
    R = torch.tensor(...)

    n_cells = X.shape[0]
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    if tau <= 0:
        tau = np.random.randint(5, 21)

    theta = len(N_b) * (1 - torch.exp(- N_b.float() / (n_clusters * tau)) ** 2)

    for i in range(max_iter):
        R = clustering(X, Z, Pr_b, Phi, R, n_clusters, theta, tol_clustering)
        Z_new = correction(X, R, Phi)
        
        if is_convergent(Z, Z_new, level = 'harmony', tol = tol_harmony):
            break
        else:
            Z = Z_new

    return Z

def clustering(X, Z, Pr_b, Phi, R, n_clusters, theta, tol, n_init = 10, random_state = 0, max_iter = 200, sigma = 0.1):
    
    # Initialize cluster centroids
    n_cells = Z.shape[0]

    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = n_init, random_state = random_state, n_jobs = -1)
    kmeans.fit(Z)
    Y = torch.tensor(kmeans.cluster_centers_, dtype = torch.float32)

    Y_norm = normalize(Y, p = 2, dim = 1)
    Z_norm = normalize(Z, p = 2, dim = 1)

    E = torch.matmul(Pr_b.t(), torch.sum(R, dim = 0))
    O = torch.matmul(Phi.t(), R)

    for i in range(max_iter):
        idx_list = np.arange(n_cells)
        np.random.shuffle(id_list)
        block_size = int(n_cells * 0.05)
        pos = 0
        while pos < len(idx_list):
            idx_in = idx_list[pos:(pos + block_size)]
            R_in = R[idx_in,]
            Phi_in = Phi[idx_in,]
    
            # Compute O and E on left out data.
            O -= torch.matmul(Phi_in.t(), R_in)
            E -= torch.matmul(Pr_b.t(), torch.sum(R_in, dim = 0))
    
            # Update and Normalize R
            R_in = torch.exp(- 2 / sigma * (1 - torch.matmul(Z[idx_in,], Y_norm.t())))
            diverse_penalty = torch.matmul(Phi_in, torch.pow(torch.div(E + 1, O + 1), theta.view(-1, 1).expand_as(E)))
            R_in = torch.mul(R_in, diverse_penalty)
            R_in = normalize(R_in, p = 1, dim = 1)
            R[idx_in,] = R_in
    
            # Compute O and E with full data.
            O += torch.matmul(Phi_in.t(), R_in)
            E += torch.matmul(Pr_b.t(), torch.sum(R_in, dim = 0))
    
            pos += block_size

        # Compute Cluster Centroids
        Y_new = torch.matmul(R.t(), X)
        Y_new_norm = normalize(Y, p = 2, dim = 1)

        if is_convergent(Y_new_norm, Y_norm, level = 'clustering', tol = tol):
            break
        else:
            Y_norm = Y_new_norm

    return R


def correction(X, R, Phi, ridge_lambda = 1.0):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1), Phi), dim = 1)

    Z = X
    for k in range(n_clusters):
        diag_R = torch.diag(R[:,k])
        inv_mat = torch.inverse(torch.matmul(torch.matmul(Phi_1.t(), diag_R), Phi_1) + ridge_lambda * torch.eye(n_batches + 1, n_batches + 1))
        W = torch.matmul(inv_mat, torch.matmul(torch.matmul(Phi_1.t(), diag_R), X))
        W[0, :] = 0
        Z = X - torch.matmul(torch.matmul(Phi_1, W), diag_R)

    return Z


# TODO
def is_convergent(X, Y, level, tol):
    assert level in ['harmony', 'clustering']
    
    if level == 'harmony':

    else:
