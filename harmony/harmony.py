import torch

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from typing import Union, List
from .utils import one_hot_tensor, get_batch_codes



def harmonize(
    X: np.array,
    batch_mat: pd.DataFrame,
    batch_key: Union[str, List[str]],
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
    use_gpu: bool = False,
    n_jobs_kmeans: int = -1,
) -> np.array:
    """
    Integrate data using Harmony algorithm.

    Parameters
    ----------

    X: ``numpy.array``
        The input embedding with rows for cells (N) and columns for embedding coordinates (d).

    batch_mat: ``pandas.DataFrame``
        The cell barcode information as data frame, with rows for cells (N) and columns for cell attributes.

    batch_key: ``str`` or ``List[str]``
        Cell attribute(s) from ``batch_mat`` to identify batches.

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

    correction_method: ``string``, optional, default: ``fast``
        Choose which method for the correction step: ``original`` for original method, ``fast`` for improved method. By default, use improved method.

    random_state: ``int``, optional, default: ``0``
        Random seed for reproducing results.

    use_gpu: ``bool``, optional, default: ``False``
        If ``True``, use GPU if available. Otherwise, use CPU only.

    n_jobs_kmeans: ``int``, optional, default ``-1``
        How many threads to use for KMeans. By default, use all available cores.

    Returns
    -------
    ``numpy.array``
        The integrated embedding by Harmony, of the same shape as the input embedding.

    Examples
    --------
    >>> adata = anndata.read_h5ad("filename.h5ad")
    >>> X_harmony = harmonize(adata.obsm['X_pca'], adata.obs, 'Channel')

    >>> adata = anndata.read_h5ad("filename.h5ad")
    >>> X_harmony = harmonize(adata.obsm['X_pca'], adata.obs, ['Channel', 'Lab'])
    """

    device_type = "cpu"
    if use_gpu:
        if torch.cuda.is_available():
            device_type = "cuda"
            print("Use GPU mode.")
        else:
            print("CUDA is not available on your machine. Use CPU mode instead.")

    Z = torch.tensor(X, dtype=torch.float, device=device_type)
    Z_norm = normalize(Z, p=2, dim=1)
    n_cells = Z.shape[0]

    batch_codes = get_batch_codes(batch_mat, batch_key)
    n_batches = batch_codes.nunique()
    N_b = torch.tensor(
        batch_codes.value_counts(sort=False).values,
        dtype=torch.float,
        device=device_type,
    )
    Pr_b = N_b.view(-1, 1) / n_cells

    Phi = one_hot_tensor(batch_codes, device_type)

    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    theta = torch.tensor([theta], dtype=torch.float, device=device_type).expand(
        n_batches
    )

    if tau > 0:
        theta = theta * (1 - torch.exp(-N_b / (n_clusters * tau)) ** 2)

    theta = theta.view(1, -1)

    assert block_proportion > 0 and block_proportion <= 1
    assert correction_method in ["fast", "original"]

    np.random.seed(random_state)

    # Initialize centroids
    R, E, O, objectives_harmony = initialize_centroids(
        Z_norm,
        n_clusters,
        sigma,
        Pr_b,
        Phi,
        theta,
        None,
        device_type,
        n_jobs_kmeans,
    )

    print("\tInitialization is completed.")

    for i in range(max_iter_harmony):
        clustering(
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
            max_iter_clustering,
            sigma,
            block_proportion,
            device_type,
        )
        Z_hat = correction(Z, R, Phi, O, ridge_lambda, correction_method, device_type)
        Z_norm = normalize(Z_hat, p=2, dim=1)

        print(
            "\tCompleted {cur_iter} / {total_iter} iteration(s).".format(
                cur_iter=i + 1,
                total_iter=max_iter_harmony,
            )
        )

        if is_convergent_harmony(objectives_harmony, tol=tol_harmony):
            print("Reach convergence after {} iteration(s).".format(i + 1))
            break

    if device_type == "cpu":
        return Z_hat.numpy()
    else:
        return Z_hat.cpu().numpy()


def initialize_centroids(
    Z_norm,
    n_clusters,
    sigma,
    Pr_b,
    Phi,
    theta,
    random_state,
    device_type,
    n_jobs,
    n_init=10,
):
    n_cells = Z_norm.shape[0]

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
        n_jobs=n_jobs,
        max_iter=25,
    )

    if device_type == "cpu":
        kmeans.fit(Z_norm)
    else:
        kmeans.fit(Z_norm.cpu())

    Y = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device_type)
    Y_norm = normalize(Y, p=2, dim=1)

    # Initialize R
    R = torch.exp(
        -2 / sigma * (1 - torch.matmul(Z_norm, Y_norm.t()))
    )
    R = normalize(R, p=1, dim=1)

    E = torch.matmul(Pr_b, torch.sum(R, dim=0, keepdim=True))
    O = torch.matmul(Phi.t(), R)

    objectives_harmony = []
    compute_objective(
        Y_norm, Z_norm, R, theta, sigma, O, E, objectives_harmony, device_type
    )

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
    max_iter,
    sigma,
    block_proportion,
    device_type,
    n_init=10,
):

    n_cells = Z_norm.shape[0]

    objectives_clustering = []

    for i in range(max_iter):
        # Compute Cluster Centroids
        Y = torch.matmul(R.t(), Z_norm)
        Y_norm = normalize(Y, p=2, dim=1)

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

        compute_objective(
            Y_norm, Z_norm, R, theta, sigma, O, E, objectives_clustering, device_type
        )

        if is_convergent_clustering(objectives_clustering, tol):
            objectives_harmony.append(objectives_clustering[-1])
            break


def correction(X, R, Phi, O, ridge_lambda, correction_method, device_type):
    if correction_method == "fast":
        return correction_fast(X, R, Phi, O, ridge_lambda, device_type)
    else:
        return correction_original(X, R, Phi, ridge_lambda, device_type)


def correction_original(X, R, Phi, ridge_lambda, device_type):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1, device=device_type), Phi), dim=1)

    Z = X.clone()
    id_mat = torch.eye(n_batches + 1, n_batches + 1, device = device_type)
    id_mat[0, 0] = 0
    Lambda = ridge_lambda * id_mat
    for k in range(n_clusters):
        Phi_t_diag_R = Phi_1.t() * R[:, k].view(1, -1)
        inv_mat = torch.inverse(
            torch.matmul(Phi_t_diag_R, Phi_1) + Lambda
        )
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0
        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def correction_fast(X, R, Phi, O, ridge_lambda, device_type):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1, device=device_type), Phi), dim=1)

    Z = X.clone()
    P = torch.eye(n_batches + 1, n_batches + 1, device=device_type)
    for k in range(n_clusters):
        O_k = O[:, k]
        N_k = torch.sum(O_k)

        factor = 1 / (O_k + ridge_lambda)
        c = N_k + torch.sum(-factor * O_k ** 2)
        c_inv = 1 / c

        P[0, 1:] = -factor * O_k

        P_t_B_inv = torch.diag(
            torch.cat(
                (torch.tensor([[c_inv]], device=device_type), factor.view(1, -1)), dim=1
            ).squeeze()
        )
        P_t_B_inv[1:, 0] = P[0, 1:] * c_inv
        inv_mat = torch.matmul(P_t_B_inv, P)

        Phi_t_diag_R = Phi_1.t() * R[:, k].view(1, -1)
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0

        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def compute_objective(
    Y_norm, Z_norm, R, theta, sigma, O, E, objective_arr, device_type
):
    kmeans_error = torch.sum(R * 2 * (1 - torch.matmul(Z_norm, Y_norm.t())))
    entropy_term = sigma * torch.sum(-torch.distributions.Categorical(probs=R).entropy())
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

    return (obj_old - obj_new) < tol * torch.abs(obj_old)


def is_convergent_clustering(objectives_clustering, tol, window_size=3):
    if len(objectives_clustering) < window_size + 1:
        return False

    obj_old = 0
    obj_new = 0
    for i in range(window_size):
        obj_old += objectives_clustering[-2 - i]
        obj_new += objectives_clustering[-1 - i]

    return (obj_old - obj_new) < tol * torch.abs(obj_old)
