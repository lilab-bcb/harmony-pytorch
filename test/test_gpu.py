import numpy as np
import pegasus as pg
import pandas as pd

import os, sys, time, re

from harmony import harmonize
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
from anndata import AnnData


def check_metrics(Z, base, prefix):
    assert Z.shape == base.shape

    cors = []
    errors = []
    for i in range(Z.shape[1]):
        cor, _ = pearsonr(Z[:, i], base[:, i])
        cors.append(cor)

        err = np.linalg.norm(Z[:, i] - base[:, i]) / np.linalg.norm(base[:, i])
        errors.append(err)

    print("For {name}, mean r = {cor:.4f}, mean L2 error = {err:.4f}.".format(name = prefix, cor = np.mean(cors), err = np.mean(errors)))
    np.savetxt("./result/{}_r.txt".format(prefix), cors)
    np.savetxt("./result/{}_L2.txt".format(prefix), errors)


def plot_umap(adata, Z_cpu, Z_gpu, Z_R, prefix, batch_key):
    if adata is not None:
        adata.obsm['X_cpu'] = Z_cpu
        adata.obsm['X_gpu'] = Z_gpu
        adata.obsm['X_harmony'] = Z_R

        pg.neighbors(adata, rep = 'cpu')
        pg.umap(adata, rep = 'cpu', out_basis = 'umap_cpu')

        pg.neighbors(adata, rep = 'gpu')
        pg.umap(adata, rep = 'gpu', out_basis = 'umap_gpu')

        pg.neighbors(adata, rep = 'harmony')
        pg.umap(adata, rep = 'harmony', out_basis = 'umap_harmony')

        pg.write_output(adata, "./result/{}_result".format(prefix))
    else:
        print("Use precalculated AnnData result.")

    if os.system("pegasus plot scatter --basis umap --attributes {attr} --alpha 0.5 ./result/{prefix}_result.h5ad ./plots/{prefix}.before.umap.pdf".format(attr = batch_key, prefix = prefix)):
        sys.exit(1)

    if os.system("pegasus plot scatter --basis umap_cpu --attributes {attr} --alpha 0.5 ./result/{prefix}_result.h5ad ./plots/{prefix}.cpu.umap.pdf".format(attr = batch_key, prefix = prefix)):
        sys.exit(1)

    if os.system("pegasus plot scatter --basis umap_gpu --attributes {attr} --alpha 0.5 ./result/{prefix}_result.h5ad ./plots/{prefix}.gpu.umap.pdf".format(attr = batch_key, prefix = prefix)):
        sys.exit(1)

    if os.system("pegasus plot scatter --basis umap_harmony --attributes {attr} --alpha 0.5 ./result/{prefix}_result.h5ad ./plots/{prefix}.harmony.umap.pdf".format(attr = batch_key, prefix = prefix)):
        sys.exit(1)


def test_cell_lines():
    print("Testing on Cell Lines...")

    z_files = [f for f in os.listdir("./result") if re.match("cell_lines.*_z.(txt|npy)", f)]
    if len(z_files) < 3 or not os.path.exists("./result/cell_lines_result.h5ad"):
        X = np.loadtxt("./data/cell_lines/pca.txt")
        df_metadata = pd.read_csv("./data/cell_lines/metadata.csv")

    if os.path.exists("./result/cell_lines_cpu_z.npy"):
        Z_cpu = np.load("./result/cell_lines_cpu_z.npy")
        print("Precalculated CPU mode result is loaded.")
    else:
        start_cpu = time.time()
        Z_cpu = harmonize(X, df_metadata, 'dataset')
        end_cpu = time.time()

        print("Time spent in CPU mode = {:.2f}s.".format(end_cpu - start_cpu))
        np.save("./result/cell_lines_cpu_z.npy", Z_cpu)

    if os.path.exists("./result/cell_lines_gpu_z.npy"):
        Z_gpu = np.load("./result/cell_lines_gpu_z.npy")
        print("Precalculated GPU mode result is loaded.")
    else:
        start_gpu = time.time()
        Z_gpu = harmonize(X, df_metadata, 'dataset', use_gpu = True)
        end_gpu = time.time()

        print("Time spent in GPU mode = {:.2f}s".format(end_gpu - start_gpu))
        np.save("./result/cell_lines_gpu_z.npy", Z_gpu)

    Z_R = np.loadtxt("./result/cell_lines_harmony_z.txt")

    check_metrics(Z_cpu, Z_R, prefix = "cell_lines_cpu")
    check_metrics(Z_gpu, Z_R, prefix = "cell_lines_gpu")

    if os.path.exists("./result/cell_lines_result.h5ad"):
        adata = None
    else:
        n_obs = X.shape[0]
        adata = AnnData(X = csr_matrix((n_obs, 2)), obs = df_metadata)
        adata.obsm['X_pca'] = X

        pg.neighbors(adata, rep = 'pca')
        pg.umap(adata)

    umap_list = [f for f in os.listdir("./plots") if re.match("cell_lines.*.pdf")]
    if len(umap_list) < 4:
        plot_umap(adata, Z_cpu, Z_gpu, Z_R, prefix = "cell_lines", batch_key = 'dataset')


def test_pbmc():
    print("Testing on 10x PBMC...")

    z_files = [f for f in os.listdir("./result") if re.match("pbmc.*_z.(txt|npy)", f)]
    if len(z_files) < 3
    adata = pg.read_input("./data/10x_pbmc/original_data.h5ad")

    if os.path.exists("./result/pbmc_cpu_z.npy"):
        Z_cpu = np.load("./result/pbmc_cpu_z.npy")
        print("Precalculated CPU mode result is loaded.")
    else:
        start_cpu = time.time()
        Z_cpu = harmonize(adata.obsm['X_pca'], adata.obs, 'Channel')
        end_cpu = time.time()

        print("Time spent in CPU mode = {:.2f}s.".format(end_cpu - start_cpu))
        np.save("./result/pbmc_cpu_z.npy", Z_cpu)

    if os.path.exists("./result/pbmc_gpu_z.npy"):
        Z_gpu = np.load("./result/pbmc_gpu_z.npy")
        print("Precalculated GPU mode result is loaded.")
    else:
        start_gpu = time.time()
        Z_gpu = harmonize(adata.obsm['X_pca'], adata.obs, 'Channel', use_gpu = True)
        end_gpu = time.time()

        print("Time spent in GPU mode = {:.2f}s".format(end_gpu - start_gpu))
        np.save("./result/pbmc_gpu_z.npy", Z_gpu)

    Z_R = np.loadtxt("./result/pbmc_harmony_z.txt")

    check_metrics(Z_cpu, Z_R, prefix = "pbmc_cpu")
    check_metrics(Z_gpu, Z_R, prefix = "pbmc_gpu")

    if os.path.exists("./result/pbmc_result.h5ad"):
        adata = None

    umap_list = [f for f in os.listdir("./plots") if re.match("pbmc.*.pdf", f)]
    if len(umap_list) < 4:
        plot_umap(adata, Z_cpu, Z_gpu, Z_R, prefix = "pbmc", batch_key = 'Channel')


def test_mantonbm():
    print("Testing on MantonBM...")

    z_files = [f for f in os.listdir("./result") if re.match("MantonBM.*_z.(txt|npy)", f)]
    if len(z_files) < 3:
        adata = pg.read_input("./data/MantonBM/original_data.h5ad")
        adata.obs['Individual'] = pd.Categorical(adata.obs['Channel'].apply(lambda s: s.split('_')[0][-1]))

    if os.path.exists("./result/MantonBM_cpu_z.npy"):
        Z_cpu = np.load("./result/MantonBM_cpu_z.npy")
        print("Precalculated CPU mode result is loaded.")
    else:
        start_cpu = time.time()
        Z_cpu = harmonize(adata.obsm['X_pca'], adata.obs, 'Channel')
        end_cpu = time.time()

        print("Time spent in CPU mode = {:.2f}s.".format(end_cpu - start_cpu))
        np.save("./result/MantonBM_cpu_z.npy", Z_cpu)

    if os.path.exists("./result/MantonBM_gpu_z.npy"):
        Z_gpu = np.load("./result/MantonBM_gpu_z.npy")
        print("Precalculated GPU mode result is loaded.")
    else:
        start_gpu = time.time()
        Z_gpu = harmonize(adata.obsm['X_pca'], adata.obs, 'Channel', use_gpu = True)
        end_gpu = time.time()

        print("Time spent in GPU mode = {:.2f}s".format(end_gpu - start_gpu))
        np.save("./result/MantonBM_gpu_z.npy", Z_gpu)

    Z_R = np.loadtxt("./result/MantonBM_harmony_z.txt")

    check_metrics(Z_cpu, Z_R, prefix = "MantonBM_cpu")
    check_metrics(Z_gpu, Z_R, prefix = "MantonBM_gpu")

    if os.path.exists("./result/MantonBM_result.h5ad"):
        adata = None

    umap_list = [f for f in os.listdir("./plots") if re.match("MantonBM.*.pdf", f)]
    if len(umap_list) < 4:
        plot_umap(adata, Z_cpu, Z_gpu, Z_R, prefix = "MantonBM", batch_key = 'Individual')


if __name__ == '__main__':
    dataset = sys.argv[1]

    assert dataset in ['cell_lines', 'pbmc', 'MantonBM']
    if dataset == 'cell_lines':
        test_cell_lines()
    elif dataset == 'pbmc':
        test_pbmc()
    else:
        test_mantonbm()