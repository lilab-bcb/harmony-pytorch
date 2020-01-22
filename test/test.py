import numpy as np
import pandas as pd
import pegasus as pg
import seaborn as sns
import matplotlib.pyplot as plt

import os, sys, time, re

from harmony import harmonize
from harmonypy import run_harmony
from anndata import AnnData
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix


metric_dict = {'r': 'Correlation', 'L2': 'L2 Error'}

def check_metric(Z_torch, Z_py, Z_R, prefix, norm):
    assert Z_torch.shape == Z_py.shape and Z_py.shape == Z_R.shape

    metric_torch = []
    for i in range(Z_torch.shape[1]):
        m = get_measure(Z_torch[:, i], Z_R[:, i], norm)
        metric_torch.append(m)

    print("Mean {metric} by harmony-pytorch = {value:.4f}".format(metric = metric_dict[norm], value = np.mean(metric_torch)))
    np.savetxt("./result/{prefix}_{metric}_torch.txt".format(prefix = prefix, metric = norm), metric_torch)

    metric_py = []
    for i in range(Z_py.shape[1]):
        m = get_measure(Z_py[:, i], Z_R[:, i], norm)
        metric_py.append(m)

    print("Mean {metric} by harmonypy = {value:.4f}".format(metric = metric_dict[norm], value = np.mean(metric_py)))
    np.savetxt("./result/{prefix}_{metric}_py.txt".format(prefix = prefix, metric = norm), metric_py)


def get_measure(x, base, norm):
    assert norm in ['r', 'L2']

    if norm == 'r':
        corr, _ = pearsonr(x, base)
        return corr
    else:
        return np.linalg.norm(x - base) / np.linalg.norm(base)


def plot_umap(adata, Z_torch, Z_py, Z_R, prefix, batch_key):
    if adata is not None:
        adata.obsm['X_torch'] = Z_torch
        adata.obsm['X_py'] = Z_py
        adata.obsm['X_harmony'] = Z_R

        pg.neighbors(adata, rep = 'torch')
        pg.umap(adata, rep = 'torch', out_basis = 'umap_torch')

        pg.neighbors(adata, rep = 'py')
        pg.umap(adata, rep = 'py', out_basis = 'umap_py')

        pg.neighbors(adata, rep = 'harmony')
        pg.umap(adata, rep = 'harmony', out_basis = 'umap_harmony')

        pg.write_output(adata, "./result/{}_result".format(prefix))
    else:
        print("Use precalculated AnnData result.")

    if os.system("pegasus plot scatter --basis umap --attributes {attr} --alpha 0.5 ./result/{name}_result.h5ad ./plots/{name}.before.umap.pdf".format(name = prefix, attr = batch_key)):
        sys.exit(1)

    if os.system("pegasus plot scatter --basis umap_torch --attributes {attr} --alpha 0.5 ./result/{name}_result.h5ad ./plots/{name}.torch.umap.pdf".format(name = prefix, attr = batch_key)):
        sys.exit(1)

    if os.system("pegasus plot scatter --basis umap_py --attributes {attr} --alpha 0.5 ./result/{name}_result.h5ad ./plots/{name}.py.umap.pdf".format(name = prefix, attr = batch_key)):
        sys.exit(1)

    if os.system("pegasus plot scatter --basis umap_harmony --attributes {attr} --alpha 0.5 ./result/{name}_result.h5ad ./plots/{name}.harmony.umap.pdf".format(name = prefix, attr = batch_key)):
        sys.exit(1)


def test_cell_lines():
    print("Testing on cell lines dataset...")

    z_files = [f for f in os.listdir("./result") if re.match("cell_lines.*_z.(txt|npy)", f)]
    if len(z_files) < 3 or not os.path.exists("./result/cell_lines_result.h5ad"):
        X = np.loadtxt("./data/cell_lines/pca.txt")
        df_metadata = pd.read_csv("./data/cell_lines/metadata.csv")
        source_loaded = True

    if os.path.exists("./result/cell_lines_torch_z.npy"):
        Z_torch = np.load("./result/cell_lines_torch_z.npy")
        print("Precalculated embedding by harmony-pytorch is loaded.")
    else:
        start_torch = time.time()
        Z_torch = harmonize(X, df_metadata, batch_key = 'dataset')
        end_torch = time.time()

        print("Time spent for harmony-pytorch = {:.2f}s.".format(end_torch - start_torch))
        np.save("./result/cell_lines_torch_z.npy", Z_torch)

    if os.path.exists("./result/cell_lines_py_z.npy"):
        Z_py = np.load("./result/cell_lines_py_z.npy")
        print("Precalculated embedding by harmonypy is loaded.")
    else:
        start_py = time.time()
        ho = run_harmony(X, df_metadata, ['dataset'])
        end_py = time.time()

        print("Time spent for harmonypy = {:.2f}s.".format(end_py - start_py))
        print(ho.objective_harmony)

        Z_py = np.transpose(ho.Z_corr)
        np.save("./result/cell_lines_py_z.npy", Z_py)

    Z_R = np.loadtxt("./result/cell_lines_harmony_z.txt")

    check_metric(Z_torch, Z_py, Z_R, prefix = "cell_lines", norm = 'r')
    check_metric(Z_torch, Z_py, Z_R, prefix = "cell_lines", norm = 'L2')

    if os.path.exists("./result/cell_lines_result.h5ad"):
        adata = None
    else:
        n_obs = X.shape[0]
        adata = AnnData(X = csr_matrix((n_obs, 2)), obs = df_metadata)
        adata.obsm['X_pca'] = X

        pg.neighbors(adata, rep = 'pca')
        pg.umap(adata)

    umap_list = [f for f in os.listdir("./plots") if re.match("cell_lines.*.pdf", f)]
    if len(umap_list) < 4:
        plot_umap(adata, Z_torch, Z_py, Z_R, prefix = "cell_lines", batch_key = "dataset")

    if os.path.exists("./result/cell_lines_result.h5ad"):
       adata = pg.read_input("./result/cell_lines_result.h5ad", h5ad_mode = 'r')

       stat, pvalue, ac_rate = pg.calc_kBET(adata, attr = 'dataset', rep = 'harmony')
       print("kBET for Harmony: statistic = {stat}, p-value = {pval}, ac rate = {ac_rate}".format(stat = stat, pval = pvalue, ac_rate = ac_rate))

       stat, pvalue, ac_rate = pg.calc_kBET(adata, attr = 'dataset', rep = 'py')
       print("kBET for harmonypy: statistic = {stat}, p-value = {pval}, ac rate = {ac_rate}".format(stat = stat, pval = pvalue, ac_rate = ac_rate))

       stat, pvalue, ac_rate = pg.calc_kBET(adata, attr = 'dataset', rep = 'torch')
       print("kBET for harmony-pytorch: statistic = {stat}, p-value = {pval}, ac rate = {ac_rate}".format(stat = stat, pval = pvalue, ac_rate = ac_rate))


def test_pbmc():
    print("Testing on 10x pbmc dataset...")

    z_files = [f for f in os.listdir("./result") if re.match("pbmc.*_z.(txt|npy)", f)]
    if len(z_files) < 3 or not os.path.exists("./result/pbmc_result.h5ad"):
        adata = pg.read_input("./data/10x_pbmc/original_data.h5ad")

    if os.path.exists("./result/pbmc_torch_z.npy"):
        Z_torch = np.load("./result/pbmc_torch_z.npy")
        print("Precalculated embedding by harmony-pytorch is loaded.")
    else:
        start_torch = time.time()
        Z_torch = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = 'Channel')
        end_torch = time.time()

        print("Time spent for harmony-pytorch = {:.2f}s.".format(end_torch - start_torch))
        np.save("./result/pbmc_torch_z.npy", Z_torch)

    if os.path.exists("./result/pbmc_py_z.npy"):
        Z_py = np.load("./result/pbmc_py_z.npy")
        print("Precalculated embedding by harmonypy is loaded.")
    else:
        start_py = time.time()
        ho = run_harmony(adata.obsm['X_pca'], adata.obs, ['Channel'])
        end_py = time.time()

        print(ho.objective_harmony)
        print("Time spent for harmonypy = {:.2f}s.".format(end_py - start_py))

        Z_py = np.transpose(ho.Z_corr)
        np.save("./result/pbmc_py_z.npy", Z_py)

    Z_R = np.loadtxt("./result/pbmc_harmony_z.txt")

    check_metric(Z_torch, Z_py, Z_R, prefix = "pbmc", norm = 'r')
    check_metric(Z_torch, Z_py, Z_R, prefix = "pbmc", norm = 'L2')

    if os.path.exists("./result/pbmc_result.h5ad"):
        adata = None

    umap_list = [f for f in os.listdir("./plots") if re.match("pbmc.*.pdf", f)]
    if len(umap_list) < 4:
        plot_umap(adata, Z_torch, Z_py, Z_R, prefix = "pbmc", batch_key = "Channel")


def test_mantonbm():
    print("Testing on MantonBM dataset...")

    z_files = [f for f in os.listdir("./result") if re.match("MantonBM.*_z.(txt|npy)", f)]
    if len(z_files) < 3 or not os.path.exists("./result/MantonBM_result.h5ad"):
        adata = pg.read_input("./data/MantonBM/original_data.h5ad")
        adata.obs['Individual'] = pd.Categorical(adata.obs['Channel'].apply(lambda s: s.split('_')[0][-1]))

    if os.path.exists("./result/MantonBM_torch_z.npy"):
        Z_torch = np.load("./result/MantonBM_torch_z.npy")
        print("Precalculated embedding by harmony-pytorch is loaded.")
    else:
        start_torch = time.time()
        Z_torch = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = 'Channel')
        end_torch = time.time()

        print("Time spent for harmony-pytorch = {:.2f}s.".format(end_torch - start_torch))
        np.save("./result/MantonBM_torch_z.npy", Z_torch)

    if os.path.exists("./result/MantonBM_py_z.npy"):
        Z_py = np.load("./result/MantonBM_py_z.npy")
        print("Precalculated embedding by harmonypy is loaded.")
    else:
        start_py = time.time()
        ho = run_harmony(adata.obsm['X_pca'], adata.obs, ['Channel'])
        end_py = time.time()

        print("Time spent for harmonypy = {:.2f}s.".format(end_py - start_py))

        Z_py = np.transpose(ho.Z_corr)
        np.save("./result/MantonBM_py_z.npy", Z_py)


    Z_R = np.loadtxt("./result/MantonBM_harmony_z.txt")

    check_metric(Z_torch, Z_py, Z_R, prefix = "MantonBM", norm = 'r')
    check_metric(Z_torch, Z_py, Z_R, prefix = "MantonBM", norm = 'L2')

    if os.path.exists("./result/MantonBM_result.h5ad"):
        adata = None

    umap_list = [f for f in os.listdir("./plots") if re.match("MantonBM.*.pdf", f)]
    if len(umap_list) < 4:
        plot_umap(adata, Z_torch, Z_py, Z_R, prefix = "MantonBM", batch_key = "Individual")


def gen_plot(norm):

    # Cell Lines
    metric_celllines_torch = np.loadtxt("./result/cell_lines_{}_torch.txt".format(norm))
    metric_celllines_py = np.loadtxt("./result/cell_lines_{}_py.txt".format(norm))

    df1 = pd.DataFrame({'dataset' : np.repeat(['Cell Lines'], metric_celllines_torch.size + metric_celllines_py.size),
                        'package' : np.concatenate((np.repeat(['Torch'], metric_celllines_torch.size),
                                                   np.repeat(['Py'], metric_celllines_py.size)), axis = 0),
                        'metric' : np.concatenate((metric_celllines_torch, metric_celllines_py), axis = 0)})

    # PBMC
    metric_pbmc_torch = np.loadtxt("./result/pbmc_{}_torch.txt".format(norm))
    metric_pbmc_py = np.loadtxt("./result/pbmc_{}_py.txt".format(norm))

    df2 = pd.DataFrame({'dataset' : np.repeat(['10x PBMC'], metric_pbmc_torch.size + metric_pbmc_py.size),
                        'package' : np.concatenate((np.repeat(['Torch'], metric_pbmc_torch.size),
                                                    np.repeat(['Py'], metric_pbmc_py.size)), axis = 0),
                        'metric' : np.concatenate((metric_pbmc_torch, metric_pbmc_py), axis = 0)})

    # MantonBM
    metric_mantonbm_torch = np.loadtxt("./result/MantonBM_{}_torch.txt".format(norm))
    metric_mantonbm_py = np.loadtxt("./result/MantonBM_{}_py.txt".format(norm))

    df3 = pd.DataFrame({'dataset' : np.repeat(['Bone Marrow'], metric_mantonbm_torch.size + metric_mantonbm_py.size),
                        'package' : np.concatenate((np.repeat(['Torch'], metric_mantonbm_torch.size),
                                                    np.repeat(['Py'], metric_mantonbm_py.size)), axis = 0),
                        'metric' : np.concatenate((metric_mantonbm_torch, metric_mantonbm_py), axis = 0)})

    df = pd.concat([df1, df2, df3])

    # Plot
    ax = sns.violinplot(x = "dataset", y = "metric", hue = "package", data = df, palette = "muted", split = True, cut = 0)
    ax.set_title("{} between Harmonypy and Harmony-pytorch Integration".format(metric_dict[norm]))
    ax.set(xlabel = 'Dataset', ylabel = "{} on PCs".format(metric_dict[norm]))
    if norm == 'r':
        ax.set(ylim = (0.98, 1.001))
    else:
        ax.set(ylim = (0, 0.1))
    figure = ax.get_figure()
    legend_loc = 'lower right' if norm == 'r' else 'upper right'
    figure.get_axes()[0].legend(title = "Package", loc = legend_loc)
    figure.savefig("./plots/{}_stats.png".format(norm), dpi = 400)
    plt.close()


if __name__ == '__main__':
    dataset = sys.argv[1]

    assert dataset in ["cell_lines", "pbmc", "MantonBM", "plot"]

    if not os.path.exists("./result"):
        if os.system("mkdir ./result"):
            sys.exit(1)

    if not os.path.exists("./plots"):
        if os.system("mkdir ./plots"):
            sys.exit(1)

    if dataset == 'cell_lines':
        test_cell_lines()
    elif dataset == 'pbmc':
        test_pbmc()
    elif dataset == 'MantonBM':
        test_mantonbm()
    else:
        gen_plot('r')
        gen_plot('L2')
