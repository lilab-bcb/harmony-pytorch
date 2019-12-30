from anndata import read_h5ad

adata = read_h5ad("../test/MantonBM_nonmix_tiny.h5ad")
print(adata.obsm['X_pca'].shape)