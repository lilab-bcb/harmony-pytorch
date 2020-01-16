Harmony-Pytorch
---------------

This is a Pytorch implementation of Harmony algorithm on single-cell sequencing data integration. Please see [Ilya Korsunsky et al., 2019](https://www.nature.com/articles/s41592-019-0619-0) for details.

Installation
^^^^^^^^^^^^^

This package is published on PyPI::

	pip install harmony-pytorch


Usage
^^^^^^^^

General Case
##############

Given an embedding ``X`` as a N-by-d matrix (N for number of cells, d for embedding components) and cell attributes as a Data Frame ``df_metadata``, use Harmony for data integration as the following::

	from harmony import harmonize
	Z = harmonize(X, df_metadata, batch_key = 'Channel')


where ``Channel`` is the attribute in ``df_metadata`` for batches. 

Alternatively, if there are multiple attributes for batches, write::

	Z = harmonize(X, df_metadata, batch_key = ['Lab', 'Date'])


Input as AnnData Object
##########################

It's easy for Harmony-pytorch to work with annotated count matrix data structure from `anndata <https://icb-anndata.readthedocs-hosted.com/en/stable/index.html>`_ package. Let ``adata`` be an AnnData object in Python::

	from harmony import harmonize
	Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = 'Channel')
	adata.obsm['X_harmony'] = Z


For details about ``AnnData`` data structure, please refer to its `documentation <https://icb-anndata.readthedocs-hosted.com/en/stable/anndata.AnnData.html>`_.