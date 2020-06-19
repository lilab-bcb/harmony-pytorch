Harmony-Pytorch
---------------

|PyPI| |Python|

.. |PyPI| image:: https://img.shields.io/pypi/v/harmony-pytorch.svg
   :target: https://pypi.org/project/harmony-pytorch

.. |Python| image:: https://img.shields.io/pypi/pyversions/harmony-pytorch.svg
   :target: https://pypi.org/project/harmony-pytorch

This is a Pytorch implementation of Harmony algorithm on single-cell sequencing data integration. Please see [Ilya Korsunsky et al., 2019](https://www.nature.com/articles/s41592-019-0619-0) for details.

Installation
^^^^^^^^^^^^^

This package is published on PyPI::

	pip install harmony-pytorch


Usage
^^^^^^^^

General Case
##############

Given an embedding ``X`` as a N-by-d matrix in numpy array structure (N for number of cells, d for embedding components) and cell attributes as a Data Frame ``df_metadata``, use Harmony for data integration as the following::

	from harmony import harmonize
	Z = harmonize(X, df_metadata, batch_key = 'Channel')


where ``Channel`` is the attribute in ``df_metadata`` for batches.

Alternatively, if there are multiple attributes for batches, write::

	Z = harmonize(X, df_metadata, batch_key = ['Lab', 'Date'])

Input as MultimodalData Object
###############################

It's easy for Harmony-pytorch to work with count matrix data structure from `PegasusIO <https://pegasusio.readthedocs.io>`_ package. Let ``data`` be a MultimodalData object in Python::

    from harmony import harmonize
    Z = harmonize(data.obsm['X_pca'], data.obs, batch_key = 'Channel')
    data.obsm['X_pca_harmony'] = Z

This will calculate the harmonized PCA matrix for the default UnimodalData of ``data``.

Given a UnimodalData object ``unidata``, you can also use the code above to perform Harmony algorithm: simply substitute ``unidata`` for ``data`` there.

Input as AnnData Object
##########################

It's easy for Harmony-pytorch to work with annotated count matrix data structure from `anndata <https://icb-anndata.readthedocs-hosted.com/en/stable/index.html>`_ package. Let ``adata`` be an AnnData object in Python::

	from harmony import harmonize
	Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = '<your-batch-key>')
	adata.obsm['X_harmony'] = Z

where ``<your-batch-key>`` should be replaced by the actual batch key attribute name in your data.

For details about ``AnnData`` data structure, please refer to its `documentation <https://icb-anndata.readthedocs-hosted.com/en/stable/anndata.AnnData.html>`_.
