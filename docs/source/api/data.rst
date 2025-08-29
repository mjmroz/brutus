Data Module (``brutus.data``)
==============================

The data module provides functions for downloading and loading stellar evolution models, isochrones, dust maps, and other data files required by brutus.

.. currentmodule:: brutus.data

Data Downloading
----------------

.. autofunction:: fetch_grids

.. autofunction:: fetch_isos

.. autofunction:: fetch_tracks

.. autofunction:: fetch_dustmaps

.. autofunction:: fetch_offsets

.. autofunction:: fetch_nns

Data Loading
------------

.. autofunction:: load_models

.. autofunction:: load_offsets

Submodules
----------

For advanced users who need access to internal implementations:

.. automodule:: brutus.data.download
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.data.loader
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: