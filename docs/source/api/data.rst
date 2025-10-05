Data Module (``brutus.data``)
==============================

The data module manages all external data dependencies for brutus, including MIST stellar evolution grids, isochrones, 3-D dust maps, neural network weights, and photometric calibration offsets. It uses ``pooch`` for automatic downloading and caching of data files.

**Data Dependencies:**

brutus requires several types of external data:

1. **MIST Grids**: HDF5 files with stellar evolutionary tracks (~500 MB - 2 GB)
2. **MIST Isochrones**: Tabulated isochrones for population synthesis (~100 MB)
3. **Dust Maps**: HEALPix 3-D extinction maps (Bayestar19, ~2 GB)
4. **Neural Networks**: Trained weights for bolometric corrections (~10 MB)
5. **Photometric Offsets**: Empirical calibration tables (~1 MB)

**Automatic Data Management:**

Data files are automatically downloaded on first use and cached locally. The default cache location is ``~/.brutus/data/`` but can be configured via environment variables.

**Typical Usage:**

**First-time setup** (download all data files):

.. code-block:: python

   from brutus.data import fetch_grids, fetch_isos, fetch_dustmaps

   # Download MIST stellar evolution grids
   fetch_grids()  # Downloads default grids for common filter sets

   # Download MIST isochrones
   fetch_isos()

   # Download 3-D dust maps
   fetch_dustmaps()  # Downloads Bayestar19 by default

**Loading data for fitting**:

.. code-block:: python

   from brutus.data import load_models
   from brutus.core import StarGrid

   # Load pre-computed grid
   models, labels, params = load_models('grid_gaiadr3_2mass_wise.h5')
   grid = StarGrid(models, labels, params)

   # Grid is now ready for use with BruteForce fitter

**Custom data locations**:

.. code-block:: python

   # Load grid from custom path
   models, labels, params = load_models('/path/to/my_custom_grid.h5')

**Data File Formats:**

- **Grids**: HDF5 with datasets for models, labels, reddening coefficients
- **Isochrones**: HDF5 with structured arrays
- **Dust Maps**: HEALPix FITS files (via ``healpy``)
- **Neural Networks**: Pickle files with layer weights

**Storage Requirements:**

- Minimal installation (single grid): ~1-2 GB
- Full installation (all grids + dust maps): ~10-15 GB
- Custom grids only: ~1-5 GB depending on resolution

**See Also:**

- :doc:`/installation` - Setting up data files
- :doc:`/grid_generation` - Creating custom grids
- :doc:`/quickstart` - Basic data loading examples

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
