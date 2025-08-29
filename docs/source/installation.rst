Installation
============

Requirements
------------

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL (see Windows note below)

Quick Install
-------------

For most users, install from PyPI:

.. code-block:: bash

   pip install astro-brutus

Development Install
-------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/joshspeagle/brutus.git
   cd brutus
   pip install -e ".[dev]"

Windows Users - Important Note
-------------------------------

⚠️ **Windows Compatibility**: Due to the ``healpy`` dependency (required for dust mapping), brutus does not work reliably on native Windows. **Windows users should install and run brutus in WSL (Windows Subsystem for Linux)**.

Alternative Windows installation options:

- **WSL (Recommended)**: Install Ubuntu or another Linux distribution via WSL and use the standard installation
- **Conda**: Try ``conda install -c conda-forge astro-brutus`` which may have pre-compiled Windows wheels
- **Docker**: Use a Linux-based Docker container

Conda Installation
------------------

If you use conda, you can install from conda-forge:

.. code-block:: bash

   conda install -c conda-forge astro-brutus

Dependencies
------------

Core dependencies that will be automatically installed:

- ``numpy`` (≥1.19) - Numerical computing
- ``scipy`` (≥1.6) - Scientific computing  
- ``matplotlib`` (≥3.3) - Plotting
- ``h5py`` (≥3.0) - HDF5 file support
- ``healpy`` (≥1.14) - HEALPix utilities for incorporating dust maps
- ``numba`` (≥0.53) - Just-in-time compilation for performance
- ``pooch`` (≥1.4) - Data downloading and management

Testing the Installation
-------------------------

To verify your installation works correctly:

.. code-block:: python

   import brutus
   print(f"brutus version: {brutus.__version__}")

   # Test core functionality
   from brutus import Isochrone, load_models
   print("Installation successful!")