brutus Documentation
====================

*Et tu, Brute?*

**brutus** is a Pure Python package for **"brute force" Bayesian inference** to derive distances, reddenings, and stellar properties from photometry. The package is designed to be highly modular and user-friendly, with comprehensive support for modeling individual stars, star clusters, and 3-D dust mapping.

.. image:: https://github.com/joshspeagle/brutus/blob/master/brutus_logo.png?raw=true
   :alt: brutus logo
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Key Features
============

🌟 **Individual Star Modeling**: Fit distances, reddenings, and stellar properties for individual stars using Bayesian inference

🌟 **Cluster Analysis**: Model stellar clusters with consistent ages, metallicities, and distances

🌟 **3D Dust Mapping**: Integrate with 3D dust maps and model extinction along lines of sight

🌟 **Modern Stellar Models**: Built-in support for MIST isochrones and evolutionary tracks

🌟 **Flexible & Fast**: Optimized algorithms with numba acceleration and modular design

🌟 **Publication Ready**: Designed for ease of use in research workflows

Quick Start
===========

Install brutus from PyPI:

.. code-block:: bash

   pip install astro-brutus

For individual star fitting:

.. code-block:: python

   import numpy as np
   from brutus import BruteForce, StarGrid, load_models

   # Load stellar models
   models, labels = load_models('path/to/models.h5')

   # Create a StarGrid
   star_grid = StarGrid(models, labels)

   # Set up the fitter
   fitter = BruteForce(star_grid)

   # Your photometry data (flux units)
   photometry = np.array([1.2e-3, 0.8e-3, 0.6e-3])  # g, r, i bands
   errors = np.array([1e-5, 1e-5, 1e-5])

   # Fit the star
   results = fitter.fit(photometry, errors, parallax=2.5, parallax_err=0.1)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`