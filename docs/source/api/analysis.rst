Analysis Module (``brutus.analysis``)
======================================

The analysis module contains advanced analysis workflows and statistical methods including individual star fitting, population analysis, and photometric offset computation.

.. currentmodule:: brutus.analysis

Individual Star Fitting
------------------------

.. autoclass:: BruteForce
   :members:
   :undoc-members:
   :show-inheritance:

Population Analysis
-------------------

.. autofunction:: isochrone_population_loglike

.. autofunction:: generate_isochrone_population_grid

.. autofunction:: compute_isochrone_cluster_loglike

.. autofunction:: compute_isochrone_outlier_loglike

.. autofunction:: apply_isochrone_mixture_model

.. autofunction:: marginalize_isochrone_grid

Photometric Offsets
-------------------

.. autoclass:: PhotometricOffsetsConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: photometric_offsets

Line-of-Sight Dust
-------------------

.. autofunction:: los_clouds_priortransform

.. autofunction:: los_clouds_loglike_samples

.. autofunction:: kernel_tophat

.. autofunction:: kernel_gauss

.. autofunction:: kernel_lorentz

