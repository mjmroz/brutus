Priors Module (``brutus.priors``)
==================================

The priors module provides log-prior functions for Bayesian stellar parameter estimation, including stellar, astrometric, galactic structure, and extinction priors.

.. currentmodule:: brutus.priors

Stellar Priors
--------------

.. autofunction:: logp_imf

.. autofunction:: logp_ps1_luminosity_function

Astrometric Priors
------------------

.. autofunction:: logp_parallax

.. autofunction:: logp_parallax_scale

Galactic Structure Priors
--------------------------

.. autofunction:: logp_galactic_structure

Extinction Priors
-----------------

.. autofunction:: logp_extinction

Submodules
----------

For advanced users who need access to internal implementations:

.. automodule:: brutus.priors.stellar
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.priors.astrometric
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.priors.galactic
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.priors.extinction
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: