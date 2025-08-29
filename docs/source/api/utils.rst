Utilities Module (``brutus.utils``)
===================================

The utils module contains mathematical, photometric, and sampling utilities that support the core functionality of brutus.

.. currentmodule:: brutus.utils

Photometry Functions
--------------------

.. autofunction:: magnitude

.. autofunction:: inv_magnitude

.. autofunction:: luptitude

.. autofunction:: inv_luptitude

.. autofunction:: add_mag

.. autofunction:: phot_loglike

Mathematical Functions
----------------------

.. autofunction:: _function_wrapper

.. autofunction:: inverse3

.. autofunction:: isPSD

.. autofunction:: chisquare_logpdf

.. autofunction:: truncnorm_pdf

.. autofunction:: truncnorm_logpdf

Sampling Utilities
------------------

.. autofunction:: quantile

.. autofunction:: sample_multivariate_normal

.. autofunction:: draw_sar

Submodules
----------

For advanced users who need access to internal implementations:

.. automodule:: brutus.utils.math
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.utils.photometry
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.utils.sampling
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: