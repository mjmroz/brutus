Core Module (``brutus.core``)
==============================

The core module contains fundamental stellar modeling utilities including isochrones, evolutionary tracks, SED computation, and neural network utilities.

.. currentmodule:: brutus.core

Individual Star Models
----------------------

.. autoclass:: EEPTracks
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: StarGrid
   :members:
   :undoc-members:
   :show-inheritance:

Population Models
-----------------

.. autoclass:: Isochrone
   :members:
   :undoc-members:
   :show-inheritance:

Neural Networks
---------------

.. autoclass:: FastNN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FastNNPredictor
   :members:
   :undoc-members:
   :show-inheritance:

SED Utilities
-------------

.. autofunction:: get_seds

.. autofunction:: _get_seds

Submodules
----------

For advanced users who need access to internal implementations:

.. automodule:: brutus.core.individual
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.core.populations
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.core.neural_nets
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.core.sed_utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: