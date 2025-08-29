Quick Start Guide
=================

This guide provides a quick introduction to using brutus for common workflows.

Individual Star Fitting
------------------------

The most common use case is fitting stellar parameters for individual stars:

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

Isochrone Generation
--------------------

Generate stellar parameters for stellar populations:

.. code-block:: python

   from brutus import Isochrone

   # Create isochrone generator
   iso = Isochrone()

   # Generate stellar parameters for an isochrone
   params = iso.get_predictions(
       feh=0.0,        # Solar metallicity [Fe/H]
       afe=0.0,        # Solar alpha enhancement [alpha/Fe] 
       loga=9.0        # 1 Gyr age (log10(age/yr))
   )

Data Management
---------------

Download and manage stellar evolution data:

.. code-block:: python

   from brutus import fetch_grids, fetch_isos, fetch_dustmaps

   # Download stellar evolution grids  
   fetch_grids()

   # Download isochrone data
   fetch_isos()

   # Download 3D dust maps
   fetch_dustmaps()

Working with Results
--------------------

Brutus provides comprehensive posterior distributions for all fitted parameters:

.. code-block:: python

   # Access posterior samples
   distances = results['dist_samples']
   extinctions = results['av_samples'] 
   stellar_params = results['stellar_params']

   # Plot results
   from brutus.plotting import cornerplot
   cornerplot(results, show_titles=True)

Common Workflows
----------------

For typical research workflows:

1. **Download data** using ``fetch_*`` functions
2. **Load models** using ``load_models``
3. **Create fitting objects** (``BruteForce``, ``Isochrone``)
4. **Fit your data** and analyze results
5. **Visualize results** using plotting utilities

Next Steps
----------

- See the :doc:`tutorials` for detailed examples
- Check the :doc:`api/index` for complete function documentation
- View the `tutorials/` directory for Jupyter notebook examples