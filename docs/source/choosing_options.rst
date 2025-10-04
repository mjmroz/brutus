Configuration Guide: Choosing Options
======================================

This page provides guidance on selecting appropriate configuration options for brutus fitting, including model choices, prior settings, optimization parameters, and performance tuning.

Model Selection
---------------

Grid vs On-the-Fly Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use Pre-computed Grids** (``StarGrid`` + ``BruteForce``):

✓ Large samples (> 1000 stars)
✓ Standard filter combinations (Gaia, 2MASS, WISE, Pan-STARRS, SDSS)
✓ Speed is critical
✓ Publication-quality uncertainties needed

.. code-block:: python

   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   models, labels, params = load_models('grid_gaiadr3_2mass.h5')
   grid = StarGrid(models, labels, params)
   fitter = BruteForce(grid)

**Use On-the-Fly Models** (``StarEvolTrack`` / ``StellarPop``):

✓ Custom filter combinations
✓ Exploratory analysis
✓ Small samples (< 100 stars)
✓ Cluster modeling with MCMC

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i', 'z'])

Filter Selection
^^^^^^^^^^^^^^^^

**Minimum recommended**: 3-4 photometric bands spanning optical to near-IR

**Optimal combinations**:

- **Gaia + 2MASS**: G, BP, RP, J, H, Ks (6 bands, excellent for most stars)
- **Pan-STARRS**: g, r, i, z, y (5 bands, optical-only)
- **Full coverage**: Gaia + 2MASS + WISE (G, BP, RP, J, H, Ks, W1, W2 = 8 bands)

**Why multi-wavelength matters**:

- **Optical**: Sensitive to temperature
- **Near-IR**: Breaks distance-extinction degeneracy
- **Mid-IR**: Constrains cool stars and circumstellar material

.. code-block:: python

   # Create custom grid with specific filters
   from brutus.core import GridGenerator, EEPTracks

   tracks = EEPTracks()
   generator = GridGenerator(tracks, filters=['bp', 'g', 'rp', 'j', 'h', 'ks'])
   generator.make_grid('my_grid.h5')

Grid Parameters
---------------

Resolution Trade-offs
^^^^^^^^^^^^^^^^^^^^^^

**High resolution** (fine grid spacing):

- Mass: 500+ points
- EEP: 300+ points
- [Fe/H]: 40+ points
- Total: 5-10 million models

✓ Smooth posteriors
✓ Accurate parameter estimates
✗ Large files (5-10 GB)
✗ Slower fitting (more models to evaluate)

**Medium resolution** (default):

- Mass: 200-300 points
- EEP: 150-200 points
- [Fe/H]: 20-30 points
- Total: 1-3 million models

✓ Good balance
✓ Manageable file sizes (1-3 GB)
✓ Reasonable speed

**Low resolution** (coarse grid):

- Mass: 100-150 points
- EEP: 80-100 points
- [Fe/H]: 10-15 points
- Total: 200k-500k models

✓ Fast fitting
✓ Small files (< 500 MB)
✗ Discretization artifacts
✗ Less precise parameters

**Recommendation**: Start with medium resolution. Upgrade to high resolution for publication-quality results if artifacts are visible.

Parameter Coverage
^^^^^^^^^^^^^^^^^^

Ensure grid spans your targets:

.. code-block:: python

   generator.make_grid(
       output_file='custom_grid.h5',
       mini_range=(0.08, 150.0),    # Mass range (Msun)
       eep_range=(202, 808),         # Full evolutionary range
       feh_range=(-4.0, 0.5),        # Metallicity range (dex)
       afe_range=(-0.2, 0.6)         # Alpha enhancement range (dex)
   )

**Tips**:

- **Metal-poor stars**: Extend [Fe/H] to -4.0
- **Young stars**: Include pre-main-sequence (EEP < 353)
- **Giants**: Ensure coverage beyond EEP 454 (TAMS)
- **Low-mass**: Extend down to 0.08 Msun for M dwarfs

Prior Configuration
-------------------

Enabling/Disabling Priors
^^^^^^^^^^^^^^^^^^^^^^^^^^

Control which priors are applied:

.. code-block:: python

   from brutus.analysis import BruteForce

   fitter = BruteForce(
       grid,
       use_galactic_prior=True,   # Galactic structure prior (default: True)
       use_dust_prior=True,        # 3-D dust map prior (default: True)
       use_imf_prior=True          # IMF prior (default: True)
   )

**When to disable priors**:

- **Diagnostic purposes**: Test prior sensitivity
- **Non-Galactic objects**: Extra-galactic stars, satellite galaxies
- **Known unusual populations**: Very young clusters, special stellar types

.. warning::
   Disabling priors can lead to highly degenerate results. Only disable when you understand the implications.

Custom Prior Functions
^^^^^^^^^^^^^^^^^^^^^^

Advanced users can provide custom prior functions:

.. code-block:: python

   import numpy as np

   def custom_distance_prior(dist, gal_l, gal_b):
       """Custom distance prior for specific region."""
       # Example: Uniform in distance for Local Bubble
       if dist < 100.0:  # Within 100 pc
           return 0.0  # Log-prior (uniform)
       else:
           # Fall back to default Galactic prior
           from brutus.priors.galactic import logp_galactic_structure
           return logp_galactic_structure(dist, gal_l, gal_b)

   # Apply custom prior (requires modifying BruteForce internals)
   # See API documentation for details

Dust Map Selection
^^^^^^^^^^^^^^^^^^

Choose which 3-D dust map to use:

.. code-block:: python

   from brutus.dust.maps import use_dust_map

   # Use Bayestar19 (default)
   use_dust_map('bayestar19')

   # Alternatives (if available):
   # use_dust_map('bayestar17')
   # use_dust_map('3d_dust_map_custom')

**Considerations**:

- **Bayestar19**: Best for \|b\| > 5°, distances < 5 kpc
- **High latitudes**: Dust priors less important (low extinction)
- **Galactic plane**: Dust priors critical (high, variable extinction)

Optimization Settings
---------------------

Distance and Extinction Bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set bounds for optimization:

.. code-block:: python

   fitter = BruteForce(grid)

   results = fitter.fit(
       phot, phot_err,
       parallax=parallax, parallax_err=parallax_err,
       dist_bounds=(10.0, 10000.0),     # Distance range in pc
       av_max=5.0,                       # Maximum extinction in mag
       rv_bounds=(2.0, 6.0)              # R_V range
   )

**Guidelines**:

- **dist_bounds**: Set lower bound > 0 (default: 10 pc). Upper bound should exceed maximum plausible distance
- **av_max**: Use dust map maximum + margin (default: 10 mag)
- **rv_bounds**: Standard range is (2.0, 6.0). Narrow for well-understood sight lines

Likelihood Formulation
^^^^^^^^^^^^^^^^^^^^^^

Choose between different likelihood models:

.. code-block:: python

   results = fitter.fit(
       phot, phot_err,
       dim_prior=True    # Use chi-square formulation (default: True)
   )

**dim_prior=True** (chi-square with implicit distance prior):

.. math::

   \mathcal{L} \propto \exp(-\chi^2/2) \times d^2

✓ Appropriate for most individual star fitting
✓ Includes geometric volume factor

**dim_prior=False** (pure Gaussian):

.. math::

   \mathcal{L} \propto \exp(-\chi^2/2)

✓ Use when distance prior is explicitly included elsewhere
✓ Cluster fitting sometimes uses this

Convergence Tolerances
^^^^^^^^^^^^^^^^^^^^^^^

Adjust optimization tolerances:

.. code-block:: python

   results = fitter.fit(
       phot, phot_err,
       ftol=1e-4,    # Function tolerance (default: 1e-4)
       maxiter=1000  # Maximum iterations (default: 1000)
   )

**When to adjust**:

- **Increase maxiter**: If "optimization did not converge" warnings appear
- **Relax ftol**: For faster fitting with slightly less precision

Sampling Parameters
-------------------

Number of Posterior Samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Control the number of posterior samples returned:

.. code-block:: python

   results = fitter.fit(
       phot, phot_err,
       n_samples=10000  # Number of posterior samples (default: 10000)
   )

**Trade-offs**:

- **More samples** (50k-100k): Smoother posteriors, better tail coverage, slower
- **Fewer samples** (1k-5k): Faster, sufficient for medians but less precise percentiles

Importance Sampling
^^^^^^^^^^^^^^^^^^^

brutus uses importance sampling to draw from the posterior. The sampling is weighted by posterior probability, so regions of high probability are sampled more densely.

**No user configuration needed** for standard use. Advanced users can access the raw grid likelihoods:

.. code-block:: python

   # Get log-likelihoods for all grid points
   lnl_grid = fitter.compute_lnlike_grid(phot, phot_err)
   lnprior_grid = fitter.compute_lnprior_grid(gal_l, gal_b)
   lnpost_grid = lnl_grid + lnprior_grid

   # Custom sampling or analysis
   # ...

Performance Tuning
------------------

Parallelization
^^^^^^^^^^^^^^^

**Multi-star parallelization** (recommended for large samples):

.. code-block:: python

   from multiprocessing import Pool
   from brutus.analysis import BruteForce

   # Initialize fitter
   models, labels, params = load_models('grid.h5')
   grid = StarGrid(models, labels, params)
   fitter = BruteForce(grid)

   def fit_one_star(star_data):
       """Fit function for one star."""
       phot, phot_err, parallax, parallax_err = star_data
       return fitter.fit(phot, phot_err, parallax=parallax,
                        parallax_err=parallax_err)

   # Parallel execution
   with Pool(processes=32) as pool:
       results_list = pool.map(fit_one_star, star_data_list)

**Within-star parallelization** (not yet implemented):

Future versions may support multi-threading for grid evaluation within a single star fit.

Memory Management
^^^^^^^^^^^^^^^^^

For very large grids:

.. code-block:: python

   # Use memory-mapped HDF5 files (doesn't load full grid into RAM)
   models, labels, params = load_models('huge_grid.h5', memmap=True)
   grid = StarGrid(models, labels, params)

**Batch processing**:

.. code-block:: python

   # Process stars in batches to limit memory usage
   batch_size = 1000
   for i in range(0, len(star_catalog), batch_size):
       batch = star_catalog[i:i+batch_size]
       results_batch = [fitter.fit(s['phot'], s['phot_err']) for s in batch]
       # Save results_batch to disk
       # Clear memory

Caching
^^^^^^^

EEPTracks and Isochrone objects support caching:

.. code-block:: python

   from brutus.core import EEPTracks, Isochrone

   # Enable pickle caching (speeds up repeated loads)
   tracks = EEPTracks(use_cache=True)  # Creates .pkl cache file
   iso = Isochrone(use_cache=True)

   # Subsequent loads are much faster
   tracks2 = EEPTracks(use_cache=True)  # Loads from cache

**When useful**:

- Repeatedly loading same models in scripts
- Interactive sessions with multiple runs

Cluster Modeling Options
-------------------------

Grid Configuration for Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cluster fitting uses different grid parameters:

.. code-block:: python

   from brutus.analysis.populations import generate_isochrone_population_grid

   grid = generate_isochrone_population_grid(
       stellarpop=pop,
       feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0,
       smf_grid=None,              # Binary mass fraction grid (default: adaptive)
       eep_grid=None,              # EEP grid (default: 2000 points)
       mini_bound=0.08,            # Minimum mass (Msun)
       eep_binary_max=480.0,       # Max EEP for binaries (MS only)
       corr_params=None            # Empirical corrections
   )

**Custom SMF grid** (if you have knowledge about binary fraction):

.. code-block:: python

   import numpy as np

   # Fine sampling near equal-mass binaries
   smf_grid = np.concatenate([
       np.array([0.0]),           # Single stars
       np.linspace(0.2, 0.9, 8),  # Unequal mass
       np.linspace(0.9, 1.0, 5)   # Near equal mass (fine sampling)
   ])

Outlier Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^

Choose outlier model for field contamination:

.. code-block:: python

   from brutus.analysis.populations import isochrone_population_loglike

   lnl = isochrone_population_loglike(
       feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0, field_fraction=0.1,
       stellarpop=pop,
       obs_flux=flux, obs_err=flux_err,
       dim_prior=True,             # Chi-square cluster likelihood
       outlier_model='chisquare'  # or 'uniform' or custom function
   )

**Chi-square outlier** (default):
   Assumes outliers follow cluster model with extra scatter. Good for photometric binaries or cluster members with variable extinction.

**Uniform outlier**:
   Assigns constant low likelihood. More aggressive at excluding outliers. Good for clean clusters with known field contamination.

**Custom outlier**:
   Provide your own function based on known contaminant properties (e.g., field star color distribution).

MCMC Configuration
^^^^^^^^^^^^^^^^^^

When using emcee for cluster fitting:

.. code-block:: python

   import emcee

   ndim = 6  # [Fe/H], log(age), A_V, R_V, dist, field_frac
   nwalkers = 32  # Recommended: 2-4 × ndim
   nsteps = 5000  # Burn-in + production

   # Initialize walkers in small ball around guess
   initial = np.array([0.0, 9.0, 0.3, 3.1, 2000.0, 0.1])
   pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)

   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   sampler.run_mcmc(pos, nsteps, progress=True)

   # Diagnostics
   print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))
   # Target: 0.2-0.5

**Common issues**:

- **Low acceptance (<0.1)**: Step size too large or bad initial conditions
- **High acceptance (>0.7)**: Step size too small (slow convergence)
- **Check convergence**: Use ``emcee.autocorr`` to estimate autocorrelation time

Empirical Calibration Options
------------------------------

Applying Corrections
^^^^^^^^^^^^^^^^^^^^

Include empirical corrections:

.. code-block:: python

   # Define correction parameters
   corr_params = [
       dtdm,        # Temperature correction (K/Msun)
       drdm,        # Radius correction (Rsun/Msun)
       msto_smooth, # Smoothing parameter (Msun)
       feh_scale    # Metallicity scaling factor
   ]

   # Apply in grid generation
   generator.make_grid('grid_corrected.h5', corr_params=corr_params)

   # Apply in cluster modeling
   grid = generate_isochrone_population_grid(
       stellarpop=pop, feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0,
       corr_params=corr_params
   )

**When to apply**:

✓ Main-sequence stars with well-calibrated cluster corrections
✓ Publication-quality distance estimates
✗ Giants or post-MS stars (corrections may not apply)
✗ Very metal-poor stars (outside calibration range)

Photometric Offsets
^^^^^^^^^^^^^^^^^^^

Apply filter-specific photometric offsets:

.. code-block:: python

   # After fitting, apply offsets to model magnitudes
   model_mags_corrected = model_mags + offsets

   # Or include in likelihood (modify residuals)
   residuals_corrected = (obs_mags - model_mags) - offsets

See :doc:`photometric_offsets` for deriving survey-specific offsets.

Decision Tree: Configuration Quick Reference
---------------------------------------------

**For individual field stars**:

.. code-block:: python

   # Standard configuration
   grid = StarGrid(models, labels, params)
   fitter = BruteForce(
       grid,
       use_galactic_prior=True,
       use_dust_prior=True
   )
   results = fitter.fit(
       phot, phot_err,
       parallax=parallax, parallax_err=parallax_err,
       dim_prior=True,
       n_samples=10000
   )

**For stellar clusters**:

.. code-block:: python

   # MCMC approach
   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   def lnprob(theta):
       feh, loga, av, rv, dist, field_frac = theta
       return isochrone_population_loglike(
           feh=feh, loga=loga, av=av, rv=rv, dist=dist,
           field_fraction=field_frac,
           stellarpop=pop,
           obs_flux=flux, obs_err=flux_err,
           parallax=plx, parallax_err=plx_err,
           cluster_prob=0.9,
           dim_prior=True,
           outlier_model='chisquare'
       )

   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   sampler.run_mcmc(pos, nsteps, progress=True)

**For large surveys** (millions of stars):

.. code-block:: python

   # Use coarse grid for speed
   generator.make_grid(
       'fast_grid.h5',
       mini_range=(0.1, 10.0),  # Limit to dwarfs/subgiants
       eep_range=(300, 500),    # Main sequence only
       feh_range=(-1.0, 0.5)    # Solar neighborhood
   )

   # Parallelize across stars
   with Pool(processes=64) as pool:
       results = pool.map(fit_one_star, star_list)

Summary
-------

Key configuration decisions:

1. **Model type**: Grid (fast, fixed filters) vs on-the-fly (flexible, slower)
2. **Grid resolution**: High (precise, slow) vs medium (balanced) vs low (fast, artifacts)
3. **Priors**: Full Galactic model (default) vs custom vs disabled
4. **Likelihood**: Chi-square (dim_prior=True, default) vs Gaussian
5. **Sampling**: 10k samples (default) vs more (smooth) vs fewer (fast)
6. **Calibration**: Empirical corrections (recommended) vs raw models

For most applications, the **defaults are sensible**. Customize when you understand the trade-offs.

Next Steps
----------

- Understand your results: :doc:`understanding_results`
- Review common questions: :doc:`faq`
- See complete API reference: :doc:`api/index`

References
----------

- Speagle et al. (2025), arXiv:2503.02227
