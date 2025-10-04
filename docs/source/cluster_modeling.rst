Stellar Cluster Modeling
=========================

This page explains how brutus models **coeval stellar populations** such as open clusters, globular clusters, and star-forming regions. The key innovation is the **mixture-before-marginalization** approach that properly handles field contamination.

Coeval Population Assumptions
------------------------------

Stellar clusters provide powerful constraints because all member stars share:

- **Common age**: All stars formed at the same time
- **Common metallicity**: All stars formed from the same gas cloud
- **Common distance**: Cluster size (few pc to ~100 pc) is negligible compared to Earth distance
- **Common extinction**: Foreground dust affects all members similarly (though differential extinction can vary)

These shared properties allow us to fit a **single isochrone** to the entire cluster population, rather than treating each star independently.

The Isochrone Fitting Problem
------------------------------

Given photometry for :math:`N` stars in a cluster, we want to infer the population parameters:

.. math::

   \Theta = ([{\rm Fe/H}], \log_{10}({\rm age}), A_V, R_V, d)

For each star :math:`i`, the observed photometry :math:`\mathbf{F}_i` depends on:

- **Population parameters** :math:`\Theta` (shared by all stars)
- **Stellar mass** :math:`M_i` (varies between stars)
- **Binary companion**: Secondary mass fraction :math:`{\rm SMF}_i` (varies between stars)

The challenge is that we don't know each star's mass—we must **marginalize over mass** to get the population likelihood.

Naive Marginalization (Wrong!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common but **incorrect** approach is to marginalize first, then mix:

.. math::

   \mathcal{L}_{\rm naive}(\Theta) = \prod_{i=1}^N \left[ \int \mathcal{L}({\bf F}_i | M, \Theta) \, \pi(M) \, dM \right]

This treats each star independently and then multiplies likelihoods. The problem: This doesn't properly account for field contamination because outliers are mixed in at the product level, not at the integral level.

Mixture-Before-Marginalization (Correct!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mathematically correct approach is to **apply the mixture model before marginalizing** over mass:

.. math::

   \mathcal{L}(\Theta) = \prod_{i=1}^N \left[ \int \left( w_{\rm mem} \mathcal{L}_{\rm cluster}({\bf F}_i | M, \Theta) + w_{\rm field} \mathcal{L}_{\rm outlier}({\bf F}_i) \right) \pi(M) \, dM \right]

where:

- :math:`w_{\rm mem} = 1 - f_{\rm field}` is the cluster membership probability
- :math:`w_{\rm field} = f_{\rm field}` is the field contamination fraction
- :math:`\mathcal{L}_{\rm cluster}` is the isochrone model likelihood
- :math:`\mathcal{L}_{\rm outlier}` is the outlier/field star likelihood
- :math:`\pi(M)` is the IMF prior over mass

**Key insight**: Each star's data is explained as a mixture of cluster member and field contaminant **before** integrating over the unknown mass. This properly accounts for the fact that field stars don't follow the cluster isochrone.

Why This Matters
^^^^^^^^^^^^^^^^^

Mixture-after-marginalization can produce **severely biased** results:

1. **Overestimated ages**: Field contamination by older red giants biases age estimates high
2. **Incorrect metallicities**: Field stars with different [Fe/H] bias population metallicity
3. **Wrong distances**: Outliers at different distances bias ensemble distance estimates

Mixture-before-marginalization correctly down-weights field contaminants during the mass marginalization, preventing these biases.

The brutus Cluster Fitting Workflow
------------------------------------

brutus implements mixture-before-marginalization through a grid-based approach. The workflow has six steps:

Step 1: Generate Isochrone Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For fixed population parameters :math:`\Theta = ([{\rm Fe/H}], \log_{10}({\rm age}), A_V, R_V, d)`, generate a grid over the *stellar* parameters (mass, SMF):

.. code-block:: python

   from brutus.analysis.populations import generate_isochrone_population_grid
   from brutus.core import Isochrone, StellarPop

   # Initialize population model
   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   # Generate grid for specific population parameters
   grid = generate_isochrone_population_grid(
       stellarpop=pop,
       feh=0.0,      # Solar metallicity
       loga=9.0,     # 1 Gyr (log10(age in years))
       av=0.5,       # 0.5 mag extinction
       rv=3.1,       # Standard R_V
       dist=2000.0   # 2 kpc distance
   )

   # Grid contains:
   # - grid['photometry']: Model fluxes, shape (N_grid, N_filters)
   # - grid['masses']: Stellar masses, shape (N_grid,)
   # - grid['smf_values']: Binary SMF values, shape (N_grid,)
   # - grid['mass_jacobians']: dm grid spacing, shape (N_grid,)
   # - grid['smf_jacobians']: d(SMF) grid spacing, shape (N_grid,)

The grid spans:

- **Mass**: From ~0.08 to maximum mass on isochrone at given age
- **SMF** (Secondary Mass Fraction): 0.0 (single stars) to 1.0 (equal mass binaries)

Grid points are computed by varying EEP along the isochrone and computing photometry for each (EEP, SMF) combination.

Step 2: Compute Cluster Likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each (grid_point, star) pair, compute the likelihood that the star is a cluster member:

.. code-block:: python

   from brutus.analysis.populations import compute_isochrone_cluster_loglike

   lnl_cluster = compute_isochrone_cluster_loglike(
       obs_flux=obs_flux,       # Shape (N_stars, N_filters)
       obs_err=obs_err,         # Shape (N_stars, N_filters)
       isochrone_grid=grid,
       parallax=parallax,       # Optional, shape (N_stars,)
       parallax_err=parallax_err,
       distance=2000.0,
       dim_prior=True           # Use chi-square likelihood
   )

   # Output shape: (N_grid_points, N_stars)

The cluster likelihood includes:

- **Photometric component**: :math:`\mathcal{L}_{\rm phot} \propto \exp(-\chi^2/2)` where :math:`\chi^2 = \sum_{\rm bands} (F_{\rm obs} - F_{\rm model})^2 / \sigma_F^2`
- **Parallax component** (if provided): :math:`\mathcal{L}_{\varpi} \propto \exp[-({\varpi}_{\rm obs} - 1000/d)^2 / 2\sigma_\varpi^2]`

.. note::
   The ``dim_prior=True`` option uses a chi-square formulation that includes an implicit distance prior :math:`\propto d^2`. This is appropriate for cluster fitting where distance is a fitted parameter.

Step 3: Compute Outlier Likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each star, compute the likelihood under the field contamination model:

.. code-block:: python

   from brutus.analysis.populations import compute_isochrone_outlier_loglike

   lnl_outlier = compute_isochrone_outlier_loglike(
       obs_flux=obs_flux,
       obs_err=obs_err,
       isochrone_grid=grid,     # Provides stellar parameter info if needed
       parallax=parallax,
       parallax_err=parallax_err,
       dim_prior=True,
       outlier_model='chisquare'  # or 'uniform' or custom function
   )

   # Output shape: (N_grid_points, N_stars) or (N_stars,) depending on model

brutus provides two built-in outlier models:

**Chi-square outlier model** (default):
   Assumes field stars follow the same photometric model but with **additional intrinsic scatter**:

   .. math::

      \mathcal{L}_{\rm outlier} \propto \exp\left(-\frac{\chi^2}{2(1 + \sigma_{\rm int}^2)}\right)

   This down-weights stars that are significantly discrepant from the isochrone without completely excluding them.

**Uniform outlier model**:
   Assigns constant (low) likelihood to all stars, representing ignorance about field star properties:

   .. math::

      \mathcal{L}_{\rm outlier} = {\rm const}

   More aggressive at excluding outliers but makes strong assumptions.

**Custom outlier model**:
   Users can provide a custom function that computes outlier likelihood based on observables and stellar parameters. Useful for incorporating known field star populations or specific contaminant models.

Step 4: Apply Mixture Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combine cluster and outlier likelihoods with mixture weights **before** marginalization:

.. code-block:: python

   from brutus.analysis.populations import apply_isochrone_mixture_model

   lnl_mixed = apply_isochrone_mixture_model(
       lnl_cluster=lnl_cluster,
       lnl_outlier=lnl_outlier,
       cluster_prob=0.8,        # External prior: 80% of stars are members
       field_fraction=0.1       # Fitted parameter: 10% field contamination
   )

   # Output shape: (N_grid_points, N_stars)

The mixture at each grid point is:

.. math::

   \mathcal{L}_{\rm mix}({\bf F}_i | M, {\rm SMF}, \Theta) = (1 - f_{\rm field}) \mathcal{L}_{\rm cluster} + f_{\rm field} \mathcal{L}_{\rm outlier}

where ``field_fraction`` is :math:`f_{\rm field}`.

**Two mixture parameters**:

- ``cluster_prob``: **External prior** on membership (e.g., from spatial/kinematic selection)
- ``field_fraction``: **Fitted parameter** representing contamination level

These are conceptually different: ``cluster_prob`` encodes prior knowledge about which stars are likely members (from CMD position, radial velocity, etc.), while ``field_fraction`` is a nuisance parameter that absorbs systematic contamination.

Step 5: Marginalize Over Mass and SMF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integrate the mixed likelihood over the (mass, SMF) grid:

.. code-block:: python

   from brutus.analysis.populations import marginalize_isochrone_grid

   lnl_marginalized = marginalize_isochrone_grid(
       lnl_mixed=lnl_mixed,
       isochrone_grid=grid
   )

   # Output shape: (N_stars,)

The marginalization uses proper geometric jacobians:

.. math::

   \mathcal{L}({\bf F}_i | \Theta) = \int \int \mathcal{L}_{\rm mix}({\bf F}_i | M, {\rm SMF}, \Theta) \, \pi(M) \, dM \, d({\rm SMF})

where the integrals are evaluated as discrete sums weighted by grid spacing (``mass_jacobians`` and ``smf_jacobians``).

**Jacobians are critical**: Unequal grid spacing must be accounted for to avoid biasing the marginalization. The jacobians represent the :math:`dM` and :math:`d({\rm SMF})` factors in the integral.

Step 6: Combine Across Stars
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The total population log-likelihood is the sum over all stars:

.. code-block:: python

   import numpy as np

   total_lnl = np.sum(lnl_marginalized)

This is the value returned to an external MCMC sampler (e.g., ``emcee``, ``dynesty``) that varies the population parameters :math:`\Theta`.

Binary Stars in Cluster Modeling
---------------------------------

Binary companions affect cluster photometry by adding flux from the secondary star. brutus models binaries via the **Secondary Mass Fraction (SMF)**:

.. math::

   {\rm SMF} = \frac{M_{\rm secondary}}{M_{\rm primary}}

where :math:`0 \leq {\rm SMF} \leq 1`. Values:

- **SMF = 0**: Single star (no companion)
- **SMF = 0.5**: Companion with half the primary mass
- **SMF = 1**: Equal-mass binary

Binary Photometry
^^^^^^^^^^^^^^^^^

The combined photometry is the sum of fluxes from both components:

.. math::

   F_{\rm total} = F_{\rm primary}(M_{\rm pri}, {\rm EEP}) + F_{\rm secondary}(M_{\rm sec}, {\rm EEP})

where :math:`M_{\rm sec} = {\rm SMF} \times M_{\rm pri}` and both stars have the same age, metallicity, distance, and extinction.

**Main sequence binaries**: Binary companions are only modeled for EEP ≤ 480 (roughly the main sequence turnoff). Post-main-sequence binaries involve complex evolution (mass transfer, common envelopes) that is not captured by simple flux addition.

Binary Fraction
^^^^^^^^^^^^^^^

The population grid includes SMF values from 0.0 to 1.0, effectively marginalizing over the binary population. If you want to model a specific binary fraction :math:`f_{\rm bin}`, you can weight the SMF prior:

.. code-block:: python

   # Modify grid generation to weight binaries
   smf_grid = np.array([0.0, 0.5, 0.7, 0.9, 1.0])  # Emphasize high SMF

   grid = generate_isochrone_population_grid(
       stellarpop=pop,
       feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0,
       smf_grid=smf_grid
   )

Alternatively, apply a binary fraction prior during marginalization (this requires modifying the ``marginalize_isochrone_grid`` function to include an SMF-dependent prior).

Complete Example: Fitting a Cluster
------------------------------------

Here's a full workflow for fitting cluster population parameters:

.. code-block:: python

   import numpy as np
   from brutus.core import Isochrone, StellarPop
   from brutus.analysis.populations import isochrone_population_loglike

   # Initialize models
   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   # Observed cluster data
   obs_flux = np.array([...])       # Shape (N_stars, N_filters)
   obs_err = np.array([...])        # Shape (N_stars, N_filters)
   parallax = np.array([...])       # Shape (N_stars,), optional
   parallax_err = np.array([...])   # Shape (N_stars,), optional

   # Define log-likelihood function for MCMC
   def lnlike(theta):
       """Log-likelihood for cluster population parameters."""
       feh, loga, av, rv, dist, field_frac = theta

       # Compute population likelihood using brutus
       lnl = isochrone_population_loglike(
           feh=feh, loga=loga, av=av, rv=rv, dist=dist,
           field_fraction=field_frac,
           stellarpop=pop,
           obs_flux=obs_flux,
           obs_err=obs_err,
           parallax=parallax,
           parallax_err=parallax_err,
           cluster_prob=0.9,  # External prior: 90% are likely members
           dim_prior=True
       )

       return lnl

   # Define log-prior function
   def lnprior(theta):
       """Log-prior for population parameters."""
       feh, loga, av, rv, dist, field_frac = theta

       # Check bounds
       if not (-2.0 < feh < 0.5):
           return -np.inf
       if not (6.0 < loga < 10.2):  # 1 Myr to 16 Gyr
           return -np.inf
       if not (0.0 < av < 5.0):
           return -np.inf
       if not (2.0 < rv < 6.0):
           return -np.inf
       if not (100.0 < dist < 10000.0):
           return -np.inf
       if not (0.0 < field_frac < 0.5):
           return -np.inf

       # Flat priors within bounds
       return 0.0

   # Full log-probability
   def lnprob(theta):
       lp = lnprior(theta)
       if not np.isfinite(lp):
           return -np.inf
       return lp + lnlike(theta)

   # Run MCMC with emcee
   import emcee

   ndim = 6
   nwalkers = 32
   nsteps = 5000

   # Initial guess: [Fe/H], log(age), A_V, R_V, dist, field_frac
   initial = np.array([0.0, 9.0, 0.3, 3.1, 2000.0, 0.1])

   # Initialize walkers with small scatter around initial guess
   pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)

   # Create sampler
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

   # Run MCMC
   print("Running MCMC...")
   sampler.run_mcmc(pos, nsteps, progress=True)

   # Extract results
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)

   # Posterior summaries
   feh_median = np.median(samples[:, 0])
   age_median = 10**np.median(samples[:, 1]) / 1e9  # Convert to Gyr
   dist_median = np.median(samples[:, 4])

   print(f"Metallicity: {feh_median:.2f} dex")
   print(f"Age: {age_median:.2f} Gyr")
   print(f"Distance: {dist_median:.1f} pc")

Advanced Topics
---------------

Differential Extinction
^^^^^^^^^^^^^^^^^^^^^^^

For clusters with significant differential extinction (cloud-to-cloud variation), you can fit individual :math:`A_V` values while keeping :math:`R_V`, age, and metallicity fixed. This requires modifying the workflow to treat :math:`A_V` as a per-star parameter rather than a population parameter.

Photometric Corrections
^^^^^^^^^^^^^^^^^^^^^^^

Empirical calibration corrections (see :doc:`photometric_offsets`) can be applied during grid generation:

.. code-block:: python

   corr_params = [dtdm, drdm, msto_smooth, feh_scale]  # From calibration

   grid = generate_isochrone_population_grid(
       stellarpop=pop,
       feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0,
       corr_params=corr_params
   )

This applies temperature/radius corrections to improve agreement between models and data.

Non-Coeval Populations
^^^^^^^^^^^^^^^^^^^^^^^

For stellar associations or star-forming regions with age spread, you can:

1. **Fit multiple isochrones**: Grid over several ages and marginalize
2. **Use age prior**: Weight different ages according to star formation history
3. **Hybrid approach**: Fit dominant age + age dispersion parameter

Computational Performance
^^^^^^^^^^^^^^^^^^^^^^^^^

Cluster fitting is computationally expensive because it requires generating a new isochrone grid for each MCMC step. Optimizations:

- **Coarse EEP grid**: Use ~100-200 EEP points instead of 2000 (faster grid generation)
- **Limited SMF grid**: Use 5-10 SMF values instead of 15 (smaller grids)
- **Cache grids**: For fixed age/metallicity, cache grids and vary only distance/extinction
- **Parallel MCMC**: Use ``emcee``'s multiprocessing to parallelize likelihood evaluations

Summary
-------

brutus cluster modeling implements the mathematically correct **mixture-before-marginalization** approach:

1. Generate (mass, SMF) grid for fixed population parameters
2. Compute cluster likelihood for each (grid_point, star) pair
3. Compute outlier likelihood for each star
4. Apply mixture model at each grid point: :math:`\mathcal{L} = w_{\rm mem} \mathcal{L}_{\rm cluster} + w_{\rm field} \mathcal{L}_{\rm outlier}`
5. Marginalize over (mass, SMF) with proper jacobians
6. Sum log-likelihoods across stars

This approach avoids biases from field contamination and properly propagates uncertainties in stellar masses and binary companions.

Next Steps
----------

- Understand photometric calibration: :doc:`photometric_offsets`
- Learn to interpret results: :doc:`understanding_results`
- Configure fitting options: :doc:`choosing_options`

References
----------

Mixture Models in Astronomy:

- Hogg et al. (2010), "Data analysis recipes: Fitting a model to data", arXiv:1008.4686
- Bovy et al. (2011), "Extreme Deconvolution: Inferring Complete Distribution Functions from Noisy, Heterogeneous and Incomplete Observations", Annals of Applied Statistics, 5, 1657

brutus Implementation:

- Speagle et al. (2025), arXiv:2503.02227
