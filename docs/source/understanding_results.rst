Understanding and Interpreting Results
========================================

This page explains how to interpret brutus output, diagnose potential issues, and assess the reliability of stellar parameter estimates.

Output Structure
----------------

brutus fitting functions return dictionaries containing posterior samples and summary statistics. The exact contents depend on the fitting mode (individual star vs cluster), but typical outputs include:

Individual Star Fitting (``BruteForce.fit()``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   results = {
       # Posterior samples
       'dist_samples': array,      # Distance samples (pc), shape (n_samples,)
       'av_samples': array,        # Extinction samples (mag), shape (n_samples,)
       'rv_samples': array,        # R_V samples, shape (n_samples,)
       'mass_samples': array,      # Initial mass samples (Msun), shape (n_samples,)
       'age_samples': array,       # Age samples (yr), shape (n_samples,)
       'feh_samples': array,       # Metallicity samples (dex), shape (n_samples,)
       'teff_samples': array,      # Effective temperature samples (K), shape (n_samples,)
       'logg_samples': array,      # Surface gravity samples (cgs), shape (n_samples,)
       'lum_samples': array,       # Luminosity samples (Lsun), shape (n_samples,)
       'radius_samples': array,    # Radius samples (Rsun), shape (n_samples,)

       # Summary statistics
       'dist_median': float,       # Median distance
       'dist_std': float,          # Standard deviation
       'dist_16': float,           # 16th percentile (lower 1-sigma)
       'dist_84': float,           # 84th percentile (upper 1-sigma)
       # Similar stats for av, mass, age, etc.

       # Best-fit information
       'best_fit_idx': int,        # Grid index of maximum posterior
       'best_fit_mags': array,     # Model magnitudes at best fit
       'chi2_best': float,         # Chi-square at best fit
       'lnL_max': float,           # Maximum log-likelihood

       # Diagnostic information
       'n_grid_points': int,       # Number of grid points evaluated
       'converged': bool,          # Whether optimization converged
   }

Cluster Fitting (``isochrone_population_loglike()``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When used with MCMC (e.g., ``emcee``), the sampler contains:

.. code-block:: python

   # Extract chain from emcee sampler
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)

   # samples shape: (n_samples, n_params)
   # where n_params = [feh, loga, av, rv, dist, field_frac, ...]

   feh_samples = samples[:, 0]
   loga_samples = samples[:, 1]  # log10(age in years)
   av_samples = samples[:, 2]
   # etc.

Posterior Distributions
-----------------------

The fundamental output from brutus is **posterior samples** representing the probability distribution over parameters given the data.

Visualizing Posteriors
^^^^^^^^^^^^^^^^^^^^^^^

**Histograms** show 1-D marginal distributions:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Plot distance posterior
   plt.figure(figsize=(8, 5))
   plt.hist(results['dist_samples'], bins=50, density=True, alpha=0.7)
   plt.axvline(results['dist_median'], color='r', linestyle='--',
               label=f"Median: {results['dist_median']:.1f} pc")
   plt.axvline(results['dist_16'], color='orange', linestyle=':',
               label=f"16-84%: [{results['dist_16']:.1f}, {results['dist_84']:.1f}]")
   plt.axvline(results['dist_84'], color='orange', linestyle=':')
   plt.xlabel('Distance (pc)')
   plt.ylabel('Probability Density')
   plt.legend()
   plt.title('Distance Posterior Distribution')
   plt.show()

**Corner plots** show joint distributions and correlations:

.. code-block:: python

   import corner

   # Select parameters for corner plot
   params_to_plot = np.column_stack([
       results['dist_samples'],
       results['av_samples'],
       results['mass_samples'],
       results['age_samples'] / 1e9,  # Convert to Gyr
       results['feh_samples']
   ])

   labels = ['Distance (pc)', r'$A_V$ (mag)', r'Mass ($M_\odot$)',
             'Age (Gyr)', '[Fe/H] (dex)']

   fig = corner.corner(
       params_to_plot,
       labels=labels,
       quantiles=[0.16, 0.5, 0.84],
       show_titles=True,
       title_fmt='.2f'
   )
   plt.show()

Interpreting Posterior Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Gaussian-like (symmetric)**:
   Well-constrained parameter with good data. Example: Distance for bright star with accurate parallax.

   .. code-block:: none

      |     ****
      |    *    *
      |  **      **
      |**          **
      +---------------
        d_median

**Skewed (asymmetric)**:
   Parameter hitting a boundary or degeneracy. Example: Extinction near A_V = 0.

   .. code-block:: none

      |**
      | ****
      |    ***
      |      ****
      +------------
       0    A_V

**Bimodal (multiple peaks)**:
   Degeneracy between solutions. Example: Faint red star could be nearby M dwarf or distant K giant.

   .. code-block:: none

      |  **      **
      | *  *    *  *
      |**  **  **  **
      +--------------
       d1    d2

   **Action**: Check if parallax helps resolve degeneracy. If not, data may be insufficient.

**Flat/uniform**:
   Parameter unconstrained by data. Example: R_V when extinction is negligible.

   .. code-block:: none

      |************
      |************
      |************
      +-----------
         R_V

   **Action**: This is expected when parameter doesn't affect the data. Not a problem.

Common Degeneracies
-------------------

Distance-Extinction Degeneracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: A faint reddened star can look similar to a nearby intrinsically red star.

**Symptoms**:
   - Strong correlation between ``dist_samples`` and ``av_samples`` in corner plot
   - Elongated posterior contours along distance-extinction diagonal

**Solutions**:
   - **Parallax**: Breaks degeneracy by independently constraining distance
   - **Multi-band photometry**: Different wavelength dependence helps separate intrinsic color from reddening
   - **Dust priors**: 3-D dust maps constrain expected extinction at different distances

**Example diagnostic**:

.. code-block:: python

   # Check correlation
   import numpy as np
   correlation = np.corrcoef(results['dist_samples'], results['av_samples'])[0, 1]
   print(f"Distance-extinction correlation: {correlation:.2f}")

   # Strong positive correlation (r > 0.7) indicates degeneracy
   if abs(correlation) > 0.7:
       print("WARNING: Strong distance-extinction degeneracy detected")
       print("Consider adding parallax or improving photometric coverage")

Mass-Age-Metallicity Degeneracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Different combinations of (mass, age, metallicity) can produce similar temperatures and luminosities.

**Symptoms**:
   - Broad, multi-modal posteriors for ``mass_samples``, ``age_samples``, ``feh_samples``
   - Multiple solutions in parameter space

**Solutions**:
   - **Asteroseismology**: Directly constrains surface gravity and mass
   - **Spectroscopy**: Breaks metallicity degeneracy
   - **Galactic priors**: Age and metallicity priors from Galactic position help
   - **Multi-epoch data**: Variability or astrometry can constrain mass

**Example**: An old, metal-poor, massive star can look like a young, metal-rich, low-mass star.

Binary Degeneracy
^^^^^^^^^^^^^^^^^

**Problem**: Unresolved binary companions add flux, mimicking a brighter single star.

**Symptoms**:
   - Inferred parameters inconsistent with spectroscopic measurements
   - Residuals suggesting excess flux in some bands

**Solutions**:
   - **Check for binarity**: Radial velocity variations, eclipses, imaging
   - **Model binaries explicitly**: Use SMF parameter in cluster fitting
   - **Be cautious**: Binary contamination affects ~50% of stars

Diagnostic Checks
-----------------

χ² and Goodness-of-Fit
^^^^^^^^^^^^^^^^^^^^^^^

Check the quality of the best-fit model:

.. code-block:: python

   # Compute reduced chi-square
   n_data = len(phot)  # Number of photometric bands
   n_params = 6  # Distance, extinction, mass, age, metallicity, alpha
   dof = n_data - n_params

   chi2_reduced = results['chi2_best'] / dof

   print(f"Reduced χ²: {chi2_reduced:.2f}")

   if chi2_reduced < 0.5:
       print("WARNING: χ² too low - may indicate overestimated errors")
   elif chi2_reduced > 3.0:
       print("WARNING: χ² too high - poor fit or underestimated errors")
   else:
       print("Good fit quality")

Residual Analysis
^^^^^^^^^^^^^^^^^

Examine residuals between data and best-fit model:

.. code-block:: python

   # Observed vs model magnitudes
   obs_mags = -2.5 * np.log10(phot)
   model_mags = results['best_fit_mags']
   residuals = obs_mags - model_mags

   # Plot residuals
   bands = ['g', 'r', 'i', 'z', 'y']
   plt.figure()
   plt.errorbar(range(len(bands)), residuals, yerr=phot_err/phot*1.0857,
                fmt='o', capsize=5)
   plt.axhline(0, color='k', linestyle='--')
   plt.xticks(range(len(bands)), bands)
   plt.ylabel('Residual (mag)')
   plt.xlabel('Band')
   plt.title('Photometric Residuals')
   plt.show()

   # Check for systematic trends
   if np.all(np.abs(residuals) < 0.1):
       print("Residuals look good (< 0.1 mag)")
   else:
       problematic = np.where(np.abs(residuals) > 0.1)[0]
       print(f"Large residuals in bands: {[bands[i] for i in problematic]}")

Parallax Consistency
^^^^^^^^^^^^^^^^^^^^

If parallax was used, check consistency between photometric and parallax-based distance:

.. code-block:: python

   # Parallax-implied distance
   parallax_dist = 1000.0 / parallax  # pc (for parallax in mas)
   parallax_dist_err = 1000.0 * parallax_err / parallax**2

   # Brutus distance estimate
   brutus_dist = results['dist_median']
   brutus_dist_err = (results['dist_84'] - results['dist_16']) / 2.0

   # Consistency check (within 2-sigma?)
   diff = abs(brutus_dist - parallax_dist)
   combined_err = np.sqrt(brutus_dist_err**2 + parallax_dist_err**2)

   print(f"Parallax distance: {parallax_dist:.1f} ± {parallax_dist_err:.1f} pc")
   print(f"Brutus distance: {brutus_dist:.1f} ± {brutus_dist_err:.1f} pc")
   print(f"Difference: {diff:.1f} pc ({diff/combined_err:.1f} sigma)")

   if diff / combined_err > 2.0:
       print("WARNING: Parallax and photometric distances inconsistent")
       print("Possible issues: bad parallax, photometric errors, binarity")

Prior Sensitivity
^^^^^^^^^^^^^^^^^

Test how results change with/without priors:

.. code-block:: python

   from brutus.analysis import BruteForce

   # Fit with full priors
   fitter_with_priors = BruteForce(grid,
                                   use_galactic_prior=True,
                                   use_dust_prior=True)
   results_with = fitter_with_priors.fit(phot, phot_err,
                                          parallax=plx, parallax_err=plx_err)

   # Fit without priors
   fitter_no_priors = BruteForce(grid,
                                 use_galactic_prior=False,
                                 use_dust_prior=False)
   results_without = fitter_no_priors.fit(phot, phot_err,
                                          parallax=plx, parallax_err=plx_err)

   # Compare
   print("With priors:", results_with['dist_median'], "±",
         results_with['dist_std'], "pc")
   print("Without priors:", results_without['dist_median'], "±",
         results_without['dist_std'], "pc")

   # If results change significantly, data is prior-dominated
   fractional_change = abs(results_with['dist_median'] - results_without['dist_median']) / results_with['dist_median']
   if fractional_change > 0.3:
       print("WARNING: Results strongly prior-dependent (>30% change)")
       print("Data may be insufficient to constrain parameters")

Reliability Indicators
----------------------

When to Trust Results
^^^^^^^^^^^^^^^^^^^^^

✅ **High confidence**:
   - χ²_reduced ~ 1
   - Narrow, Gaussian-like posteriors
   - Parallax and photometric distances agree (if parallax available)
   - Residuals < 0.1 mag across all bands
   - Results stable with/without priors

✅ **Moderate confidence**:
   - χ²_reduced between 0.5 and 2
   - Asymmetric but unimodal posteriors
   - Some degeneracies but broken by parallax or priors
   - Residuals < 0.2 mag

⚠ **Low confidence**:
   - χ²_reduced > 3 or < 0.3
   - Bimodal or very broad posteriors
   - Strong parameter correlations unbroken by data
   - Large residuals (> 0.3 mag) in multiple bands
   - Results change dramatically without priors

❌ **Unreliable**:
   - Optimization failed to converge
   - Posteriors hit parameter boundaries
   - Parallax and photometric distances disagree by > 3σ
   - Systematic residual patterns (e.g., all blue bands too bright)

Uncertainty Quantification
---------------------------

Credible Intervals
^^^^^^^^^^^^^^^^^^

brutus provides Bayesian **credible intervals** (not frequentist confidence intervals):

.. code-block:: python

   # 68% credible interval (analogous to 1-sigma)
   dist_lower = results['dist_16']
   dist_upper = results['dist_84']
   print(f"Distance: {results['dist_median']:.1f} (+{dist_upper - results['dist_median']:.1f} / -{results['dist_median'] - dist_lower:.1f}) pc")

   # 95% credible interval
   dist_025 = np.percentile(results['dist_samples'], 2.5)
   dist_975 = np.percentile(results['dist_samples'], 97.5)
   print(f"95% interval: [{dist_025:.1f}, {dist_975:.1f}] pc")

**Interpretation**: "There is a 68% probability that the true distance lies in [dist_16, dist_84] given the data and priors."

Systematic vs Statistical Uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

brutus uncertainties are primarily **statistical** (measurement errors + parameter degeneracies). They do **not** include:

- Model systematics (errors in stellar evolution models)
- Photometric zero-point uncertainties
- Extinction curve uncertainties
- Prior misspecification

**Recommended practice**: Add systematic error floor (~10% for distances, ~0.05 mag for extinction) in quadrature:

.. code-block:: python

   # Statistical uncertainty from brutus
   dist_err_stat = (results['dist_84'] - results['dist_16']) / 2.0

   # Add 10% systematic floor
   dist_err_sys = 0.10 * results['dist_median']

   # Total uncertainty
   dist_err_total = np.sqrt(dist_err_stat**2 + dist_err_sys**2)

   print(f"Distance: {results['dist_median']:.1f} ± {dist_err_total:.1f} pc")

Derived Quantities
------------------

brutus provides direct samples of physical parameters, enabling straightforward propagation of uncertainties to derived quantities.

Absolute Magnitude
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Absolute magnitude from distance and apparent magnitude
   app_mag_g = obs_mags[0]  # g-band
   dist_modulus_samples = 5.0 * np.log10(results['dist_samples']) - 5.0
   abs_mag_g_samples = app_mag_g - dist_modulus_samples - results['av_samples'] * results['R_g']

   abs_mag_g_median = np.median(abs_mag_g_samples)
   abs_mag_g_err = np.std(abs_mag_g_samples)

   print(f"M_g = {abs_mag_g_median:.2f} ± {abs_mag_g_err:.2f} mag")

Luminosity
^^^^^^^^^^

.. code-block:: python

   # Already provided in results['lum_samples'], but can also compute:
   M_bol_sun = 4.74  # Solar bolometric magnitude
   M_bol_samples = results['M_bol_samples']  # From brutus
   lum_samples = 10**((M_bol_sun - M_bol_samples) / 2.5)  # In solar units

   lum_median = np.median(lum_samples)
   lum_16, lum_84 = np.percentile(lum_samples, [16, 84])

   print(f"Luminosity: {lum_median:.3f} (+{lum_84-lum_median:.3f} / -{lum_median-lum_16:.3f}) L_sun")

Galactic Coordinates
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from astropy.coordinates import SkyCoord
   import astropy.units as u

   # Sky position
   coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

   # Add distance samples
   coords_3d = SkyCoord(
       ra=ra*u.deg,
       dec=dec*u.deg,
       distance=results['dist_samples']*u.pc,
       frame='icrs'
   )

   # Transform to Galactic coordinates
   coords_gal = coords_3d.galactocentric

   X_samples = coords_gal.x.to(u.kpc).value
   Y_samples = coords_gal.y.to(u.kpc).value
   Z_samples = coords_gal.z.to(u.kpc).value

   print(f"Galactic X: {np.median(X_samples):.2f} ± {np.std(X_samples):.2f} kpc")
   print(f"Galactic Z: {np.median(Z_samples):.2f} ± {np.std(Z_samples):.2f} kpc")

Troubleshooting Common Issues
------------------------------

"Optimization did not converge"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Gradient-based optimizer failed to find minimum in flux space.

**Solutions**:
   - Check for bad photometry (negative fluxes, very large errors)
   - Increase distance or extinction bounds
   - Try different starting values
   - If persistent, data may be incompatible with models

"Distance hits lower bound"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Best-fit distance is at minimum allowed value (typically 10 pc).

**Solutions**:
   - Check parallax—is star truly very nearby?
   - Inspect residuals—may indicate very bright intrinsic source
   - Consider exotic objects (white dwarfs, brown dwarfs) outside model grid

"Extinction posterior is flat"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Data insufficient to constrain reddening.

**This is often OK**: For bright, blue stars with minimal extinction, A_V is genuinely unconstrained. Not a problem unless you need precise reddening.

**Solutions if you need A_V**:
   - Add redder photometric bands (near-IR helps constrain reddening)
   - Use dust map priors more aggressively
   - Check for instrumental systematics

"All parameters have huge uncertainties"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Data quality poor or star outside model coverage.

**Solutions**:
   - Check photometric S/N—are errors realistic?
   - Verify star is within grid boundaries (mass, metallicity, age)
   - Add parallax if available
   - Consider if object is non-stellar (galaxy, QSO) or exotic (WD, CV)

Summary
-------

Interpreting brutus results requires:

✓ Visualizing posterior distributions (histograms, corner plots)
✓ Checking goodness-of-fit (χ², residuals)
✓ Assessing degeneracies (correlations between parameters)
✓ Testing prior sensitivity
✓ Validating against independent measurements (parallax, spectroscopy)

Results are most reliable when posteriors are unimodal, χ²~1, and independent checks agree.

Next Steps
----------

- Configure fitting options: :doc:`choosing_options`
- Review common questions: :doc:`faq`
- Learn about priors: :doc:`priors`

References
----------

Bayesian Inference and Uncertainty Quantification:

- Hogg & Foreman-Mackey (2018), "Data Analysis Recipes: Using Markov Chain Monte Carlo", ApJS, 236, 11
- Gelman et al. (2013), "Bayesian Data Analysis" (3rd ed.), CRC Press

Stellar Parameter Degeneracies:

- Jørgensen & Lindegren (2005), "Systemic Biases in Star Formation History Studies", A&A, 436, 127
- Bovy (2016), "The Stellar Spectroscopic Surveys In The Gaia Era", in Astrophysical Applications of Gravitational Lensing, IAU Symposium 319

brutus Implementation:

- Speagle et al. (2025), arXiv:2503.02227
