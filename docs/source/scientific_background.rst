Scientific Background
=====================

This page provides an overview of the scientific foundations and statistical framework underlying **brutus**. For detailed mathematical derivations and methodological choices, see `Speagle et al. (2025) <https://arxiv.org/abs/2503.02227>`_.

What brutus Does
----------------

**brutus** performs **Bayesian inference** to derive stellar properties, distances, and interstellar reddenings from photometric and astrometric observations. Given multi-band photometry and optionally parallax measurements for a star, brutus estimates:

- **Distance** and **extinction** (reddening)
- **Stellar parameters**: mass, age, metallicity, temperature, luminosity, radius, surface gravity
- **Full posterior distributions** with properly propagated uncertainties

The package is designed for three main use cases:

1. **Individual stars**: Fit single stellar sources with measured photometry and parallax
2. **Stellar clusters**: Model coeval populations with shared age, metallicity, and distance
3. **3-D dust mapping**: Map extinction along lines of sight using ensemble stellar data

The Statistical Framework
--------------------------

brutus uses a forward modeling approach combined with Bayesian inference. The framework has three key components: a physical model connecting intrinsic stellar properties to observations, a statistical likelihood relating data to model predictions, and prior probability distributions encoding astrophysical knowledge.

The Noiseless Model
^^^^^^^^^^^^^^^^^^^

The fundamental relationship between observed magnitudes and underlying stellar properties is:

.. math::

   \mathbf{m}_{\theta,\phi} = \mathbf{M}_\theta + \mu + A_V \times (\mathbf{R}_\theta + R_V \times \mathbf{R}'_\theta)

where:

- :math:`\mathbf{m}_{\theta,\phi}` are the predicted observed magnitudes across multiple photometric bands
- :math:`\mathbf{M}_\theta` are the intrinsic absolute magnitudes (derived from stellar evolution models)
- :math:`\mu = 5 \log_{10}(d/10\,{\rm pc})` is the distance modulus
- :math:`A_V` is the visual extinction (dust reddening)
- :math:`\mathbf{R}_\theta` and :math:`\mathbf{R}'_\theta` are reddening vectors describing wavelength-dependent extinction
- :math:`R_V \equiv A_V / E(B-V)` parameterizes variations in the dust extinction curve

**Intrinsic parameters** :math:`\theta` describe the star itself:

- :math:`M_{\rm init}`: Initial mass (solar masses)
- :math:`[{\rm Fe/H}]_{\rm init}`: Initial metallicity (dex)
- :math:`t_{\rm age}`: Current age (years)

**Extrinsic parameters** :math:`\phi` describe environmental effects:

- :math:`d`: Distance (pc)
- :math:`A_V`: Visual extinction (magnitudes)
- :math:`R_V`: Extinction curve shape parameter

.. note::
   This model is a simplification that assumes single stars in isolation. Binary stars, stellar rotation, and detailed chemical abundance patterns are not explicitly modeled, though binary companions can be included with the secondary mass fraction (SMF) parameter.

The Noisy Data
^^^^^^^^^^^^^^

Real observations include measurement uncertainties. brutus models two types of data:

**Photometry**: Flux densities are assumed normally distributed around their true values:

.. math::

   \hat{\mathbf{F}} \sim \mathcal{N}[\mathbf{F}, \mathbf{C}_F]

where :math:`\hat{\mathbf{F}}` are the observed flux densities, :math:`\mathbf{F}` are the true fluxes, and :math:`\mathbf{C}_F` is a diagonal covariance matrix of flux uncertainties.

While the stellar models predict magnitudes, brutus converts to flux space for the likelihood calculation. This is because flux uncertainties are approximately Gaussian, while magnitude uncertainties are not. The conversion is:

.. math::

   F = 10^{-0.4 m}

**Astrometry**: Parallax measurements are modeled as normally distributed:

.. math::

   \hat{\varpi} \sim \mathcal{N}[\varpi, \sigma_\varpi]

where :math:`\hat{\varpi}` is the observed parallax (milliarcseconds), :math:`\varpi = 1000/d` is the true parallax, and :math:`\sigma_\varpi` is the parallax uncertainty.

The parallax provides a direct constraint on distance that helps break degeneracies between distance and extinction.

The Posterior Probability
^^^^^^^^^^^^^^^^^^^^^^^^^^

Bayes' theorem combines the data likelihood with prior information to yield the posterior probability distribution:

.. math::

   P(\theta, \phi \,|\, \hat{\mathbf{F}}, \hat{\varpi}) \propto \mathcal{L}_{\rm phot}(\theta, \phi) \times \mathcal{L}_{\rm astr}(\phi) \times \pi(\theta, \phi)

where:

- :math:`\mathcal{L}_{\rm phot}` is the photometric likelihood
- :math:`\mathcal{L}_{\rm astr}` is the astrometric (parallax) likelihood
- :math:`\pi(\theta, \phi)` encodes prior knowledge from Galactic models

The posterior represents our updated knowledge about the stellar parameters and distance/extinction given the observed data and our astrophysical priors.

Prior Probability Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

brutus uses informative priors based on Galactic structure and stellar population models. These priors factorize into independent components:

.. math::

   \pi(\theta, \phi) \propto \pi(M_{\rm init}) \times \pi(d\,|\,\ell,b) \times \pi([{\rm Fe/H}]_{\rm init}\,|\,d,\ell,b) \times \pi(t_{\rm age}\,|\,d,\ell,b) \times \pi(A_V\,|\,d,\ell,b) \times \pi(R_V)

where :math:`\ell,b` are Galactic longitude and latitude. The priors include:

1. **Initial Mass Function (IMF)**: :math:`\pi(M_{\rm init})` describes the stellar mass distribution at formation (Kroupa IMF)

2. **3-D Stellar Number Density**: :math:`\pi(d\,|\,\ell,b)` models the spatial distribution of stars (thin disk, thick disk, halo components)

3. **3-D Metallicity Distribution**: :math:`\pi([{\rm Fe/H}]_{\rm init}\,|\,d,\ell,b)` captures metallicity gradients in the Galaxy

4. **3-D Age Distribution**: :math:`\pi(t_{\rm age}\,|\,d,\ell,b)` represents the age structure of different Galactic populations

5. **3-D Dust Extinction**: :math:`\pi(A_V\,|\,d,\ell,b)` uses 3-D dust maps (e.g., Bayestar) to constrain extinction along sight lines

6. **Dust Curve Variation**: :math:`\pi(R_V)` models variations in extinction curve shape (mean :math:`R_V \approx 3.3`, scatter :math:`\sigma_{R_V} \approx 0.2`)

.. seealso::
   See :doc:`priors` for detailed descriptions of each prior component and guidance on customization.

Why Bayesian Inference?
------------------------

The Bayesian framework offers several advantages for stellar parameter estimation:

**Breaking Degeneracies**
   Photometry alone cannot uniquely determine stellar properties—a faint red star could be either a nearby cool dwarf or a distant reddened giant. Priors on Galactic structure help resolve these ambiguities by incorporating knowledge about where different types of stars are likely to be found.

**Proper Uncertainty Quantification**
   The posterior distribution provides full probabilistic uncertainties that account for both measurement errors and model limitations. This is critical for propagating uncertainties into derived quantities.

**Incorporating Diverse Information**
   Bayesian inference naturally combines multiple data types (photometry, parallax, proper motions) and prior knowledge (Galactic models, dust maps, stellar physics) into a coherent probabilistic framework.

**Handling Missing Data**
   The framework gracefully handles missing photometric bands or parallax measurements by marginalizing over unobserved quantities.

Common Use Cases
----------------

**Individual Field Stars**
   For isolated stars with good photometry and parallax (e.g., from Gaia), brutus provides robust distance and extinction estimates even in moderately dusty regions. The Galactic priors help constrain the stellar population (thin disk vs thick disk vs halo) which informs the age and metallicity.

**Stellar Clusters**
   Coeval populations offer powerful constraints because all stars share the same age, metallicity, and distance. brutus uses a mixture model to handle field contamination while fitting isochrones to cluster members. See :doc:`cluster_modeling` for details.

**3-D Dust Mapping**
   By inverting the stellar parameter estimation problem, brutus can map the 3-D distribution of dust extinction using ensemble data from many stars along a sight line. This provides independent validation and refinement of existing dust maps.

Limitations and Caveats
-----------------------

**Model Simplifications**
   The stellar models assume non-rotating single stars with solar-scaled abundance patterns (except for α-enhancement). Real stars may have rotation, exotic abundances, or binary companions that violate these assumptions.

**Systematic Uncertainties**
   Theoretical stellar models have systematic errors, particularly for low-mass and pre-main-sequence stars. brutus includes empirical calibration corrections (see :doc:`photometric_offsets`) but residual systematics remain.

**Prior Dependence**
   Results can be sensitive to prior choices, especially for faint or poorly measured stars. It's important to check that priors are appropriate for your science case and consider how prior assumptions affect conclusions.

**Computational Cost**
   The brute-force grid approach is computationally intensive for large samples. Pre-computed model grids and parallel processing help, but fitting millions of stars requires substantial resources.

Next Steps
----------

- Learn about the stellar evolution models: :doc:`stellar_models`
- Understand the grid-based fitting approach: :doc:`grid_generation`
- Explore the prior specifications: :doc:`priors`
- See cluster modeling methodology: :doc:`cluster_modeling`

References
----------

For full mathematical details and method validation, see:

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", arXiv:2503.02227
