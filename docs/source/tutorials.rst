Tutorials
=========

The ``tutorials/`` directory contains comprehensive Jupyter notebooks that walk you through brutus's key capabilities. The tutorials progress from basic setup to advanced analysis workflows, with real data examples and complete working code.

**Learning Path:**

Start with Tutorial 0 to set up your data, then work through Tutorials 1-3 for core functionality. Tutorials 4-6 cover advanced topics that may be optional depending on your science goals.

Tutorial Descriptions
---------------------

**Overview 0 - Downloading Files** (⏱ ~15 minutes)

*Setting up brutus data dependencies*

This tutorial covers:

- Using ``fetch_grids()``, ``fetch_isos()``, and ``fetch_dustmaps()`` to download data
- Understanding brutus's data organization and caching
- Checking installation and data integrity
- Configuring custom data locations

**Prerequisites**: Fresh brutus installation

**Overview 1 - Models and Priors** (⏱ ~30 minutes)

*Understanding the stellar evolution models and prior distributions*

This tutorial explores:

- MIST stellar evolution models and the EEP parameterization
- Generating stellar parameters with ``EEPTracks`` and ``Isochrone``
- Exploring Galactic structure priors (thin/thick disk, halo)
- Visualizing IMF, metallicity distributions, and dust map priors
- How priors break distance-extinction degeneracies

**Prerequisites**: Tutorial 0 (data files downloaded)

**What you'll learn**: The scientific foundation underlying brutus fitting

**Overview 2 - Generating Model Grids** (⏱ ~45 minutes)

*Creating custom pre-computed model grids for fast fitting*

This tutorial demonstrates:

- Using ``GridGenerator`` to create custom grids with specific filter combinations
- Understanding grid structure (mass, EEP, metallicity, reddening coefficients)
- Choosing appropriate grid resolution for your science case
- Loading and inspecting grid contents
- Grid file formats and storage considerations

**Prerequisites**: Tutorial 1

**What you'll learn**: How to create optimized grids for your photometric data

**Overview 3 - Fitting Individual Sources** (⏱ ~60 minutes)

*Complete workflow for individual star parameter estimation*

This tutorial provides:

- End-to-end fitting with ``BruteForce`` using real Gaia + 2MASS data
- Incorporating parallax for improved distance constraints
- Interpreting posterior distributions (distances, extinctions, stellar parameters)
- Diagnostic checks (χ², residuals, parameter correlations)
- Comparing results with/without priors to assess prior sensitivity
- Handling edge cases (very nearby stars, highly reddened stars, evolved stars)

**Prerequisites**: Tutorial 2 (or use pre-made grids)

**What you'll learn**: The complete individual star fitting pipeline

**Overview 4 - Extinction Modeling** (⏱ ~40 minutes)

*Working with 3-D dust maps and line-of-sight extinction*

This tutorial covers:

- Loading and querying Bayestar 3-D dust maps
- Understanding distance-dependent extinction priors
- Modeling extinction variation along sight lines
- Using extinction maps to improve stellar parameter estimates
- Limitations of dust maps (Galactic plane, high extinctions, distant stars)

**Prerequisites**: Tutorial 3

**What you'll learn**: How dust maps constrain extinction and distance

**Overview 5 - Cluster Modeling** (⏱ ~75 minutes)

*Fitting coeval stellar populations with mixture models*

This tutorial demonstrates:

- Setting up isochrone population models with ``StellarPop``
- The mixture-before-marginalization approach for field contamination
- Using ``isochrone_population_loglike()`` with MCMC (emcee)
- Handling binary stars with secondary mass fraction (SMF)
- Choosing outlier models (chi-square vs uniform)
- Fitting real open cluster data with parallaxes
- Diagnosing convergence and assessing results

**Prerequisites**: Tutorial 3, familiarity with MCMC

**What you'll learn**: Advanced population synthesis and cluster fitting

**Overview 6 - Photometric Offsets** (⏱ ~50 minutes)

*Empirical calibration of systematic photometric errors*

This tutorial explains:

- Why empirical calibration is necessary (model systematic errors)
- Deriving isochrone corrections from open clusters
- Computing photometric offsets from field stars
- Applying corrections during grid generation
- Validating corrections with eclipsing binaries
- When corrections improve vs degrade results

**Prerequisites**: Tutorial 5

**What you'll learn**: Improving accuracy through empirical calibration

Running the Tutorials
----------------------

To run the tutorials:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/joshspeagle/brutus.git
      cd brutus

2. Install brutus in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

3. Start Jupyter and navigate to the ``tutorials/`` directory:

   .. code-block:: bash

      jupyter notebook tutorials/

Tutorial Data
-------------

The tutorial notebooks include sample data files and will download additional data as needed. Make sure you have sufficient disk space (several GB) for the stellar evolution models and dust maps.

Advanced Topics
---------------

The tutorials cover both basic usage and advanced topics including:

- Custom prior specification
- Multi-object fitting workflows
- Performance optimization
- Integration with other astronomical tools
- Custom visualization and analysis

For the most up-to-date tutorials and examples, see the `tutorials/ directory on GitHub <https://github.com/joshspeagle/brutus/tree/master/tutorials>`_.
