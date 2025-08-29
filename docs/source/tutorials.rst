Tutorials
=========

The ``tutorials/`` directory contains Jupyter notebooks demonstrating key workflows and advanced usage patterns.

Available Notebooks
--------------------

**Overview 0 - Downloading Files**
   Learn how to download and manage data files required by brutus.

**Overview 1 - Models and Priors**
   Understand stellar evolution models, priors, and their role in Bayesian inference.

**Overview 2 - Generating Model Grids**
   Create custom stellar model grids for specific applications.

**Overview 3 - Fitting Individual Sources**
   Comprehensive guide to individual star fitting with real data examples.

**Overview 4 - Extinction Modeling**
   Work with 3D dust maps and model interstellar extinction.

**Overview 5 - Cluster Modeling**
   Model stellar clusters with consistent ages, metallicities, and distances.

**Overview 6 - Photometric Offsets**
   Compute and apply photometric offsets between different survey systems.

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