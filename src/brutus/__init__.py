#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus: Brute-force Bayesian inference for stellar photometry

A Pure Python package for deriving distances, reddenings, and stellar
properties from photometry using "brute force" Bayesian inference.

The package is designed to be highly modular, with modules for:
- Individual star modeling and fitting
- Star cluster analysis
- 3D dust mapping
- Stellar evolution model management

Usage
-----
For individual star modeling::

    from brutus.core import EEPTracks, StarEvolTrack
    tracks = EEPTracks()
    star = StarEvolTrack(tracks=tracks)
    sed, params, params2 = star.get_seds(mini=1.0, eep=350, feh=0.0)

For stellar population modeling::

    from brutus.core import Isochrone, StellarPop
    iso = Isochrone()
    pop = StellarPop(isochrone=iso)
    seds, params, params2 = pop.synthesize(feh=0.0, afe=0.0, loga=9.0)

For data management::

    from brutus import fetch_grids, load_models
    fetch_grids(target_dir='./data/')
    models = load_models('./data/grid_mist_v9.h5')
"""

from __future__ import division, print_function

# Version management
__version__ = "0.9.0"

# Core functionality imports
try:
    # Core stellar evolution models (refactored)
    from .core import Isochrone, EEPTracks, StarGrid

    # Data management (refactored)
    from .data import fetch_grids, fetch_isos, fetch_dustmaps, load_models

    # Essential utilities (refactored)
    from .utils import magnitude, inv_magnitude

    # Analysis and fitting
    from .analysis import BruteForce

    # Dust mapping (not yet refactored)
    # from .dust import Bayestar

    # Make key classes easily accessible
    __all__ = [
        # Version
        "__version__",
        # Core classes
        "Isochrone",
        "EEPTracks",
        "StarGrid",
        # Analysis classes
        "BruteForce",
        # Data utilities (refactored)
        "fetch_grids",
        "fetch_isos",
        "fetch_dustmaps",
        "load_models",
        # Photometry utilities (refactored)
        "magnitude",
        "inv_magnitude",
        # Analysis classes (not yet refactored)
        # 'BruteForce', 'isochrone_loglike',
        # Dust mapping (not yet refactored)
        # 'Bayestar',
    ]

except ImportError as e:
    # During the transition period, some imports might fail
    # Provide graceful fallback
    import warnings

    warnings.warn(
        f"Some brutus modules are not yet available during reorganization: {e}. "
        "Please use the original module imports temporarily.",
        ImportWarning,
    )

    # Minimal fallback
    __all__ = ["__version__"]


# Convenience functions for common workflows
# These will be implemented in later phases
def quick_star_fit(*args, **kwargs):
    """
    Convenience function for quick individual star fitting.

    This function will provide a simplified interface for common
    individual star fitting workflows.

    Notes
    -----
    This functionality is planned for Phase 2 of the refactoring.
    Currently, please use the core classes directly:

    >>> from brutus.core import EEPTracks, StarEvolTrack
    >>> tracks = EEPTracks()
    >>> star = StarEvolTrack(tracks=tracks)

    Raises
    ------
    NotImplementedError
        Function not yet implemented in current refactoring phase.
    """
    raise NotImplementedError(
        "Convenience functions will be implemented in Phase 2. "
        "Please use EEPTracks and StarEvolTrack classes directly for now."
    )


def quick_cluster_fit(*args, **kwargs):
    """
    Convenience function for quick cluster fitting.

    This function will provide a simplified interface for common
    stellar cluster fitting workflows.

    Notes
    -----
    This functionality is planned for Phase 2 of the refactoring.
    Currently, please use the core classes directly:

    >>> from brutus.core import Isochrone, StellarPop
    >>> iso = Isochrone()
    >>> pop = StellarPop(isochrone=iso)

    Raises
    ------
    NotImplementedError
        Function not yet implemented in current refactoring phase.
    """
    raise NotImplementedError(
        "Convenience functions will be implemented in Phase 2. "
        "Please use Isochrone and StellarPop classes directly for now."
    )
