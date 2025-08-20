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
For individual star fitting::

    from brutus import BruteForce
    fitter = BruteForce(models, labels, masks)
    results = fitter.fit(photometry, errors)

For isochrone generation::

    from brutus import Isochrone
    iso = Isochrone(filters=['g', 'r', 'i'])
    sed, params = iso.get_isochrone(age=1e9, metallicity=0.0)

For data management::

    from brutus import fetch_grids, load_models
    fetch_grids(target_dir='./data/')
    models = load_models('./data/grid_mist_v9.h5')
"""

from __future__ import (division, print_function)

# Version management
__version__ = "0.9.0"

# Core functionality - import from existing modules during transition
# These imports maintain backward compatibility while we reorganize
try:
    # Core stellar evolution models
    from .core import Isochrone, MISTtracks, SEDmaker
    
    # Analysis and fitting
    from .analysis import BruteForce, isochrone_loglike
    
    # Dust mapping
    from .dust import Bayestar
    
    # Data management  
    from .data import fetch_grids, fetch_isos, load_models
    
    # Essential utilities
    from .utils import magnitude, inv_magnitude
    
    # Make key classes easily accessible
    __all__ = [
        # Version
        '__version__',
        
        # Core classes
        'Isochrone', 'MISTtracks', 'SEDmaker',
        
        # Analysis classes  
        'BruteForce', 'isochrone_loglike',
        
        # Dust mapping
        'Bayestar',
        
        # Data utilities
        'fetch_grids', 'fetch_isos', 'load_models',
        
        # Photometry utilities
        'magnitude', 'inv_magnitude',
    ]

except ImportError as e:
    # During the transition period, some imports might fail
    # Provide graceful fallback
    import warnings
    warnings.warn(
        f"Some brutus modules are not yet available during reorganization: {e}. "
        "Please use the original module imports temporarily.",
        ImportWarning
    )
    
    # Minimal fallback
    __all__ = ['__version__']

# Convenience functions for common workflows
# These will be implemented in later phases
def quick_star_fit(*args, **kwargs):
    """
    Convenience function for quick individual star fitting.
    
    Note: This will be implemented in Phase 2.
    """
    raise NotImplementedError(
        "Convenience functions will be implemented in Phase 2. "
        "Please use BruteForce class directly for now."
    )

def quick_cluster_fit(*args, **kwargs):
    """
    Convenience function for quick cluster fitting.
    
    Note: This will be implemented in Phase 2.
    """
    raise NotImplementedError(
        "Convenience functions will be implemented in Phase 2. "
        "Please use isochrone_loglike function directly for now."
    )