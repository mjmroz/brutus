#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus analysis module: Advanced analysis workflows and statistical methods.

This module contains functionality for complex analysis workflows including
photometric offset computation, stellar parameter optimization, and
statistical analysis tools.
"""

# Import analysis utilities
try:
    from .offsets import photometric_offsets, PhotometricOffsetsConfig

    __all__ = [
        # Photometric offset analysis
        "photometric_offsets",
        "PhotometricOffsetsConfig",
    ]
except ImportError:
    # During development, modules might not be available yet
    __all__ = []

# NOTE: Additional analysis modules will be added in Phase 2:
# - individual.py: Individual star fitting (BruteForce class)
# - clusters.py: Cluster fitting (isochrone_loglike function)  
# - samplers.py: MCMC and nested sampling
# - optimizers.py: Optimization methods
