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
    from .individual import BruteForce
    from .populations import (
        isochrone_population_loglike,
        generate_isochrone_population_grid,
        compute_isochrone_cluster_likelihood,
        compute_isochrone_outlier_likelihood,
        apply_isochrone_mixture_model,
        marginalize_isochrone_grid,
    )

    __all__ = [
        # Individual star fitting
        "BruteForce",
        # Stellar population analysis (isochrone-based)
        "isochrone_population_loglike",
        "generate_isochrone_population_grid",
        "compute_isochrone_cluster_likelihood",
        "compute_isochrone_outlier_likelihood",
        "apply_isochrone_mixture_model",
        "marginalize_isochrone_grid",
        # Photometric offset analysis
        "photometric_offsets",
        "PhotometricOffsetsConfig",
    ]
except ImportError:
    # During development, modules might not be available yet
    __all__ = []
