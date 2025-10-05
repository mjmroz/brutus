#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus analysis module: Advanced analysis workflows and statistical methods.

This module contains functionality for complex analysis workflows including
photometric offset computation, stellar parameter optimization, and
statistical analysis tools.
"""

from .individual import BruteForce
from .los_dust import (
    kernel_gauss,
    kernel_lorentz,
    kernel_tophat,
    los_clouds_loglike_samples,
    los_clouds_priortransform,
)

# Import analysis utilities
from .offsets import PhotometricOffsetsConfig, photometric_offsets
from .populations import (
    apply_isochrone_mixture_model,
    compute_isochrone_cluster_loglike,
    compute_isochrone_outlier_loglike,
    generate_isochrone_population_grid,
    isochrone_population_loglike,
    marginalize_isochrone_grid,
)

__all__ = [
    # Individual star fitting
    "BruteForce",
    # Stellar population analysis (isochrone-based)
    "isochrone_population_loglike",
    "generate_isochrone_population_grid",
    "compute_isochrone_cluster_loglike",
    "compute_isochrone_outlier_loglike",
    "apply_isochrone_mixture_model",
    "marginalize_isochrone_grid",
    # Photometric offset analysis
    "photometric_offsets",
    "PhotometricOffsetsConfig",
    # Line-of-sight dust extinction analysis
    "los_clouds_priortransform",
    "los_clouds_loglike_samples",
    "kernel_tophat",
    "kernel_gauss",
    "kernel_lorentz",
]
