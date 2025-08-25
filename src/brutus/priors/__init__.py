#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prior probability distributions for stellar and galactic modeling.

This module provides log-prior functions for Bayesian stellar parameter estimation,
including stellar, astrometric, galactic structure, and extinction priors.
All functions follow the naming convention logp_* for log-probability densities
and logn_* for log-number densities.
"""

# Stellar priors
from .stellar import logp_imf, logp_ps1_luminosity_function

# Astrometric priors
from .astrometric import logp_parallax, logp_parallax_scale, convert_parallax_to_scale

# Galactic structure priors
from .galactic import (
    logp_galactic_structure,
    logn_disk,
    logn_halo,
    logp_feh,
    logp_age_from_feh,
)

# Extinction priors
from .extinction import logp_extinction

# Utility functions
from .utilities import bin_distance_reddening_pdfs

__all__ = [
    # Stellar priors
    "logp_imf",
    "logp_ps1_luminosity_function",
    # Astrometric priors
    "logp_parallax",
    "logp_parallax_scale",
    "convert_parallax_to_scale",
    # Galactic structure priors
    "logp_galactic_structure",
    "logn_disk",
    "logn_halo",
    "logp_feh",
    "logp_age_from_feh",
    # Extinction priors
    "logp_extinction",
    # Utilities
    "bin_distance_reddening_pdfs",
]
