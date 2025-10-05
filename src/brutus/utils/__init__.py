#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus utilities module: Mathematical, photometric, and sampling utilities.

This module contains utility functions that were previously scattered
throughout the codebase, now organized by functionality.
"""

# Mathematical functions
from .math import (
    _function_wrapper,
    chisquare_logpdf,
    inverse3,
    isPSD,
    truncnorm_logpdf,
    truncnorm_pdf,
)

# Import from reorganized modules
# Photometry functions
from .photometry import (
    add_mag,
    inv_luptitude,
    inv_magnitude,
    luptitude,
    magnitude,
    phot_loglike,
)

# Sampling utilities
from .sampling import (
    draw_sar,
    quantile,
    sample_multivariate_normal,
)

__all__ = [
    # Photometry functions
    "magnitude",
    "inv_magnitude",
    "luptitude",
    "inv_luptitude",
    "add_mag",
    "phot_loglike",
    # Mathematical functions
    "_function_wrapper",
    "inverse3",
    "isPSD",
    "chisquare_logpdf",
    "truncnorm_pdf",
    "truncnorm_logpdf",
    # Sampling utilities
    "quantile",
    "sample_multivariate_normal",
    "draw_sar",
]
