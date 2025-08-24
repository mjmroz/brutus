#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus utilities module: Mathematical, photometric, and sampling utilities.

This module contains utility functions that were previously scattered
throughout the codebase, now organized by functionality.
"""

# Import from reorganized modules
try:
    # Photometry functions
    from .photometry import (
        magnitude,
        inv_magnitude,
        luptitude,
        inv_luptitude,
        add_mag,
        phot_loglike,
    )

    # Mathematical functions
    from .math import (
        _function_wrapper,
        adjoint3,
        dot3,
        inverse_transpose3,
        inverse3,
        isPSD,
        chisquare_logpdf,
        truncnorm_pdf,
        truncnorm_logpdf,
    )

    # Sampling utilities
    from .sampling import (
        quantile,
        sample_multivariate_normal,
        draw_sar,
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
        "adjoint3",
        "dot3",
        "inverse_transpose3",
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
except ImportError:
    # During transition, the modules might not be available yet
    __all__ = []
