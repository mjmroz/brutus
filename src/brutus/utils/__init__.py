#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus utilities module: Mathematical, photometric, and sampling utilities.

This module contains utility functions that were previously scattered
throughout the codebase, now organized by functionality.
"""

# For now, import from the original locations to maintain compatibility
# These will be moved and organized in Week 2
try:
    from ..utils import (
        magnitude, inv_magnitude, luptitude, inv_luptitude,
        quantile, sample_multivariate_normal,
        _chisquare_logpdf, _inverse3
    )
    
    __all__ = [
        'magnitude', 'inv_magnitude', 'luptitude', 'inv_luptitude',
        'quantile', 'sample_multivariate_normal'
    ]
except ImportError:
    # During transition, the modules might not be available yet
    __all__ = []