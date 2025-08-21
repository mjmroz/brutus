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

# TODO: Add imports for other analysis modules when they are reorganized:
# from .individual import BruteForce
# from .clusters import isochrone_loglike
# from .priors import *
# from .sampling import *
