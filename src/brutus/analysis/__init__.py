#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus analysis module: Fitting algorithms and statistical analysis.

This module contains classes and functions for fitting individual stars
and stellar clusters, including the core brute-force Bayesian inference
algorithms.
"""

# For now, import from the original locations to maintain compatibility
# These will be moved here in Week 2
try:
    from ..fitting import BruteForce, loglike, lnpost
    from ..cluster import isochrone_loglike
    
    __all__ = [
        'BruteForce', 'loglike', 'lnpost', 'isochrone_loglike'
    ]
except ImportError:
    # During transition, the modules might not be available yet
    __all__ = []