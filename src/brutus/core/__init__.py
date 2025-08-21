#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus core module: Fundamental stellar modeling utilities.

This module contains the core functionality for stellar evolution modeling,
including isochrones, tracks, and SED computation utilities.
"""

# Import SED utilities
try:
    from .sed_utils import _get_seds, get_seds

    __all__ = [
        # SED utilities
        "_get_seds",
        "get_seds",
    ]
except ImportError:
    # During development, the modules might not be available yet
    __all__ = []

# TODO: Add imports for other core modules when they are reorganized:
# from .isochrones import Isochrone, SEDmaker
# from .tracks import MISTtracks
# from .models import ModelRegistry
