#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus core module: Fundamental stellar modeling utilities.

This module contains the core functionality for stellar evolution modeling,
including isochrones, tracks, SED computation, and neural network utilities.
"""

# Import SED utilities
try:
    from .sed_utils import _get_seds, get_seds
except ImportError:
    # SED utilities might not be available during transition
    _get_seds = None
    get_seds = None

# Import neural network classes
try:
    from .neural_nets import FastNN, FastNNPredictor
except ImportError:
    # Neural network classes might not be available during transition
    FastNN = None
    FastNNPredictor = None

# Import tracks classes
try:
    from .tracks import MISTtracks
except ImportError:
    # Tracks classes might not be available during transition
    MISTtracks = None

# Build __all__ list dynamically based on what's available
__all__ = []

# Add SED utilities if available
if _get_seds is not None and get_seds is not None:
    __all__.extend(["_get_seds", "get_seds"])

# Add neural network classes if available
if FastNN is not None and FastNNPredictor is not None:
    __all__.extend(["FastNN", "FastNNPredictor"])

# Add tracks classes if available
if MISTtracks is not None:
    __all__.extend(["MISTtracks"])

# TODO: Add imports for remaining core modules when they are reorganized:
# from .isochrones import Isochrone, SEDmaker
# from .models import ModelRegistry
