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

# Import tracks and grid classes
try:
    from .individual import EEPTracks, StarGrid
except ImportError:
    # Tracks/grid classes might not be available during transition
    EEPTracks = None
    StarGrid = None

# Import isochrones classes
try:
    from .populations import Isochrone
except ImportError:
    # Isochrone classes might not be available during transition
    Isochrone = None

# Build __all__ list dynamically based on what's available
__all__ = []

# Add SED utilities if available
if _get_seds is not None and get_seds is not None:
    __all__.extend(["_get_seds", "get_seds"])

# Add neural network classes if available
if FastNN is not None and FastNNPredictor is not None:
    __all__.extend(["FastNN", "FastNNPredictor"])

# Add tracks and grid classes if available
if EEPTracks is not None:
    __all__.extend(["EEPTracks"])
if StarGrid is not None:
    __all__.extend(["StarGrid"])

# Add isochrones classes if available
if Isochrone is not None:
    __all__.extend(["Isochrone"])
