#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus core module: Fundamental stellar modeling utilities.

This module contains the core functionality for stellar evolution modeling,
including isochrones, tracks, SED computation, neural network utilities,
and grid generation.
"""

# Import core components
from .sed_utils import _get_seds, get_seds
from .neural_nets import FastNN, FastNNPredictor
from .individual import EEPTracks, StarGrid
from .populations import Isochrone
from .grid_generation import GridGenerator

__all__ = [
    "_get_seds",
    "get_seds",
    "FastNN",
    "FastNNPredictor",
    "EEPTracks",
    "StarGrid",
    "Isochrone",
    "GridGenerator",
]
