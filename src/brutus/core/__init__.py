#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus core module: Isochrones, tracks, and stellar evolution models.

This module contains the fundamental stellar evolution model classes
and utilities for generating synthetic stellar data.
"""

# For now, import from the original locations to maintain compatibility
# These will be moved here in Week 2
try:
    from ..seds import Isochrone, MISTtracks, SEDmaker, FastNN, FastNNPredictor
    
    __all__ = [
        'Isochrone', 'MISTtracks', 'SEDmaker', 
        'FastNN', 'FastNNPredictor'
    ]
except ImportError:
    # During transition, the modules might not be available yet
    __all__ = []