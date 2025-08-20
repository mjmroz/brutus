#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus dust module: 3D dust mapping and extinction modeling.

This module contains classes and functions for working with 3D dust maps,
extinction laws, and line-of-sight dust modeling.
"""

# For now, import from the original locations to maintain compatibility
# These will be moved here in Week 2
try:
    from ..dust import Bayestar, DustMap
    from ..los import los_loglike
    
    __all__ = [
        'Bayestar', 'DustMap', 'los_loglike'
    ]
except ImportError:
    # During transition, the modules might not be available yet
    __all__ = []