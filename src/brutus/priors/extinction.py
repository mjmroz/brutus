#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extinction priors for Bayesian stellar parameter estimation.

This module provides log-prior functions for dust extinction modeling
using Bayestar 3D dust maps. These priors incorporate spatial dust
distribution information to constrain extinction in stellar fitting.

Functions
---------
logp_extinction : Dust map extinction prior
    Gaussian prior from Bayestar dust maps

See Also
--------
brutus.dust.maps : 3D dust map utilities
brutus.priors.galactic : Galactic structure priors
brutus.analysis.individual.BruteForce : Uses extinction priors for fitting

Notes
-----
The extinction prior uses the Bayestar 3D dust maps (Green et al. 2015,
2018) which provide distance-dependent extinction estimates across the sky.

The prior is Gaussian when dust map data is available, and uniform when
coverage is unavailable. This gracefully handles regions outside the
mapped volume.

Examples
--------
>>> from brutus.priors.extinction import logp_extinction
>>> from astropy.coordinates import SkyCoord
>>> import numpy as np
>>>
>>> # Coordinates and dust map (assume loaded)
>>> coords = SkyCoord(ra=180, dec=30, unit='deg', frame='icrs')
>>> # dustmap = load_dustmap()  # hypothetical
>>>
>>> # Evaluate extinction prior
>>> av_values = np.linspace(0, 2, 100)
>>> # log_prior = logp_extinction(av_values, dustmap, coords)
"""

import numpy as np

__all__ = ["logp_extinction"]


def logp_extinction(avs, dustmap, coord, return_components=False):
    """
    Log-prior for dust extinction using Bayestar dust maps.

    Implements Gaussian extinction priors based on the Bayestar dust maps
    with systematic uncertainty treatment. Returns uniform prior when
    dust map coverage is unavailable.

    Parameters
    ----------
    avs : array_like
        Extinction values (A_V) in magnitudes to evaluate prior for.
    dustmap : object
        Dust map object providing query interface for mean and standard
        deviation of extinction. Expected to support coord queries.
    coord : astropy.coordinates.SkyCoord
        Sky coordinates for dust map query.
    return_components : bool, optional
        If True, returns tuple (logp, (av_mean, av_err)) including
        extracted dust map statistics. Default is False.

    Returns
    -------
    logp : array_like
        Log-prior probability density for the input extinction values.
        Returns 0 (uniform prior) when no dust map coverage available.
    components : tuple, optional
        If return_components=True, returns (av_mean, av_err) tuple
        containing dust map mean and standard deviation.

    Notes
    -----
    The log-prior follows a Gaussian distribution when dust map data
    is available:

    .. math::
        \\log p(A_V | A_{V,\\text{map}}, \\sigma_{A_V}) =
        -\\frac{1}{2} \\left[ \\frac{(A_V - A_{V,\\text{map}})^2}{\\sigma_{A_V}^2} +
        \\log(2\\pi\\sigma_{A_V}^2) \\right]

    For regions without dust map coverage, returns uniform prior.

    The dust map is expected to provide both mean extinction and
    uncertainty estimates suitable for Bayesian inference.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> coord = SkyCoord(ra=180., dec=0., unit='deg')
    >>> logp = logp_extinction([0.1, 0.2, 0.5], dustmap, coord)
    >>> logp, (mean, err) = logp_extinction([0.1], dustmap, coord, True)
    """
    avs = np.asarray(avs)

    # Query the dust map for extinction statistics
    try:
        # Most dust maps provide mean and standard deviation
        av_mean, av_err = dustmap.query(coord)

        # Check if we have valid dust map coverage
        if np.isfinite(av_mean) and np.isfinite(av_err) and av_err > 0:
            # Gaussian log-prior
            chi2 = (avs - av_mean) ** 2 / av_err**2
            lnorm = np.log(2.0 * np.pi * av_err**2)
            lnprior = -0.5 * (chi2 + lnorm)
        else:
            # No coverage - uniform prior
            lnprior = np.zeros_like(avs, dtype=float)
            av_mean, av_err = np.nan, np.nan

    except (AttributeError, ValueError):
        # Fallback for dust maps with different interfaces or errors
        lnprior = np.zeros_like(avs, dtype=float)
        av_mean, av_err = np.nan, np.nan

    if not return_components:
        return lnprior
    else:
        return lnprior, (av_mean, av_err)
