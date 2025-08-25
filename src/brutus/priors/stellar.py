#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stellar priors for Bayesian parameter estimation.

This module provides log-prior functions for stellar properties including
the initial mass function (IMF) and luminosity functions.
"""

import numpy as np

__all__ = ["logp_imf", "logp_ps1_luminosity_function"]


def logp_imf(mgrid, alpha_low=1.3, alpha_high=2.3, mass_break=0.5, mgrid2=None):
    """
    Log-prior for a Kroupa-like broken initial mass function.

    Implements a broken power-law IMF with separate slopes for low and high
    stellar masses, following Kroupa (2001). Supports binary systems with
    a secondary mass component.

    Parameters
    ----------
    mgrid : array_like
        Grid of initial stellar masses in solar units. Must be > 0.
    alpha_low : float, optional
        Power-law slope for low-mass stars (M ≤ mass_break).
        Default is 1.3 (Kroupa 2001).
    alpha_high : float, optional
        Power-law slope for high-mass stars (M > mass_break).
        Default is 2.3 (Kroupa 2001).
    mass_break : float, optional
        Transition mass between low and high mass regimes in solar units.
        Default is 0.5.
    mgrid2 : array_like, optional
        Grid of secondary stellar masses for binary systems in solar units.
        If provided, computes joint prior for binary system.

    Returns
    -------
    logp : array_like
        Normalized log-prior probability density for the input mass grid(s).
        Returns -inf for masses below hydrogen burning limit (0.08 solar masses).

    Notes
    -----
    The IMF follows the form:

    .. math::
        \\xi(M) \\propto M^{-\\alpha}

    where α = α_low for M ≤ M_break and α = α_high for M > M_break.

    For binary systems, assumes independent sampling from the same IMF
    for both components.

    References
    ----------
    Kroupa, P. (2001), MNRAS, 322, 231
    """
    mgrid = np.asarray(mgrid)

    # Initialize log-prior with -inf for invalid masses
    logp = np.full_like(mgrid, -np.inf, dtype=float)

    # Hydrogen burning limit
    valid_mass = mgrid > 0.08

    # Low-mass regime: M ≤ mass_break
    low_mass = valid_mass & (mgrid <= mass_break)
    logp[low_mass] = -alpha_low * np.log(mgrid[low_mass])

    # High-mass regime: M > mass_break
    high_mass = valid_mass & (mgrid > mass_break)
    logp[high_mass] = -alpha_high * np.log(mgrid[high_mass]) + (
        alpha_high - alpha_low
    ) * np.log(mass_break)

    # Compute normalization factor
    norm_low = mass_break ** (1.0 - alpha_low) / (alpha_high - 1.0)
    norm_high = (0.08 ** (1.0 - alpha_low) - mass_break ** (1.0 - alpha_low)) / (
        alpha_low - 1.0
    )
    norm = norm_low + norm_high

    # Handle binary component if provided
    if mgrid2 is not None:
        mgrid2 = np.asarray(mgrid2)

        # Compute prior for secondary
        logp2 = np.full_like(mgrid2, -np.inf, dtype=float)

        valid_mass2 = mgrid2 > 0.08
        low_mass2 = valid_mass2 & (mgrid2 <= mass_break)
        high_mass2 = valid_mass2 & (mgrid2 > mass_break)

        logp2[low_mass2] = -alpha_low * np.log(mgrid2[low_mass2])
        logp2[high_mass2] = -alpha_high * np.log(mgrid2[high_mass2]) + (
            alpha_high - alpha_low
        ) * np.log(mass_break)

        # Combined log-prior for binary system
        logp = logp + logp2

        # Updated normalization for binary (independent sampling)
        norm = norm**2

    # Apply normalization
    return logp - np.log(norm)


# Global interpolator for PS1 luminosity function (loaded on first use)
_ps1_lf_interpolator = None


def logp_ps1_luminosity_function(Mr):
    """
    Pan-STARRS 1 r-band luminosity function log-prior.

    Implements the stellar luminosity function derived from Pan-STARRS 1
    observations, specifically calibrated for use with Bayestar stellar
    evolutionary models and dust maps.

    Parameters
    ----------
    Mr : array_like
        Absolute r-band magnitudes in the Pan-STARRS 1 photometric system.

    Returns
    -------
    logp : array_like
        Log-prior probability density for the given absolute magnitudes.

    Notes
    -----
    This prior is designed for integration with:

    - Bayestar stellar evolutionary tracks
    - Bayestar 3D dust extinction maps
    - Pan-STARRS 1 photometric system

    The luminosity function is based on empirical measurements from the
    Pan-STARRS 1 survey and provides realistic stellar population weights
    for Bayesian inference.

    References
    ----------
    Green et al. (2015) - 3D Dust Mapping with Pan-STARRS 1
    Green et al. (2018) - Bayestar dust maps
    """
    global _ps1_lf_interpolator

    # Load data file on first use
    if _ps1_lf_interpolator is None:
        import os
        from scipy.interpolate import interp1d

        # Get path to data file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(module_dir, "PSMrLF_lnprior.dat")

        # Load PS1 luminosity function data
        grid_Mr, grid_lnp = np.loadtxt(data_path).T

        # Create interpolator with extrapolation
        _ps1_lf_interpolator = interp1d(
            grid_Mr, grid_lnp, fill_value="extrapolate", kind="linear"
        )

    # Evaluate log-prior
    return _ps1_lf_interpolator(Mr)
