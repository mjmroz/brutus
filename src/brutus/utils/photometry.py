#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Photometric utility functions for brutus.

This module contains functions for converting between magnitudes and fluxes,
handling asinh magnitudes ("luptitudes"), and computing photometric likelihoods.
"""

import numpy as np
from numba import jit
from scipy.special import gammaln, xlogy

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = [
    "magnitude",
    "inv_magnitude",
    "luptitude",
    "inv_luptitude",
    "add_mag",
    "phot_loglike",
    "photometric_offsets",
]


def magnitude(phot, err, zeropoints=1.0):
    """
    Convert photometry to AB magnitudes.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes corresponding to input `phot`.

    mag_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes errors corresponding to input `err`.

    """
    # Compute magnitudes.
    mag = -2.5 * np.log10(phot / zeropoints)

    # Compute errors.
    mag_err = 2.5 / np.log(10.0) * err / phot

    return mag, mag_err


def inv_magnitude(mag, err, zeropoints=1.0):
    """
    Convert AB magnitudes to photometry.

    Parameters
    ----------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitude errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `mag`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """
    # Compute magnitudes.
    phot = 10 ** (-0.4 * mag) * zeropoints

    # Compute errors.
    phot_err = err * 0.4 * np.log(10.0) * phot

    return phot, phot_err


def luptitude(phot, err, skynoise=1.0, zeropoints=1.0):
    """
    Convert photometry to asinh magnitudes (i.e. "Luptitudes"). See Lupton et
    al. (1999) for more details.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    lupt : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitudes corresponding to input `phot`.

    lupt_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitudes errors corresponding to input `err`.

    """
    # Normalize photometry.
    f = phot / zeropoints
    df = err / zeropoints
    b = skynoise / zeropoints

    # Compute luptitudes.
    lupt = -2.5 / np.log(10.0) * (np.arcsinh(0.5 * f / b) + np.log(b))

    # Compute errors.
    lupt_err = 2.5 / np.log(10.0) * df / (2.0 * b * np.sqrt(1.0 + (0.5 * f / b) ** 2))

    return lupt, lupt_err


def inv_luptitude(lupt, err, skynoise=1.0, zeropoints=1.0):
    """
    Convert asinh magnitudes (i.e. "Luptitudes") to photometry. See Lupton et
    al. (1999) for more details.

    Parameters
    ----------
    lupt : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitude errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `lupt`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """
    # Normalize sky noise.
    b = skynoise / zeropoints

    # Compute flux.
    f = 2.0 * b * np.sinh(-0.4 * np.log(10.0) * lupt - np.log(b))

    # Compute errors.
    df = (
        err
        * 0.4
        * np.log(10.0)
        * 2.0
        * b
        * np.cosh(-0.4 * np.log(10.0) * lupt - np.log(b))
    )

    # Convert back to original units.
    phot = f * zeropoints
    phot_err = df * zeropoints

    return phot, phot_err


def add_mag(mag1, mag2):
    """
    Add magnitudes.

    Parameters
    ----------
    mag1 : float or `~numpy.ndarray`
        First set of magnitudes.

    mag2 : float or `~numpy.ndarray`
        Second set of magnitudes.

    Returns
    -------
    mag_combined : float or `~numpy.ndarray`
        Combined magnitudes corresponding to the combined flux from
        `mag1` and `mag2`.

    """
    # Compute combined flux.
    flux_combined = 10 ** (-0.4 * mag1) + 10 ** (-0.4 * mag2)

    # Convert back to magnitudes.
    mag_combined = -2.5 * np.log10(flux_combined)

    return mag_combined


def phot_loglike(flux, err, mfluxes, mask=None, dim_prior=False):
    """
    Compute the log-likelihood between observed and model fluxes.

    Parameters
    ----------
    flux : `~numpy.ndarray` with shape (Nobj, Nfilt)
        Observed flux values.

    err : `~numpy.ndarray` with shape (Nobj, Nfilt)
        Associated flux errors.

    mfluxes : `~numpy.ndarray` with shape (Nobj, Nmod, Nfilt)
        Model fluxes (for each model).

    mask : `~numpy.ndarray` with shape (Nobj, Nfilt), optional
        Binary mask indicating whether each observed band can be
        used (1) or should be skipped (0).

    dim_prior : bool, optional
        Whether to apply a dimensionality prior from the
        chi-squared distribution. Default is `False`.

    Returns
    -------
    lnl : `~numpy.ndarray` with shape (Nobj, Nmod)
        Log-likelihood values.

    """
    # Initialize values.
    Nobj, Nfilt = flux.shape[:2]
    Nmod = mfluxes.shape[1]

    # Ensure proper dimensions.
    if mfluxes.shape != (Nobj, Nmod, Nfilt):
        raise ValueError("Inconsistent dimensions between flux and mfluxes")

    if mask is None:
        mask = np.ones_like(flux)

    # Apply mask to get effective dimensionality per object.
    Ndim = np.sum(mask, axis=1)

    # Compute variance (including errors).
    var = err**2

    # Mask fluxes and model fluxes.
    flux_masked = flux[:, None, :] * mask[:, None, :]  # (Nobj, 1, Nfilt)
    mfluxes_masked = mfluxes * mask[:, None, :]  # (Nobj, Nmod, Nfilt)
    var_masked = var[:, None, :] * mask[:, None, :]  # (Nobj, 1, Nfilt)

    # Compute residuals.
    resid = flux_masked - mfluxes_masked  # (Nobj, Nmod, Nfilt)

    # Compute chi-squared.
    chi2 = np.sum(resid**2 / np.where(var_masked > 0, var_masked, np.inf), axis=2)

    # Compute log-likelihood.
    lnl = -0.5 * chi2

    # Add normalization term.
    log_det_term = np.sum(np.log(var) * mask, axis=1)  # (Nobj,)
    lnl += -0.5 * (Ndim[:, None] * np.log(2.0 * np.pi) + log_det_term[:, None])

    # Apply dimensionality prior if requested.
    if dim_prior:
        # Compute log-pdf of chi2 distribution in a vectorized way.
        dof = Ndim - 3  # effective degrees of freedom
        a = 0.5 * dof[:, None]  # shape (Nobj, 1)
        mask_valid_dof = dof > 0  # shape (Nobj,)

        # Broadcast chi2 to (Nobj, Nmod)
        lnl_dim = np.full_like(lnl, -np.inf)
        valid_idx = np.where(mask_valid_dof)[0]
        if valid_idx.size > 0:
            chi2_valid = chi2[valid_idx]
            a_valid = a[valid_idx]
            lnl_dim[valid_idx] = (
                xlogy(a_valid - 1.0, chi2_valid)
                - (chi2_valid / 2.0)
                - gammaln(a_valid)
                - (np.log(2.0) * a_valid)
            )

        lnl = lnl_dim

    return lnl
