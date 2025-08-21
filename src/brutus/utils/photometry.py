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
        # Compute log-pdf of chi2 distribution.
        dof = Ndim - 3  # effective degrees of freedom
        mask_valid_dof = dof > 0

        lnl_dim = np.full_like(lnl, -np.inf)
        for i in range(Nobj):
            if mask_valid_dof[i]:
                a = 0.5 * dof[i]
                lnl_dim[i] = (
                    xlogy(a - 1.0, chi2[i])
                    - (chi2[i] / 2.0)
                    - gammaln(a)
                    - (np.log(2.0) * a)
                )

        lnl = lnl_dim

    return lnl


def photometric_offsets(
    phot,
    err,
    mask,
    models,
    idxs,
    reds,
    dreds,
    dists,
    sel=None,
    weights=None,
    mask_fit=None,
    Nmc=150,
    old_offsets=None,
    dim_prior=True,
    prior_mean=None,
    prior_std=None,
    verbose=True,
    rstate=None,
):
    """
    Compute (multiplicative) photometric offsets between data and model.

    Parameters
    ----------
    phot : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        The observed fluxes for all our objects.

    err : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        The associated flux errors for all our objects.

    mask : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        The associated band mask for all our objects.

    models : `~numpy.ndarray` of shape `(Nmodels, Nfilt, Ncoeffs)`
        Array of magnitude polynomial coefficients used to generate
        reddened photometry.

    idxs : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Set of models fit to each object.

    reds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Associated set of reddenings (Av values) derived for each object.

    dreds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Associated set of reddening curve shapes (Rv values) derived
        for each object.

    dists : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Associated set of distances (kpc) derived for each object.

    sel : `~numpy.ndarray` of shape `(Nobj)`, optional
        Boolean selection array of objects that should be used when
        computing offsets. If not provided, all objects will be used.

    weights : `~numpy.ndarray` of shape `(Nobj, Nsamps)`, optional
        Associated set of weights for each sample.

    mask_fit : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Boolean selection array indicating the filters that were used
        in the fit. If a filter was used, the models will be re-weighted
        ignoring that band when computing the photometric offsets. If a filter
        was not used, then no additional re-weighting will be applied.
        If not provided, by default all bands will be assumed to have been
        used.

    Nmc : int, optional
        Number of realizations used to bootstrap the sample and
        average over the model realizations. Default is `150`.

    old_offsets : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative photometric offsets that were applied to
        the data (i.e. `phot *= old_offsets`). If provided, these will be
        "backed out" before computing new offsets. Default is `None`.

    dim_prior : bool, optional
        Whether to apply the dimensionality prior when computing the
        log-likelihood. Default is `True`.

    prior_mean : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Prior means for the offsets in each band. Default is `None`.

    prior_std : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Prior standard deviations for the offsets in each band.
        Default is `None`.

    verbose : bool, optional
        Whether to print progress. Default is `True`.

    rstate : `~numpy.random.RandomState`, optional
        Random state for reproducible results.

    Returns
    -------
    offsets : `~numpy.ndarray` of shape `(Nfilt)`
        Array of constants that will be *multiplied* to the *data* to account
        for offsets (i.e. multiplicative flux offsets).

    """
    # This is a complex function that would require the _get_seds function
    # and other dependencies. For now, we'll include a placeholder.
    # TODO: Implement this function fully after moving _get_seds

    raise NotImplementedError(
        "photometric_offsets requires _get_seds function which will be "
        "moved to core/sed_utils.py. This function will be completed "
        "after the SED utilities are reorganized."
    )
