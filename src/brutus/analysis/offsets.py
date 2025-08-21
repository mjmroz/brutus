#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced photometric offset analysis for brutus.

This module provides improved photometric offset computation with better
performance, configurability, and robustness compared to the original
utilities implementation.

File: src/brutus/analysis/offsets.py
"""

from __future__ import division, print_function

import warnings
import numpy as np
from typing import Optional, Tuple

# Import utilities (will need to be updated based on final module structure)
try:
    from ..core.sed_utils import get_seds
    from ..utils.photometry import phot_loglike
    from ..utils.sampling import quantile
    from scipy.special import logsumexp
except ImportError:
    # Fallback for development/testing
    warnings.warn("Could not import brutus utilities, using placeholder imports")

    def get_seds(*args, **kwargs):
        raise NotImplementedError("get_seds not available")

    def phot_loglike(*args, **kwargs):
        raise NotImplementedError("phot_loglike not available")

    def quantile(*args, **kwargs):
        raise NotImplementedError("quantile not available")

    def logsumexp(*args, **kwargs):
        raise NotImplementedError("logsumexp not available")


__all__ = ["photometric_offsets", "PhotometricOffsetsConfig"]


class PhotometricOffsetsConfig:
    """
    Configuration class for photometric offsets computation.

    This class encapsulates all configuration parameters and provides
    sensible defaults with the ability to customize behavior.

    Parameters
    ----------
    min_bands_used : int, optional
        Minimum number of bands required for objects where the current
        band was used in fitting. Default is 4 (equivalent to >3+1).

    min_bands_unused : int, optional
        Minimum number of bands required for objects where the current
        band was not used in fitting. Default is 3.

    n_bootstrap : int, optional
        Number of bootstrap realizations for uncertainty estimation.
        Default is 300.

    uncertainty_method : str, optional
        Method for uncertainty estimation. Options:
        - 'bootstrap_std': Standard deviation of bootstrap medians
        - 'bootstrap_iqr': Scaled interquartile range of bootstrap medians
        Default is 'bootstrap_iqr'.

    progress_interval : int, optional
        Print progress every N iterations. Set to 0 for no progress.
        Default is 10.

    use_vectorized_bootstrap : bool, optional
        Use vectorized bootstrap implementation for better performance.
        Default is True.

    random_seed : int, optional
        Random seed for reproducible results. Default is None.

    validate_inputs : bool, optional
        Perform input validation. Default is True.
    """

    def __init__(
        self,
        min_bands_used: int = 4,
        min_bands_unused: int = 3,
        n_bootstrap: int = 300,
        uncertainty_method: str = "bootstrap_iqr",
        progress_interval: int = 10,
        use_vectorized_bootstrap: bool = True,
        random_seed: Optional[int] = None,
        validate_inputs: bool = True,
    ):
        self.min_bands_used = min_bands_used
        self.min_bands_unused = min_bands_unused
        self.n_bootstrap = n_bootstrap
        self.uncertainty_method = uncertainty_method
        self.progress_interval = progress_interval
        self.use_vectorized_bootstrap = use_vectorized_bootstrap
        self.random_seed = random_seed
        self.validate_inputs = validate_inputs

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.min_bands_used < 1:
            raise ValueError("min_bands_used must be >= 1")
        if self.min_bands_unused < 1:
            raise ValueError("min_bands_unused must be >= 1")
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")
        if self.uncertainty_method not in [
            "bootstrap_std",
            "bootstrap_iqr",
        ]:
            raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method}")
        if self.progress_interval < 0:
            raise ValueError("progress_interval must be >= 0")


def _validate_inputs(
    phot: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    models: np.ndarray,
    idxs: np.ndarray,
    reds: np.ndarray,
    dreds: np.ndarray,
    dists: np.ndarray,
) -> None:
    """Validate input arrays for photometric_offsets."""

    # Check basic types
    arrays = [phot, err, mask, models, idxs, reds, dreds, dists]
    names = ["phot", "err", "mask", "models", "idxs", "reds", "dreds", "dists"]

    for arr, name in zip(arrays, names):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be numpy array, got {type(arr)}")

    # Check shapes
    nobj, nfilt = phot.shape
    nsamps = idxs.shape[1]

    if err.shape != (nobj, nfilt):
        raise ValueError(f"err shape {err.shape} != phot shape {phot.shape}")
    if mask.shape != (nobj, nfilt):
        raise ValueError(f"mask shape {mask.shape} != phot shape {phot.shape}")
    if idxs.shape != (nobj, nsamps):
        raise ValueError(f"idxs shape {idxs.shape} != expected ({nobj}, {nsamps})")
    if reds.shape != (nobj, nsamps):
        raise ValueError(f"reds shape {reds.shape} != expected ({nobj}, {nsamps})")
    if dreds.shape != (nobj, nsamps):
        raise ValueError(f"dreds shape {dreds.shape} != expected ({nobj}, {nsamps})")
    if dists.shape != (nobj, nsamps):
        raise ValueError(f"dists shape {dists.shape} != expected ({nobj}, {nsamps})")

    # Check for valid values
    if not np.all(np.isfinite(phot)):
        raise ValueError("phot contains non-finite values")
    if not np.all(err > 0):
        raise ValueError("err must be positive")
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("mask must contain only 0s and 1s")
    if not np.all(dists > 0):
        raise ValueError("dists must be positive")


def _vectorized_bootstrap_median(
    ratios: np.ndarray,
    weights: np.ndarray,
    obj_weights: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized bootstrap implementation for better performance.

    Parameters
    ----------
    ratios : np.ndarray of shape (n_objects, n_samples)
        Model/data ratios for each object and sample
    weights : np.ndarray of shape (n_objects, n_samples)
        Model weights for each object and sample
    obj_weights : np.ndarray of shape (n_objects,)
        Object selection weights
    n_bootstrap : int
        Number of bootstrap realizations
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    bootstrap_medians : np.ndarray of shape (n_bootstrap,)
        Bootstrap median estimates
    """
    n_objects = len(ratios)
    bootstrap_medians = np.zeros(n_bootstrap)

    # Pre-generate random indices for better performance
    obj_indices = rng.choice(n_objects, size=(n_bootstrap, n_objects), p=obj_weights)

    for i in range(n_bootstrap):
        # Sample objects
        selected_ratios = ratios[obj_indices[i]]
        selected_weights = weights[obj_indices[i]]

        # Sample models for each selected object
        model_indices = np.array(
            [rng.choice(len(w), p=w) if np.sum(w) > 0 else 0 for w in selected_weights]
        )

        # Get final ratios and compute median
        final_ratios = selected_ratios[np.arange(n_objects), model_indices]
        bootstrap_medians[i] = np.median(final_ratios)

    return bootstrap_medians


def photometric_offsets(
    phot: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    models: np.ndarray,
    idxs: np.ndarray,
    reds: np.ndarray,
    dreds: np.ndarray,
    dists: np.ndarray,
    sel: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    mask_fit: Optional[np.ndarray] = None,
    old_offsets: Optional[np.ndarray] = None,
    dim_prior: bool = True,
    prior_mean: Optional[np.ndarray] = None,
    prior_std: Optional[np.ndarray] = None,
    verbose: bool = True,
    config: Optional[PhotometricOffsetsConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multiplicative photometric offsets between data and models.

    This function computes photometric offsets that account for systematic
    differences between observed photometry and model predictions. The offsets
    are computed by comparing model/data flux ratios across a sample of objects,
    with proper uncertainty estimation and optional prior constraints.

    Parameters
    ----------
    phot : np.ndarray of shape (n_objects, n_filters)
        Observed flux densities for all objects.

    err : np.ndarray of shape (n_objects, n_filters)
        Associated flux errors for all objects.

    mask : np.ndarray of shape (n_objects, n_filters)
        Binary mask (0/1) indicating observed bands for each object.

    models : np.ndarray of shape (n_models, n_filters, n_coeffs)
        Magnitude polynomial coefficients for generating reddened photometry.

    idxs : np.ndarray of shape (n_objects, n_samples)
        Model indices fit to each object.

    reds : np.ndarray of shape (n_objects, n_samples)
        A(V) reddening values for each object and sample.

    dreds : np.ndarray of shape (n_objects, n_samples)
        R(V) reddening curve shape values for each object and sample.

    dists : np.ndarray of shape (n_objects, n_samples)
        Distance values (kpc) for each object and sample.

    sel : np.ndarray of shape (n_objects), optional
        Boolean selection of objects to use. Default uses all objects.

    weights : np.ndarray of shape (n_objects, n_samples), optional
        Sample weights for each object. Default uses uniform weights.

    mask_fit : np.ndarray of shape (n_filters), optional
        Boolean mask indicating which filters were used in fitting.
        Default assumes all filters were used.

    old_offsets : np.ndarray of shape (n_filters), optional
        Previous offsets to remove before computing new ones.
        Default is no previous offsets.

    dim_prior : bool, optional
        Whether to apply dimensionality prior in likelihood reweighting.
        Default is True.

    prior_mean : np.ndarray of shape (n_filters), optional
        Gaussian prior means for offsets. Must be provided with prior_std.

    prior_std : np.ndarray of shape (n_filters), optional
        Gaussian prior standard deviations for offsets.

    verbose : bool, optional
        Whether to print progress information. Default is True.

    config : PhotometricOffsetsConfig, optional
        Configuration object with analysis parameters.
        Default uses standard configuration.

    rng : np.random.Generator, optional
        Random number generator for reproducible results.
        Default creates new generator.

    Returns
    -------
    offsets : np.ndarray of shape (n_filters)
        Multiplicative photometric offsets (model/data ratios).

    offset_errors : np.ndarray of shape (n_filters)
        Uncertainties on the photometric offsets.

    n_objects_used : np.ndarray of shape (n_filters)
        Number of objects used to compute each offset.

    Examples
    --------
    >>> import numpy as np
    >>> from brutus.analysis.offsets import photometric_offsets
    >>>
    >>> # Mock data for demonstration
    >>> n_obj, n_filt, n_samp = 100, 5, 50
    >>> phot = np.random.uniform(0.1, 10, (n_obj, n_filt))
    >>> err = 0.1 * phot
    >>> mask = np.random.choice([0, 1], (n_obj, n_filt), p=[0.1, 0.9])
    >>>
    >>> # Mock fitted parameters
    >>> models = np.random.random((1000, n_filt, 3))
    >>> idxs = np.random.randint(0, 1000, (n_obj, n_samp))
    >>> reds = np.random.uniform(0, 2, (n_obj, n_samp))
    >>> dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))
    >>> dists = np.random.uniform(0.1, 10, (n_obj, n_samp))
    >>>
    >>> # Compute offsets
    >>> offsets, errors, n_used = photometric_offsets(
    ...     phot, err, mask, models, idxs, reds, dreds, dists
    ... )
    >>> print(f"Computed offsets: {offsets}")

    Notes
    -----
    The photometric offset for each band is computed as:

    1. Generate model SEDs for all fitted objects
    2. Scale by distance: flux_model ∝ 1/distance^2
    3. Compute model/data flux ratios
    4. Reweight by excluding current band (if used in fit)
    5. Bootstrap sample ratios and take median +/- uncertainty
    6. Apply priors if provided

    The offsets should be applied as: corrected_data = observed_data * offsets
    """

    # Handle configuration
    if config is None:
        config = PhotometricOffsetsConfig()

    # Set up random number generator
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    # Validate inputs
    if config.validate_inputs:
        _validate_inputs(phot, err, mask, models, idxs, reds, dreds, dists)

    # Initialize parameters
    nobj, nfilt = phot.shape
    nsamps = idxs.shape[1]

    if sel is None:
        sel = np.ones(nobj, dtype=bool)
    if weights is None:
        weights = np.ones((nobj, nsamps), dtype=float)
    if mask_fit is None:
        mask_fit = np.ones(nfilt, dtype=bool)
    if old_offsets is None:
        old_offsets = np.ones(nfilt)

    # Generate model SEDs
    if verbose and config.progress_interval > 0:
        print("Generating model SEDs...")

    seds = get_seds(
        models[idxs.flatten()], av=reds.flatten(), rv=dreds.flatten(), return_flux=True
    )

    # Scale by distance
    seds /= dists.flatten()[:, None] ** 2
    seds = seds.reshape(nobj, nsamps, nfilt)

    # Initialize output arrays
    offsets = np.ones(nfilt)
    offset_errors = np.zeros(nfilt)
    n_objects_used = np.zeros(nfilt, dtype=int)

    # Process each filter
    for i in range(nfilt):
        if verbose and config.progress_interval > 0:
            print(f"Processing filter {i+1}/{nfilt}...")

        # Select objects with sufficient coverage
        min_bands = config.min_bands_used if mask_fit[i] else config.min_bands_unused

        if mask_fit[i]:
            # Exclude current band from count
            band_counts = np.sum(mask, axis=1) - mask[:, i]
            valid_objects = (
                mask[:, i]
                & sel
                & (band_counts >= min_bands - 1)
                & (np.sum(weights, axis=1) > 0)
            )
        else:
            # Don't exclude current band
            band_counts = np.sum(mask, axis=1)
            valid_objects = (
                mask[:, i]
                & sel
                & (band_counts >= min_bands)
                & (np.sum(weights, axis=1) > 0)
            )

        obj_indices = np.where(valid_objects)[0]
        n = len(obj_indices)
        n_objects_used[i] = n

        if n == 0:
            if verbose:
                print(f"  Warning: No valid objects for filter {i+1}")
            continue

        # Compute model/data ratios
        ratios = seds[obj_indices, :, i] / phot[obj_indices, None, i]

        # Compute weights (reweight if band was used in fit)
        if mask_fit[i]:
            # Recompute likelihoods excluding current band
            model_weights = np.zeros((n, nsamps))

            for j, obj_idx in enumerate(obj_indices):
                # Create temporary mask excluding current band
                temp_mask = mask[obj_idx].copy()
                temp_mask[i] = False

                # Compute likelihoods for each model
                lnl = np.array(
                    [
                        phot_loglike(
                            phot[obj_idx] * old_offsets,
                            err[obj_idx] * old_offsets,
                            temp_mask,
                            seds[obj_idx, k],
                            dim_prior=dim_prior,
                        )
                        for k in range(nsamps)
                    ]
                )

                # Convert to weights
                log_evidence = logsumexp(lnl)
                log_weights = lnl - log_evidence
                model_weights[j] = np.exp(log_weights)
        else:
            # Use uniform weights
            model_weights = np.ones((n, nsamps))

        # Apply sample weights
        model_weights *= weights[obj_indices]

        # Normalize weights
        weight_sums = np.sum(model_weights, axis=1)
        nonzero_mask = weight_sums > 0
        model_weights[nonzero_mask] /= weight_sums[nonzero_mask, None]

        # Object weights for bootstrap
        obj_weights = (weight_sums > 0).astype(float)
        if np.sum(obj_weights) > 0:
            obj_weights /= np.sum(obj_weights)
        else:
            obj_weights = np.ones(n) / n

        # Bootstrap uncertainty estimation
        if config.use_vectorized_bootstrap and n > 0:
            bootstrap_medians = _vectorized_bootstrap_median(
                ratios, model_weights, obj_weights, config.n_bootstrap, rng
            )
        else:
            # Original bootstrap implementation
            bootstrap_medians = []
            for j in range(config.n_bootstrap):
                if (
                    verbose
                    and config.progress_interval > 0
                    and j % config.progress_interval == 0
                ):
                    print(f"  Bootstrap {j+1}/{config.n_bootstrap}")

                # Sample objects
                obj_sample = rng.choice(n, size=n, p=obj_weights)

                # Sample models
                model_sample = np.array(
                    [
                        (
                            rng.choice(nsamps, p=model_weights[k])
                            if np.sum(model_weights[k]) > 0
                            else 0
                        )
                        for k in obj_sample
                    ]
                )

                # Compute median
                sample_ratios = ratios[obj_sample, model_sample]
                bootstrap_medians.append(np.median(sample_ratios))

            bootstrap_medians = np.array(bootstrap_medians)

        # Compute offset and uncertainty
        offsets[i] = np.median(bootstrap_medians)

        if config.uncertainty_method == "bootstrap_std":
            offset_errors[i] = np.std(bootstrap_medians)
        elif config.uncertainty_method == "bootstrap_iqr":
            q25, q75 = np.percentile(bootstrap_medians, [25, 75])
            offset_errors[i] = (q75 - q25) / 1.349  # Convert IQR to std equivalent

    # Apply priors if provided
    if prior_mean is not None and prior_std is not None:
        if len(prior_mean) != nfilt or len(prior_std) != nfilt:
            raise ValueError("Prior arrays must have length n_filters")

        var_total = offset_errors**2 + prior_std**2
        offsets = (offsets * prior_std**2 + prior_mean * offset_errors**2) / var_total
        offset_errors = offset_errors * prior_std / np.sqrt(var_total)

    if verbose:
        print("Photometric offset computation complete.")

    return offsets, offset_errors, n_objects_used
