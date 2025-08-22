#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stellar isochrone interpolation and SED generation.

This module provides classes for interpolating stellar isochrones from MIST
(MESA Isochrones and Stellar Tracks) models and generating synthetic photometry
using artificial neural networks for fast bolometric correction computation.

The core functionality centers around loading pre-computed isochrone grids
and providing fast interpolation for stellar population modeling, with
support for binary stars, dust extinction, and observational uncertainties.

Classes
-------
Isochrone : Main isochrone interpolation class
    Provides linear interpolation of MIST stellar isochrones across the
    parameter space of metallicity, alpha enhancement, age, and
    equivalent evolutionary point (EEP).

Examples
--------
Basic isochrone usage:

>>> from brutus.core.isochrones import Isochrone
>>>
>>> # Initialize with default MIST isochrones
>>> iso = Isochrone()
>>>
>>> # Generate an isochrone for a 1 Gyr, solar metallicity population
>>> seds, params, params2 = iso.get_seds(feh=0.0, afe=0.0, loga=9.0,
...                                       av=0.1, dist=1000.0)
>>> print(f"Generated {len(seds)} stars with magnitudes: {seds[:5, 0]}")

Advanced usage with binary stars:

>>> # Include 50% binary fraction with secondary mass ratio 0.7
>>> seds, params, params2 = iso.get_seds(feh=0.0, afe=0.0, loga=9.0,
...                                       smf=0.7, av=0.1, dist=1000.0)

Notes
-----
The MIST isochrones provide theoretical predictions for stellar populations
across a wide range of ages, metallicities, and evolutionary phases. This
module enables fast interpolation within this grid and conversion to
observational quantities using neural network-based bolometric corrections.

The isochrone parameter space uses [Fe/H], [α/Fe], log(age), and EEP as
input dimensions, differing from the MISTtracks class which uses initial
mass instead of age as a primary parameter.

References
----------
.. [1] Choi et al. 2016, "MESA Isochrones and Stellar Tracks (MIST) 0. Methods
       for the Construction of Stellar Isochrones", ApJ, 823, 102
.. [2] Dotter 2016, "MESA Isochrones and Stellar Tracks (MIST) I. Solar-scaled
       Models", ApJS, 222, 8
"""

import sys
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

# Import from utils (to be reorganized)
from ..utils.photometry import add_mag

# Import neural network predictor
from .neural_nets import FastNNPredictor

# Import filter definitions
from ..data.filters import FILTERS


__all__ = ["Isochrone"]


class Isochrone(object):
    """
    An object that generates photometry interpolated from MIST isochrones in
    EEP, metallicity, and log(age) using artificial neural networks.

    This class provides fast interpolation of stellar isochrones from the
    MESA Isochrones and Stellar Tracks (MIST) project, enabling synthetic
    photometry generation for stellar populations with support for binary
    stars, dust extinction, and complex evolutionary modeling.

    Parameters
    ----------
    filters : list of str, optional
        The names of filters for which photometry should be computed. If not
        provided, photometry will be computed for all available filters.
        Available filters include Gaia, SDSS, Pan-STARRS, and many others.

    nnfile : str, optional
        Path to the neural network file used to generate fast bolometric
        correction predictions. If not provided, defaults to the standard
        MIST neural network file.

    mistfile : str, optional
        Path to the HDF5 file containing the MIST isochrone grid. If not
        provided, defaults to the standard MIST v1.2 isochrone file.

    predictions : list of str, optional
        The names of the stellar parameters to predict at requested locations
        in the parameter space. Default is:
        `["mini", "mass", "logl", "logt", "logr", "logg", "feh_surf", "afe_surf"]`.

        **Note**: Do not modify this unless you understand the downstream
        dependencies, as many methods expect this specific order.

    verbose : bool, optional
        Whether to output progress messages to stderr during initialization.
        Default is `True`.

    Attributes
    ----------
    filters : numpy.ndarray
        Array of filter names for photometry computation

    predictions : list of str
        List of stellar parameter names for prediction

    feh_grid, afe_grid, loga_grid, eep_grid : numpy.ndarray
        Input parameter grids from MIST isochrone file

    pred_grid : numpy.ndarray
        Stellar parameter predictions organized by grid coordinates

    pred_labels : list of str
        Labels for the prediction parameters from the MIST file

    interpolator : scipy.interpolate.RegularGridInterpolator
        Main interpolation object for stellar parameter prediction

    FNNP : FastNNPredictor
        Neural network predictor for fast SED computation

    Examples
    --------
    Initialize isochrones and generate a stellar population:

    >>> iso = Isochrone()
    >>>
    >>> # Generate photometry for a 1 Gyr old, solar metallicity cluster
    >>> seds, params, params2 = iso.get_seds(
    ...     feh=0.0,           # Solar metallicity
    ...     afe=0.0,           # Solar alpha enhancement
    ...     loga=9.0,          # 1 Gyr age
    ...     av=0.2,            # Some extinction
    ...     rv=3.1,            # Standard extinction law
    ...     dist=1000.0,       # 1 kpc distance
    ...     smf=0.0            # No binaries
    ... )
    >>> print(f"Generated {len(seds)} stellar SEDs")

    Model binary stellar populations:

    >>> # Include 40% secondary mass ratio binaries
    >>> seds, params, params2 = iso.get_seds(
    ...     feh=0.0, afe=0.0, loga=9.5,
    ...     smf=0.4,           # 40% mass ratio secondaries
    ...     av=0.1, dist=800.0
    ... )

    Generate stellar parameters without SEDs:

    >>> params = iso.get_predictions(
    ...     feh=-0.5,          # Metal-poor
    ...     afe=0.3,           # Alpha-enhanced
    ...     loga=10.0,         # 10 Gyr age
    ...     eep=np.arange(200, 500, 10)  # Range of evolutionary phases
    ... )

    Notes
    -----
    The Isochrone class uses a different parameter space than MISTtracks:
    - Input space: [Fe/H], [α/Fe], log(age), EEP
    - MISTtracks uses: initial_mass, EEP, [Fe/H], [α/Fe]

    This makes Isochrone more suitable for stellar population modeling where
    age is the primary parameter, while MISTtracks is better for individual
    stellar parameter inference.

    Binary star modeling is sophisticated, including:
    - Mass ratio effects on evolutionary phase
    - Combined photometry from both components
    - Realistic constraints on binary evolution

    Empirical corrections from MISTtracks can be applied to improve agreement
    with observations, particularly for low-mass stars.
    """

    def __init__(
        self, filters=None, nnfile=None, mistfile=None, predictions=None, verbose=True
    ):

        # Initialize filter list
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters
        if verbose:
            sys.stderr.write("Filters: {}\n".format(filters))

        # Set default file paths relative to package structure
        if nnfile is None:
            package_root = Path(__file__).parent.parent.parent.parent
            nnfile = package_root / "data" / "DATAFILES" / "nnMIST_BC.h5"
        if mistfile is None:
            package_root = Path(__file__).parent.parent.parent.parent
            mistfile = package_root / "data" / "DATAFILES" / "MIST_1.2_iso_vvcrit0.0.h5"

        # Set default predictions
        if predictions is None:
            predictions = [
                "mini",
                "mass",
                "logl",
                "logt",
                "logr",
                "logg",
                "feh_surf",
                "afe_surf",
            ]
        self.predictions = predictions

        if verbose:
            sys.stderr.write("Constructing MIST isochrones...")

        # Load isochrone data from file
        try:
            with h5py.File(mistfile, "r") as f:
                self.feh_grid = f["feh"][:]
                self.afe_grid = f["afe"][:]
                self.loga_grid = f["loga"][:]
                self.eep_grid = f["eep"][:]
                self.pred_grid = f["predictions"][:]
                raw_labels = f["predictions"].attrs["labels"]
                # Normalize labels: HDF5 may store bytes; convert to str
                raw_labels = [
                    l.decode("utf-8") if isinstance(l, (bytes, bytearray)) else str(l)
                    for l in raw_labels
                ]
                # Map long HDF5 names to short internal names expected by
                # the rest of the codebase (matching `seds.py` conventions).
                hdf_to_short = {
                    "initial_mass": "mini",
                    "EEP": "eep",
                    "star_mass": "mass",
                    "log_L": "logl",
                    "log_Teff": "logt",
                    "log_R": "logr",
                    "log_g": "logg",
                    "[Fe/H]": "feh_surf",
                    "[a/Fe]": "afe_surf",
                }
                self.pred_labels = np.array(
                    [hdf_to_short.get(l, l) for l in raw_labels], dtype=object
                )
        except (OSError, KeyError) as e:
            raise RuntimeError(f"Failed to load isochrone data from {mistfile}: {e}")

        # Initialize interpolator
        self.build_interpolator()

        if verbose:
            sys.stderr.write("done!\n")

        # Initialize neural network predictor
        try:
            self.FNNP = FastNNPredictor(filters=filters, nnfile=nnfile, verbose=verbose)
        except Exception as e:
            if verbose:
                sys.stderr.write(f"Warning: Failed to initialize neural network: {e}\n")
            self.FNNP = None

    def build_interpolator(self):
        """
        Construct the RegularGridInterpolator object used to generate isochrones.

        This method processes the loaded MIST isochrone grid data and creates
        a scipy RegularGridInterpolator for fast parameter interpolation. The
        method also handles missing data by interpolating over gaps where possible
        and manages special cases like singular alpha enhancement dimensions.

        The re-structured grid is stored in the following attributes:
        - `grid_dims`: Dimensions of the interpolation grid
        - `xgrid`: Coordinate arrays for each input dimension
        - `ygrid`: Output parameter array organized by grid coordinates
        - `interpolator`: The main interpolation object

        Notes
        -----
        The method includes several important data processing steps:

        1. **Grid setup**: Creates unique coordinate arrays for each input parameter
        2. **Gap filling**: Attempts to linearly interpolate over missing EEP values
        3. **Singular dimension handling**: Pads dimensions with only one value to
           prevent interpolation errors
        4. **Interpolator creation**: Builds the final RegularGridInterpolator object

        Gap filling is attempted for each [Fe/H], [α/Fe], log(age) combination
        where data exists. Linear interpolation is used to fill missing EEP
        values within the range of available data.

        The interpolator uses linear interpolation with NaN fill values for
        out-of-bounds queries and no bounds checking, allowing graceful handling
        of edge cases.
        """

        # Set up coordinate grids for each input parameter
        self.feh_u = np.unique(self.feh_grid)
        self.afe_u = np.unique(self.afe_grid)
        self.loga_u = np.unique(self.loga_grid)
        self.eep_u = np.unique(self.eep_grid)
        self.xgrid = (self.feh_u, self.afe_u, self.loga_u, self.eep_u)

        # Determine grid dimensions
        self.grid_dims = np.array(
            [
                len(self.xgrid[0]),  # feh
                len(self.xgrid[1]),  # afe
                len(self.xgrid[2]),  # loga
                len(self.xgrid[3]),  # eep
                len(self.pred_labels),  # predictions
            ],
            dtype="int",
        )

        # Fill in "holes" in the grid where possible
        # This improves interpolation by linearly interpolating over missing EEP values
        for i in range(len(self.feh_u)):
            for j in range(len(self.afe_u)):
                for k in range(len(self.loga_u)):
                    # Select EEP values where predictions exist (not NaN)
                    sel = np.all(np.isfinite(self.pred_grid[i, j, k]), axis=1)

                    if np.sum(sel) > 1:  # Need at least 2 points for interpolation
                        try:
                            # Linearly interpolate over built-in EEP grid
                            pnew = [
                                np.interp(
                                    self.eep_u,
                                    self.eep_u[sel],
                                    par,
                                    left=np.nan,
                                    right=np.nan,
                                )
                                for par in self.pred_grid[i, j, k, sel].T
                            ]
                            pnew = np.array(pnew).T  # copy and transpose
                            self.pred_grid[i, j, k] = pnew  # assign predictions
                        except Exception:
                            # Fail silently and give up on this grid point
                            pass

        # Handle special case of singular alpha enhancement value
        # This prevents interpolation errors when afe has only one value
        if self.grid_dims[1] == 1:
            # Pad alpha enhancement dimension with small offset
            afe_val = self.xgrid[1][0]
            xgrid = list(self.xgrid)
            xgrid[1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            self.xgrid = tuple(xgrid)

            # Duplicate values in the padded dimension
            self.grid_dims[1] += 1
            ygrid = np.empty(self.grid_dims)
            ygrid[:, 0, :, :, :] = np.array(self.pred_grid[:, 0, :, :, :])  # left
            ygrid[:, 1, :, :, :] = np.array(self.pred_grid[:, 0, :, :, :])  # right
            self.pred_grid = np.array(ygrid)

        # Initialize the main interpolator
        self.interpolator = RegularGridInterpolator(
            self.xgrid,
            self.pred_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def get_predictions(
        self, feh=0.0, afe=0.0, loga=8.5, eep=None, apply_corr=True, corr_params=None
    ):
        """
        Returns interpolated stellar parameter predictions for a given isochrone.

        This method generates stellar parameter predictions along an isochrone of
        fixed metallicity, alpha enhancement, and age. The predictions are
        interpolated from the MIST isochrone grid and can include empirical
        corrections to improve agreement with observations.

        Parameters
        ----------
        feh : float, optional
            Metallicity [Fe/H] defined logarithmically relative to solar
            metallicity. Default is `0.0` (solar).

        afe : float, optional
            Alpha enhancement [α/Fe] defined logarithmically relative to solar
            values. Default is `0.0` (solar).

        loga : float, optional
            Log10(age) where age is in years. Default is `8.5` (≈316 Myr).

        eep : array-like, optional
            Equivalent evolutionary point(s) (EEPs) for which to generate
            predictions. If not provided, the default EEP grid defined on
            initialization will be used. See MIST documentation for EEP
            definitions and ranges.

        apply_corr : bool, optional
            Whether to apply empirical corrections to the effective temperature
            and radius predictions as a function of stellar parameters. These
            corrections improve agreement with observations, particularly for
            low-mass stars. Default is `True`.

        corr_params : tuple, optional
            Parameters controlling the empirical corrections as a tuple of
            (dtdm, drdm, msto_smooth, feh_scale). If not provided, default
            values are used. See MISTtracks documentation for details.

        Returns
        -------
        preds : numpy.ndarray of shape (Neep, Npred)
            Predicted stellar parameters corresponding to the input EEP values.
            The second dimension corresponds to the `predictions` attribute,
            typically including initial mass, current mass, log(L), log(Teff),
            log(R), log(g), surface [Fe/H], and surface [α/Fe].

        Examples
        --------
        Generate predictions for a solar metallicity, 1 Gyr isochrone:

        >>> iso = Isochrone()
        >>> preds = iso.get_predictions(feh=0.0, afe=0.0, loga=9.0)
        >>> initial_masses = preds[:, 0]
        >>> log_teff = preds[:, 3]
        >>> print(f"Mass range: {initial_masses.min():.2f} - {initial_masses.max():.2f} M_sun")

        Generate predictions for specific evolutionary phases:

        >>> import numpy as np
        >>> eep_range = np.arange(200, 500, 50)  # Main sequence to turnoff
        >>> preds = iso.get_predictions(feh=-0.5, afe=0.3, loga=10.0, eep=eep_range)

        Notes
        -----
        The parameter space differs from MISTtracks in that age (loga) is an
        input parameter rather than initial mass. This makes the Isochrone class
        more suitable for stellar population modeling where age is typically
        the primary parameter of interest.

        Empirical corrections follow the same functional form as in MISTtracks,
        but are applied here to the interpolated isochrone predictions rather
        than individual stellar tracks.

        EEP values outside the available grid range will return NaN predictions.
        Typical EEP ranges for main sequence evolution are approximately 200-454,
        with higher values corresponding to post-main sequence phases.
        """

        # Set default EEP grid if not provided
        if eep is None:
            eep = self.eep_u
        eep = np.array(eep, dtype="float")

        # Fill out input labels for all EEP values
        feh_arr = np.full_like(eep, feh)
        afe_arr = np.full_like(eep, afe)
        loga_arr = np.full_like(eep, loga)
        labels = np.c_[feh_arr, afe_arr, loga_arr, eep]

        # Generate predictions using interpolator
        try:
            preds = self.interpolator(labels)
        except Exception as e:
            raise RuntimeError(f"Interpolation failed: {e}")

        # Apply empirical corrections if requested
        if apply_corr and hasattr(self, "_apply_corrections"):
            # Extract stellar mass for corrections
            mini_idx = self.pred_labels.tolist().index("mini")
            mini = preds[:, mini_idx]

            # Apply corrections (method would need to be implemented)
            try:
                corrs = self._apply_corrections(
                    mini=mini, feh=feh_arr, eep=eep, corr_params=corr_params
                )

                # Apply corrections to log(Teff), log(L), and log(g)
                # Note: This assumes specific indices that may need adjustment
                logt_idx = self.pred_labels.tolist().index("logt")
                logl_idx = self.pred_labels.tolist().index("logl")
                logg_idx = self.pred_labels.tolist().index("logg")

                dlogt, dlogr = corrs.T
                preds[:, logt_idx] += dlogt
                preds[:, logl_idx] += 2.0 * dlogr  # L ∝ R^2
                preds[:, logg_idx] -= 2.0 * dlogr  # g ∝ M/R^2

            except Exception as e:
                if hasattr(self, "_verbose") and self._verbose:
                    sys.stderr.write(f"Warning: Corrections failed: {e}\n")

        return preds

    def get_seds(
        self,
        feh=0.0,
        afe=0.0,
        loga=8.5,
        eep=None,
        av=0.0,
        rv=3.3,
        smf=0.0,
        dist=1000.0,
        mini_bound=0.5,
        eep_binary_max=480.0,
        apply_corr=True,
        corr_params=None,
        return_dict=True,
        **kwargs,
    ):
        """
        Generate and return the Spectral Energy Distribution (SED) and
        associated parameters for a stellar population.

        This method generates synthetic photometry for an entire isochrone,
        including support for binary stars, dust extinction, and distance
        effects. It combines the stellar parameter predictions from the
        isochrone interpolation with neural network-based bolometric
        corrections to produce realistic observational quantities.

        Parameters
        ----------
        feh : float, optional
            Metallicity [Fe/H] defined logarithmically relative to solar
            metallicity. Default is `0.0` (solar).

        afe : float, optional
            Alpha enhancement [α/Fe] defined logarithmically relative to solar
            values. Default is `0.0` (solar).

        loga : float, optional
            Log10(age) where age is in years. Default is `8.5` (≈316 Myr).

        eep : array-like, optional
            Equivalent evolutionary point(s) (EEPs) for which to generate SEDs.
            If not provided, the default EEP grid will be used.

        av : float, optional
            Dust attenuation in units of V-band magnitudes A(V). Default is `0.0`.

        rv : float, optional
            Ratio of total to selective extinction R(V) = A(V)/E(B-V), which
            characterizes the dust extinction law. Default is `3.3`.

        smf : float, optional
            Secondary mass fraction for binary stars. If `0.0`, no binaries are
            included. If `0 < smf < 1`, binaries are modeled with secondary
            masses equal to `smf * primary_mass`. If `smf = 1.0`, equal-mass
            binaries are assumed. Default is `0.0`.

        dist : float, optional
            Distance to the stellar population in parsecs. Default is `1000.0`.

        mini_bound : float, optional
            Minimum initial mass (in solar masses) for stars to be included in
            the SED computation. Stars below this mass receive NaN SEDs.
            Default is `0.5`.

        eep_binary_max : float, optional
            Maximum EEP value for which binary companions are considered. This
            typically corresponds to the main sequence turnoff, beyond which
            binary evolution becomes complex. Default is `480.0`.

        apply_corr : bool, optional
            Whether to apply empirical corrections to stellar parameters before
            SED computation. Default is `True`.

        corr_params : tuple, optional
            Parameters controlling empirical corrections. See `get_predictions`
            for details.

        return_dict : bool, optional
            Whether to return stellar parameters as dictionaries (True) or
            arrays (False). Default is `True`.

        **kwargs
            Additional arguments passed to SED computation.

        Returns
        -------
        seds : numpy.ndarray of shape (Neep, Nfilters)
            Predicted SEDs in magnitudes for each EEP value and filter.
            Stars below `mini_bound` or outside the neural network bounds
            will have NaN values.

        params : dict or numpy.ndarray
            Stellar parameters for the primary component. If `return_dict` is
            True, returns a dictionary with parameter names as keys. Otherwise
            returns a 2D array with shape (Npred, Neep).

        params2 : dict or numpy.ndarray
            Stellar parameters for the secondary component (if binaries are
            included). Format matches `params`. If no binaries (`smf=0`),
            this will contain NaN values.

        Examples
        --------
        Generate SEDs for a simple stellar population:

        >>> iso = Isochrone()
        >>> seds, params, params2 = iso.get_seds(
        ...     feh=0.0,           # Solar metallicity
        ...     afe=0.0,           # Solar alpha
        ...     loga=9.0,          # 1 Gyr
        ...     av=0.1,            # Small extinction
        ...     dist=1000.0        # 1 kpc
        ... )
        >>> print(f"Generated SEDs for {len(seds)} stars")
        >>> print(f"Magnitude range: {np.nanmin(seds):.2f} - {np.nanmax(seds):.2f}")

        Model a binary population:

        >>> seds, params, params2 = iso.get_seds(
        ...     feh=-0.5,          # Metal poor
        ...     afe=0.3,           # Alpha enhanced
        ...     loga=10.0,         # 10 Gyr
        ...     smf=0.4,           # 40% mass ratio binaries
        ...     av=0.2,
        ...     dist=5000.0
        ... )

        Extract specific stellar parameters:

        >>> seds, params, params2 = iso.get_seds(feh=0.0, loga=9.5)
        >>> masses = params['mini']
        >>> log_teff = params['logt']
        >>> surface_feh = params['feh_surf']

        Notes
        -----
        The binary modeling includes several sophisticated features:

        1. **Mass ratio constraints**: Secondary masses are `smf * primary_mass`
        2. **Evolutionary phase matching**: Secondaries are placed at EEPs
           corresponding to the same age as primaries
        3. **Combined photometry**: SEDs are magnitude-combined using `add_mag`
        4. **Main sequence restriction**: Binaries only considered for EEP ≤ 480

        The neural network bolometric corrections are applied to convert from
        stellar parameters (log(Teff), log(g), etc.) to observed magnitudes
        in the specified filter set.

        Distance effects are included through the distance modulus:
        μ = 5 log10(d/10 pc), where d is the distance in parsecs.

        Dust extinction follows standard relations with user-specified A(V)
        and R(V) values, applied through the neural network predictions.
        """

        # Check if neural network predictor is available
        if self.FNNP is None:
            raise RuntimeError(
                "Neural network predictor not available for SED computation"
            )

        # Initialize EEP grid if not provided
        if eep is None:
            eep = self.eep_u
        eep = np.array(eep, dtype="float")
        Neep = len(eep)

        # Generate stellar parameter predictions
        try:
            params_arr = self.get_predictions(
                feh=feh,
                afe=afe,
                loga=loga,
                eep=eep,
                apply_corr=apply_corr,
                corr_params=corr_params,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate stellar parameters: {e}")

        # Convert to dictionary format
        params = dict(zip(self.predictions, params_arr.T))

        # Initialize SED array
        seds = np.full((Neep, self.FNNP.NFILT), np.nan)

        # Generate primary component SEDs
        for i in range(Neep):
            if params["mini"][i] >= mini_bound:
                try:
                    seds[i] = self.FNNP.sed(
                        logl=params["logl"][i],
                        logt=params["logt"][i],
                        logg=params["logg"][i],
                        feh_surf=params["feh_surf"][i],
                        afe=params["afe_surf"][i],
                        av=av,
                        rv=rv,
                        dist=dist,
                    )
                except Exception:
                    # SED computation failed, leave as NaN
                    pass

        # Initialize secondary component parameters
        params_arr2 = np.full_like(params_arr, np.nan)
        params2 = dict(zip(self.predictions, params_arr2.T))

        # Add binary companions if requested
        if 0.0 < smf <= 1.0:
            # Extract primary masses
            mini = params["mini"]
            mini2 = mini * smf

            # Find valid primary masses for interpolation
            mini_mask = np.where(np.isfinite(mini))[0]

            if len(mini_mask) > 0:
                # Interpolate secondary EEPs from primary mass-EEP relation
                try:
                    eep2 = np.interp(
                        mini2,
                        mini[mini_mask],
                        eep[mini_mask],
                        left=np.nan,
                        right=np.nan,
                    )
                except Exception:
                    eep2 = np.full_like(eep, np.nan)
            else:
                eep2 = np.full_like(eep, np.nan)

            # Suppress binary effects beyond main sequence turnoff
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore NaN comparisons
                eep2[(eep2 > eep_binary_max) | (eep > eep_binary_max)] = np.nan

            # Generate secondary component predictions
            try:
                params_arr2 = self.get_predictions(
                    feh=feh,
                    afe=afe,
                    loga=loga,
                    eep=eep2,
                    apply_corr=apply_corr,
                    corr_params=corr_params,
                )
                params2 = dict(zip(self.predictions, params_arr2.T))
            except Exception:
                # Secondary predictions failed, keep as NaN
                pass

            # Generate secondary component SEDs
            seds2 = np.full((Neep, self.FNNP.NFILT), np.nan)
            for i in range(Neep):
                if params2["mini"][i] >= mini_bound and np.isfinite(params2["mini"][i]):
                    try:
                        seds2[i] = self.FNNP.sed(
                            logl=params2["logl"][i],
                            logt=params2["logt"][i],
                            logg=params2["logg"][i],
                            feh_surf=params2["feh_surf"][i],
                            afe=params2["afe_surf"][i],
                            av=av,
                            rv=rv,
                            dist=dist,
                        )
                    except Exception:
                        # Secondary SED computation failed, leave as NaN
                        pass

            # Combine primary and secondary SEDs
            seds = add_mag(seds, seds2)

        elif smf == 1.0:
            # Equal mass binaries: simply make everything 2x brighter
            seds[eep <= eep_binary_max] -= 2.5 * np.log10(2.0)
            params2 = deepcopy(params)
            params_arr2 = deepcopy(params_arr.T)

        # Return parameters in requested format
        if not return_dict:
            params = params_arr.T
            params2 = params_arr2

        return seds, params, params2

    def _apply_corrections(self, mini, feh, eep, corr_params=None):
        """
        Apply empirical corrections to stellar parameters.

        This method applies the same empirical corrections as used in the
        MISTtracks class to improve agreement with observations.

        Parameters
        ----------
        mini : array-like
            Initial stellar masses in solar masses
        feh : array-like
            Metallicities [Fe/H]
        eep : array-like
            Equivalent evolutionary points
        corr_params : tuple, optional
            Correction parameters (dtdm, drdm, msto_smooth, feh_scale)

        Returns
        -------
        corrs : numpy.ndarray
            Corrections to [log(Teff), log(R)]
        """
        # Set default correction parameters
        if corr_params is not None:
            dtdm, drdm, msto_smooth, feh_scale = corr_params
        else:
            dtdm, drdm, msto_smooth, feh_scale = 0.09, -0.09, 30.0, 0.5

        # Compute mass offset from solar
        mass_offset = mini - 1.0

        # Safeguard against log10 of negative or zero values
        eps = 1e-10
        temp_arg = np.maximum(1.0 + mass_offset * dtdm, eps)
        radius_arg = np.maximum(1.0 + mass_offset * drdm, eps)

        # Baseline corrections
        dlogt = np.log10(temp_arg)
        dlogr = np.log10(radius_arg)

        # EEP suppression factor (reduces corrections post-main sequence)
        ecorr = 1.0 - 1.0 / (1.0 + np.exp(-(eep - 454.0) / msto_smooth))

        # Metallicity dependence
        fcorr = np.exp(feh_scale * feh)

        # Apply combined effects
        dlogt *= ecorr * fcorr
        dlogr *= ecorr * fcorr

        # Zero out corrections for solar mass and above
        if isinstance(mini, np.ndarray):
            mask = mini >= 1.0
            dlogt[mask] = 0.0
            dlogr[mask] = 0.0
        else:
            if mini >= 1.0:
                dlogt, dlogr = 0.0, 0.0

        return np.c_[dlogt, dlogr]
