#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stellar evolution track interpolation.

This module provides classes for interpolating stellar evolution tracks
from MIST (MESA Isochrones and Stellar Tracks) models, enabling prediction
of stellar parameters as a function of initial mass, metallicity, age,
and evolutionary phase (EEP - Equivalent Evolutionary Point).

The core functionality centers around loading pre-computed stellar evolution
grids and providing fast interpolation for stellar parameter prediction,
with optional empirical corrections to improve agreement with observations.

Classes
-------
MISTtracks : Main track interpolation class
    Provides linear interpolation of MIST stellar evolution tracks across
    the parameter space of initial mass, metallicity, alpha enhancement,
    and equivalent evolutionary point (EEP).

Examples
--------
Basic stellar track interpolation:

>>> from brutus.core.tracks import MISTtracks
>>>
>>> # Initialize with default MIST tracks
>>> tracks = MISTtracks()
>>>
>>> # Get predictions for a solar-mass star at different evolutionary phases
>>> import numpy as np
>>> labels = np.array([[1.0, 350, 0.0, 0.0],   # mini, eep, feh, afe
...                    [1.0, 400, 0.0, 0.0],   # Main sequence turnoff
...                    [1.0, 500, 0.0, 0.0]])  # Red giant branch
>>> predictions = tracks.get_predictions(labels)
>>> print(f"log(age), log(L), log(Teff), log(g): {predictions}")

Advanced usage with empirical corrections:

>>> # Get predictions with custom correction parameters
>>> corr_params = (0.09, -0.09, 30., 0.5)  # dtdm, drdm, msto_smooth, feh_scale
>>> predictions = tracks.get_predictions(labels, apply_corr=True,
...                                      corr_params=corr_params)

Notes
-----
The MIST stellar evolution tracks provide theoretical predictions for stellar
properties across a wide range of masses, metallicities, and evolutionary
phases. This module enables fast interpolation within this grid using
scipy's RegularGridInterpolator.

Empirical corrections can be applied to improve agreement with observations,
particularly for low-mass stars where stellar models may have systematic
uncertainties in effective temperature and radius predictions.

References
----------
.. [1] Choi et al. 2016, "MESA Isochrones and Stellar Tracks (MIST) 0. Methods
       for the Construction of Stellar Isochrones", ApJ, 823, 102
.. [2] Dotter 2016, "MESA Isochrones and Stellar Tracks (MIST) I. Solar-scaled
       Models", ApJS, 222, 8
.. [3] Paxton et al. 2011, "Modules for Experiments in Stellar Astrophysics
       (MESA)", ApJS, 192, 3
"""

import sys
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# Import from parent modules - updated for new structure
# Note: These paths may need adjustment based on final file organization

__all__ = ["MISTtracks"]

# Rename parameters from what is in the MIST HDF5 file.
# This makes it easier to use parameter names as keyword arguments.
rename = {
    "mini": "initial_mass",  # input parameters
    "eep": "EEP",
    "feh": "initial_[Fe/H]",
    "afe": "initial_[a/Fe]",
    "mass": "star_mass",  # outputs
    "feh_surf": "[Fe/H]",
    "afe_surf": "[a/Fe]",
    "loga": "log_age",
    "logt": "log_Teff",
    "logg": "log_g",
    "logl": "log_L",
    "logr": "log_R",
}


class MISTtracks(object):
    """
    An object that linearly interpolates the MIST tracks in EEP, initial mass,
    and metallicity using `~scipy.interpolate.RegularGridInterpolator`.

    This class provides fast interpolation of stellar evolution tracks from the
    MESA Isochrones and Stellar Tracks (MIST) project, enabling prediction of
    stellar parameters (age, luminosity, effective temperature, surface gravity,
    surface metallicity) as a function of initial conditions and evolutionary phase.

    Parameters
    ----------
    mistfile : str, optional
        Path to the HDF5 file containing the MIST tracks. If not provided,
        defaults to the standard MIST v1.2 EEP tracks file. The file should
        contain stellar evolution tracks organized by initial mass, metallicity,
        and alpha enhancement.

    predictions : iterable of str, optional
        The names of stellar parameters to predict at requested locations in
        the label parameter space. Default is
        `["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"]`.
        Available parameters include:
        - 'loga': log10(age in years)
        - 'logl': log10(luminosity in L_sun)
        - 'logt': log10(effective temperature in K)
        - 'logg': log10(surface gravity in cm/s^2)
        - 'feh_surf': surface [Fe/H]
        - 'afe_surf': surface [alpha/Fe]
        - 'mini': initial mass in M_sun
        - 'mass': current stellar mass in M_sun

    ageweight : bool, optional
        Whether to compute the associated d(age)/d(EEP) weights at each
        EEP grid point, which are needed when applying priors in age.
        The weights account for the non-uniform spacing in age along
        evolutionary tracks. Default is `True`.

    verbose : bool, optional
        Whether to output progress messages to `~sys.stderr` during
        initialization. Default is `True`.

    Attributes
    ----------
    labels : list of str
        Input parameter names: ['mini', 'eep', 'feh', 'afe']

    predictions : list of str
        Output parameter names as specified in initialization

    ndim, npred : int
        Number of input dimensions and predicted parameters

    interpolator : scipy.interpolate.RegularGridInterpolator
        The main interpolation object for stellar parameter prediction

    gridpoints : dict
        Unique grid points for each input parameter

    output : numpy.ndarray
        Full stellar parameter array from MIST tracks

    Examples
    --------
    Initialize tracks and predict stellar parameters:

    >>> tracks = MISTtracks()
    >>>
    >>> # Predict parameters for a 1 M_sun star at EEP=350 (main sequence)
    >>> labels = [1.0, 350, 0.0, 0.0]  # mini, eep, feh, afe
    >>> preds = tracks.get_predictions(labels)
    >>> age_yrs = 10**preds[0]  # Convert log(age) to years
    >>> print(f"Age: {age_yrs:.2e} years")

    Batch prediction for multiple stars:

    >>> import numpy as np
    >>> labels = np.array([[0.5, 350, -1.0, 0.3],   # Low-mass, metal-poor
    ...                    [1.0, 454, 0.0, 0.0],    # Solar at turnoff
    ...                    [2.0, 500, 0.2, 0.0]])   # Higher mass, metal-rich
    >>> preds = tracks.get_predictions(labels)
    >>> print(f"log(ages): {preds[:, 0]}")

    Notes
    -----
    The MIST tracks use Equivalent Evolutionary Points (EEPs) to parameterize
    stellar evolution. Key EEP ranges:
    - EEP < 202: Pre-main sequence
    - EEP 202-454: Main sequence (202=ZAMS, 454=TAMS)
    - EEP 454-605: Subgiant branch
    - EEP 605-631: Red giant branch base
    - EEP > 631: Advanced evolutionary phases

    Empirical corrections are available to adjust theoretical predictions
    based on observational constraints, particularly important for low-mass
    stars where model uncertainties are largest.
    """

    def __init__(
        self,
        mistfile=None,
        predictions=["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"],
        ageweight=True,
        verbose=True,
    ):

        # Define input parameter labels
        labels = ["mini", "eep", "feh", "afe"]

        # Initialize values.
        self.labels = list(np.array(labels))
        self.predictions = list(np.array(predictions))
        self.ndim, self.npred = len(self.labels), len(self.predictions)
        self.null = np.zeros(self.npred) + np.nan

        # Set label references for fast indexing
        self.mini_idx = np.where(np.array(self.labels) == "mini")[0][0]
        self.eep_idx = np.where(np.array(self.labels) == "eep")[0][0]
        self.feh_idx = np.where(np.array(self.labels) == "feh")[0][0]
        self.logt_idx = np.where(np.array(self.predictions) == "logt")[0][0]
        self.logl_idx = np.where(np.array(self.predictions) == "logl")[0][0]
        self.logg_idx = np.where(np.array(self.predictions) == "logg")[0][0]

        # Import MIST grid.
        if mistfile is None:
            package_root = Path(
                __file__
            ).parent.parent.parent.parent  # Get the package root directory
            mistfile = package_root / "data" / "DATAFILES" / "MIST_1.2_EEPtrk.h5"
        self.mistfile = mistfile

        if verbose:
            sys.stderr.write(f"Loading MIST tracks from {mistfile}...\n")

        with h5py.File(self.mistfile, "r") as misth5:
            self.make_lib(misth5, verbose=verbose)
        self.lib_as_grid()

        # Construct age weights if requested
        self._ageidx = self.predictions.index("loga")
        if ageweight:
            self.add_age_weights(verbose=verbose)

        # Construct interpolation grid
        self.build_interpolator()

        if verbose:
            sys.stderr.write("done!\n")

    def make_lib(self, misth5, verbose=True):
        """
        Convert the HDF5 input to ndarrays for labels and outputs.

        This method reads the MIST stellar evolution tracks from the HDF5 file
        and converts them into numpy arrays stored as `libparams` and `output`
        attributes. The method handles different possible formats and missing
        data gracefully.

        Parameters
        ----------
        misth5 : h5py.File
            Open HDF5 file handle to the MIST models containing stellar
            evolution tracks organized by metallicity.

        verbose : bool, optional
            Whether to print progress messages during library construction.
            Default is `True`.

        Notes
        -----
        The MIST tracks are organized hierarchically in the HDF5 file by
        metallicity. This method concatenates all tracks across metallicities
        into unified arrays for interpolation.

        If alpha enhancement data ([alpha/Fe]) is missing from the tracks, it
        is copied from [Fe/H] data and then zeroed out as a fallback.
        """

        if verbose:
            sys.stderr.write("Constructing MIST library...\n")

        # Extract input parameters (mini, eep, feh, afe)
        if verbose:
            sys.stderr.write(f"  Reading input parameters: {self.labels}\n")
        cols = [rename[p] for p in self.labels]
        self.libparams = np.concatenate(
            [np.array(misth5[z])[cols] for z in misth5["index"]]
        )
        self.libparams.dtype.names = tuple(self.labels)

        # Check if alpha enhancement column is available
        cols = [rename[p] for p in self.predictions]
        afe_col = rename["afe_surf"]
        afe_available = True
        afe_surf_idx = None

        try:
            # Test access to alpha enhancement column in first metallicity group
            first_z = list(misth5["index"])[0]
            _ = misth5[first_z][afe_col]
        except (KeyError, ValueError):
            afe_available = False
            # Find which prediction corresponds to alpha enhancement
            for i, pred in enumerate(self.predictions):
                if pred == "afe_surf":
                    afe_surf_idx = i
                    break
            if verbose:
                sys.stderr.write(
                    "  Note: [alpha/Fe] column not found, will copy from [Fe/H] and zero\n"
                )

        # Create list of columns to actually read from HDF5
        cols_to_read = []
        read_to_pred_mapping = []  # Maps read index to prediction index

        for pred_idx, col in enumerate(cols):
            if not afe_available and col == afe_col:
                # Skip reading alpha enhancement, we'll copy from [Fe/H] later
                continue
            else:
                cols_to_read.append(col)
                read_to_pred_mapping.append(pred_idx)

        # Read output parameters efficiently (single pass, no duplicate reads)
        if verbose:
            sys.stderr.write(f"  Reading output columns: {cols_to_read}\n")

        output_data = [
            np.concatenate([misth5[z][p] for z in misth5["index"]])
            for p in cols_to_read
        ]

        # Create full output array and fill it
        self.output = np.empty((len(output_data[0]), len(self.predictions)), dtype="f8")

        # Fill in the data we actually read
        for read_idx, pred_idx in enumerate(read_to_pred_mapping):
            self.output[:, pred_idx] = output_data[read_idx]

        # Handle missing alpha enhancement by copying from [Fe/H] and zeroing
        if not afe_available and afe_surf_idx is not None:
            feh_surf_idx = None
            for i, pred in enumerate(self.predictions):
                if pred == "feh_surf":
                    feh_surf_idx = i
                    break

            if feh_surf_idx is not None:
                # Copy [Fe/H] data to [α/Fe] position
                self.output[:, afe_surf_idx] = self.output[:, feh_surf_idx]
                # Zero out as in original implementation
                self.output[:, afe_surf_idx] *= 0.0
            else:
                # Fallback: just set to zero
                self.output[:, afe_surf_idx] = 0.0

        if verbose:
            sys.stderr.write(
                f"  Final array shapes: libparams={self.libparams.shape}, "
                f"output={self.output.shape}\n"
            )
            sys.stderr.write("done!\n")

    def lib_as_grid(self):
        """
        Convert the library parameters to pixel indices in each dimension.

        This method processes the stellar track library to create a regular
        grid structure suitable for interpolation. The unique grid points
        and spacing information are stored for later use in building the
        interpolator.

        Attributes Created
        ------------------
        gridpoints : dict
            Dictionary containing unique grid points for each input parameter

        binwidths : dict
            Dictionary containing grid spacing information for each parameter

        X : numpy.ndarray
            Array of grid indices corresponding to each track point

        mini_bound : float
            Minimum initial mass value in the grid
        """

        # Get the unique grid points in each parameter dimension
        self.gridpoints = {}
        self.binwidths = {}
        for p in self.labels:
            self.gridpoints[p] = np.unique(self.libparams[p])
            self.binwidths[p] = np.diff(self.gridpoints[p])

        # Digitize the library parameters to get grid indices
        X = np.array(
            [
                np.digitize(self.libparams[p], bins=self.gridpoints[p], right=True)
                for p in self.labels
            ]
        )
        self.X = X.T

        # Store minimum mass bound for later use
        self.mini_bound = self.gridpoints["mini"].min()

    def add_age_weights(self, verbose=True):
        """
        Compute the age gradient d(age)/d(EEP) over the EEP grid.

        This method calculates age weights that account for the non-uniform
        spacing in age along stellar evolution tracks. The weights are essential
        when applying age priors in stellar parameter inference, as they correct
        for the fact that stars spend different amounts of time at different
        evolutionary phases.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages during weight computation.
            Default is `True`.

        Notes
        -----
        Age weights are computed as the numerical gradient of age with respect
        to EEP for each track (constant mass, metallicity, alpha enhancement).
        Results are appended to the `output` array and 'agewt' is added to
        the `predictions` list.

        The age weights represent how much time a star spends in each EEP bin,
        which is crucial for proper statistical treatment when fitting stellar
        populations.
        """

        # Check that we indeed have log(age) as a prediction parameter
        assert "loga" in self.predictions

        # Initialize age weights array
        age_ind = self._ageidx
        ageweights = np.zeros(len(self.libparams))

        # Loop over all tracks (unique combinations of mini, feh, afe)
        for i, m in enumerate(self.gridpoints["mini"]):
            for j, z in enumerate(self.gridpoints["feh"]):
                for k, a in enumerate(self.gridpoints["afe"]):
                    if verbose:
                        sys.stderr.write(
                            "\rComputing age weights for track "
                            "(mini, feh, afe) = "
                            "({0:.3f}, {1:.3f}, {2:.3f})          ".format(m, z, a)
                        )
                        sys.stderr.flush()

                    # Get indices for this specific track
                    inds = (
                        (self.libparams["mini"] == m)
                        & (self.libparams["feh"] == z)
                        & (self.libparams["afe"] == a)
                    )

                    # Compute age weights as gradient of linear age
                    # (assumes tracks are ordered by EEP, hence by age)
                    try:
                        agewts = np.gradient(10 ** self.output[inds, age_ind])
                        ageweights[inds] = agewts
                    except:
                        # If gradient computation fails, skip this track
                        pass

        # Append age weights to outputs
        self.output = np.hstack([self.output, ageweights[:, None]])
        self.predictions += ["agewt"]

        if verbose:
            sys.stderr.write("\n")

    def build_interpolator(self):
        """
        Construct the `~scipy.interpolate.RegularGridInterpolator` object
        used to generate fast predictions.

        This method creates the main interpolation machinery by organizing
        the stellar track data into a regular grid and initializing scipy's
        RegularGridInterpolator. Special handling is included for cases with
        singular alpha enhancement values.

        Attributes Created
        ------------------
        grid_dims : numpy.ndarray
            Dimensions of the interpolation grid

        xgrid : tuple
            Tuple of grid point arrays for each input dimension

        ygrid : numpy.ndarray
            Multi-dimensional array of stellar parameters organized on the grid

        interpolator : scipy.interpolate.RegularGridInterpolator
            The main interpolation object for stellar parameter prediction

        Notes
        -----
        The method handles the special case where alpha enhancement has only
        a single value by artificially padding the grid with a tiny offset
        to avoid interpolation issues.
        """

        # Set up grid dimensions
        self.grid_dims = np.append(
            [len(self.gridpoints[p]) for p in self.labels], self.output.shape[-1]
        )
        self.xgrid = tuple([self.gridpoints[l] for l in self.labels])

        # Initialize output grid with NaN values
        self.ygrid = np.zeros(self.grid_dims) + np.nan

        # Fill in the grid with stellar track data
        for x, y in zip(self.X, self.output):
            self.ygrid[tuple(x)] = y

        # Handle special case of singular alpha enhancement value
        # This prevents interpolation errors when afe has only one value
        if self.grid_dims[-2] == 1:
            # Pad alpha enhancement dimension with small offset
            afe_val = self.xgrid[-1][0]
            xgrid = list(self.xgrid)
            xgrid[-1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            self.xgrid = tuple(xgrid)

            # Duplicate values in the padded dimension
            self.grid_dims[-2] += 1
            ygrid = np.empty(self.grid_dims)
            ygrid[:, :, :, 0, :] = np.array(self.ygrid[:, :, :, 0, :])  # left
            ygrid[:, :, :, 1, :] = np.array(self.ygrid[:, :, :, 0, :])  # right
            self.ygrid = np.array(ygrid)

        # Initialize the main interpolator
        self.interpolator = RegularGridInterpolator(
            self.xgrid,
            self.ygrid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def get_predictions(self, labels, apply_corr=True, corr_params=None):
        """
        Returns interpolated predictions for the input set of labels.

        This is the main method for obtaining stellar parameter predictions
        from the MIST tracks. It interpolates within the pre-computed grid
        and optionally applies empirical corrections to improve agreement
        with observations.

        Parameters
        ----------
        labels : array-like of shape (Nlabel,) or (Nobj, Nlabel)
            Input stellar parameters for which to generate predictions.
            Should contain [mini, eep, feh, afe] in that order:
            - mini: Initial mass in solar masses
            - eep: Equivalent evolutionary point (see MIST documentation)
            - feh: Metallicity [Fe/H] in logarithmic solar units
            - afe: Alpha enhancement [alpha/Fe] in logarithmic solar units

        apply_corr : bool, optional
            Whether to apply empirical corrections to the effective
            temperature and radius as a function of the input labels.
            These corrections improve agreement with observations for
            low-mass stars. Default is `True`.

        corr_params : tuple, optional
            Parameters controlling the empirical corrections as a tuple of
            (dtdm, drdm, msto_smooth, feh_scale). If not provided, default
            values are used. See `get_corrections` for detailed descriptions.

        Returns
        -------
        preds : numpy.ndarray of shape (Npred,) or (Nobj, Npred)
            Predicted stellar parameters corresponding to the input labels.
            The order matches the `predictions` attribute. Typical outputs
            include [log(age), log(L), log(Teff), log(g), [Fe/H]_surf, [alpha/Fe]_surf].

        Examples
        --------
        Single star prediction:

        >>> tracks = MISTtracks()
        >>> labels = [1.0, 350, 0.0, 0.0]  # Solar mass star on main sequence
        >>> preds = tracks.get_predictions(labels)
        >>> log_age, log_L, log_Teff, log_g = preds[:4]
        >>> print(f"Age: {10**log_age:.1e} yr, Teff: {10**log_Teff:.0f} K")

        Multiple star prediction:

        >>> import numpy as np
        >>> labels = np.array([[0.8, 350, -0.5, 0.2],   # Metal-poor dwarf
        ...                    [1.2, 454, 0.0, 0.0],    # Solar at turnoff
        ...                    [2.0, 500, 0.3, 0.0]])   # Massive metal-rich
        >>> preds = tracks.get_predictions(labels)
        >>> print(f"log(Teff): {preds[:, 2]}")

        Notes
        -----
        Empirical corrections are applied to log(Teff) and log(L) to account
        for known systematic differences between MIST models and observations,
        particularly for M dwarf stars. The corrections preserve surface
        gravity through the relation log(g) ∝ log(M) - 2*log(R).
        """

        labels = np.array(labels)
        ndim = labels.ndim

        # Perform interpolation
        if ndim == 1:
            preds = self.interpolator(labels)[0]
        elif ndim == 2:
            preds = self.interpolator(labels)
        else:
            raise ValueError("Input `labels` must be 1-D or 2-D array.")

        # Apply empirical corrections if requested
        if apply_corr:
            corrs = self.get_corrections(labels, corr_params=corr_params)
            if ndim == 1:
                dlogt, dlogr = corrs
                preds[self.logt_idx] += dlogt
                preds[self.logl_idx] += 2.0 * dlogr  # L ∝ R^2
                preds[self.logg_idx] -= 2.0 * dlogr  # g ∝ M/R^2
            elif ndim == 2:
                dlogt, dlogr = corrs.T
                preds[:, self.logt_idx] += dlogt
                preds[:, self.logl_idx] += 2.0 * dlogr
                preds[:, self.logg_idx] -= 2.0 * dlogr

        return preds

    def get_corrections(self, labels, corr_params=None):
        """
        Returns empirical corrections for stellar parameters.

        This method computes empirical corrections to theoretical stellar
        parameters to improve agreement with observations. The corrections
        are particularly important for low-mass stars where stellar models
        have systematic uncertainties.

        Parameters
        ----------
        labels : array-like of shape (Nlabel,) or (Nobj, Nlabel)
            Input stellar parameters [mini, eep, feh, afe].

        corr_params : tuple, optional
            Tuple of (dtdm, drdm, msto_smooth, feh_scale) controlling the
            correction magnitude and functional form:

            - dtdm : float
                Temperature correction coefficient per unit mass offset from 1 M_sun
            - drdm : float
                Radius correction coefficient per unit mass offset from 1 M_sun
            - msto_smooth : float
                EEP scale for exponential decay around main sequence turnoff
            - feh_scale : float
                Metallicity dependence scale factor

            Default values are (0.09, -0.09, 30., 0.5) if not provided.

        Returns
        -------
        corrs : numpy.ndarray of shape (2,) or (Nobj, 2)
            Corrections to [log(Teff), log(R)] for the input parameters.

        Notes
        -----
        The corrections have the functional form:

        - δlog(Teff) = log(1 + (M - 1) * dtdm) * f_eep * f_feh
        - δlog(R) = log(1 + (M - 1) * drdm) * f_eep * f_feh

        where f_eep suppresses corrections on the main sequence above the
        turnoff and f_feh provides metallicity dependence.

        Corrections are designed to be zero for solar-mass stars and to
        not affect post-main sequence evolution significantly.
        """

        # Extract relevant parameters
        labels = np.array(labels)
        ndim = labels.ndim
        mini, eep, feh = labels[[self.mini_idx, self.eep_idx, self.feh_idx]]

        # Set correction parameters
        if corr_params is not None:
            dtdm, drdm, msto_smooth, feh_scale = corr_params
        else:
            dtdm, drdm, msto_smooth, feh_scale = 0.09, -0.09, 30.0, 0.5

        # Compute baseline corrections to log(Teff) and log(R)
        dlogt = np.log10(1.0 + (mini - 1.0) * dtdm)  # Temperature correction
        dlogr = np.log10(1.0 + (mini - 1.0) * drdm)  # Radius correction

        # EEP suppression: reduce corrections post-main sequence
        # The sigmoid function transitions around EEP=454 (main sequence turnoff)
        ecorr = 1 - 1.0 / (1.0 + np.exp(-(eep - 454) / msto_smooth))

        # Metallicity dependence: enhance corrections at low metallicity
        fcorr = np.exp(feh_scale * feh)

        # Apply combined effects
        dlogt *= ecorr * fcorr
        dlogr *= ecorr * fcorr

        # Format output based on input dimensionality
        if ndim == 1:
            if mini >= 1.0:
                # No corrections for solar mass and above
                corrs = np.array([0.0, 0.0])
            else:
                corrs = np.array([dlogt, dlogr])
        elif ndim == 2:
            # Zero out corrections for M >= 1 M_sun
            dlogt[mini >= 1.0] = 0.0
            dlogr[mini >= 1.0] = 0.0
            corrs = np.c_[dlogt, dlogr]
        else:
            raise ValueError("Input `labels` must be 1-D or 2-D array.")

        return corrs
