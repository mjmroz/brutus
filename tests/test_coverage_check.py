#!/usr/bin/env python
"""
Quick coverage test for key refactored functionality.
"""

import pytest
import numpy as np
import os
import h5py


def test_stargrid_core_functions():
    """Test core StarGrid functions to ensure coverage."""
    from src.brutus.core.individual import StarGrid
    from src.brutus.data import load_models

    # Load a small subset for coverage testing
    grid_paths = [
        "/mnt/d/Dropbox/GitHub/brutus/data/DATAFILES/grid_mist_v9.h5",
        "./data/DATAFILES/grid_mist_v9.h5",
    ]

    for path in grid_paths:
        if os.path.exists(path):
            models, combined_labels, label_mask = load_models(path, verbose=False)

            # Small subset for fast coverage testing
            subset = slice(0, 10)
            models_small = models[subset]
            labels_small = combined_labels[subset]

            # Separate grid parameters from predictions
            grid_names = [
                name
                for name, is_grid in zip(combined_labels.dtype.names, label_mask[0])
                if is_grid
            ]
            pred_names = [
                name
                for name, is_grid in zip(combined_labels.dtype.names, label_mask[0])
                if not is_grid
            ]

            labels = labels_small[grid_names]
            params = labels_small[pred_names] if pred_names else None

            # Create StarGrid
            grid = StarGrid(models_small, labels, params)

            # Exercise key methods for coverage

            # 1. Test get_seds method
            first_model = labels[0]
            sed, params_out, params2 = grid.get_seds(
                mini=float(first_model["mini"]),
                eep=float(first_model["eep"]),
                feh=float(first_model["feh"]),
            )
            assert sed is not None

            # 2. Test interpolation
            sed2, _, _ = grid.get_seds(
                mini=float(first_model["mini"]) + 0.1,
                eep=float(first_model["eep"]) + 10,
                feh=float(first_model["feh"]) + 0.1,
                use_multilinear=True,
            )
            assert sed2 is not None

            # 3. Test nearest neighbor fallback
            sed3, _, _ = grid.get_seds(
                mini=float(first_model["mini"]),
                eep=float(first_model["eep"]),
                feh=float(first_model["feh"]),
                use_multilinear=False,
            )
            assert sed3 is not None

            # 4. Test extinction
            sed4, _, _ = grid.get_seds(
                mini=float(first_model["mini"]),
                eep=float(first_model["eep"]),
                feh=float(first_model["feh"]),
                av=0.5,
                rv=3.1,
            )
            assert sed4 is not None

            # 5. Test distance scaling
            sed5, _, _ = grid.get_seds(
                mini=float(first_model["mini"]),
                eep=float(first_model["eep"]),
                feh=float(first_model["feh"]),
                dist=500.0,
            )
            assert sed5 is not None

            # 6. Test flux return
            sed6, _, _ = grid.get_seds(
                mini=float(first_model["mini"]),
                eep=float(first_model["eep"]),
                feh=float(first_model["feh"]),
                return_flux=True,
            )
            assert sed6 is not None

            return  # Success

    pytest.skip("MIST grid not available")


def test_bruteforce_core_functions():
    """Test core BruteForce functions to ensure coverage."""
    from src.brutus.core.individual import StarGrid
    from src.brutus.analysis.individual import BruteForce
    from src.brutus.data import load_models

    # Load small subset
    grid_paths = [
        "/mnt/d/Dropbox/GitHub/brutus/data/DATAFILES/grid_mist_v9.h5",
        "./data/DATAFILES/grid_mist_v9.h5",
    ]

    for path in grid_paths:
        if os.path.exists(path):
            models, combined_labels, label_mask = load_models(path, verbose=False)

            # Get filter names and find first few filters
            with h5py.File(path, "r") as f:
                filter_names = list(f["mag_coeffs"].dtype.names)

            # Small subset for fast coverage testing
            subset = slice(0, 20)
            models_small = models[subset]
            labels_small = combined_labels[subset]

            grid_names = [
                name
                for name, is_grid in zip(combined_labels.dtype.names, label_mask[0])
                if is_grid
            ]
            pred_names = [
                name
                for name, is_grid in zip(combined_labels.dtype.names, label_mask[0])
                if not is_grid
            ]

            labels = labels_small[grid_names]
            params = labels_small[pred_names] if pred_names else None

            # Create StarGrid and BruteForce
            grid = StarGrid(models_small, labels, params, filters=filter_names)
            fitter = BruteForce(grid, verbose=False)

            # Exercise key methods for coverage

            # Create simple observations
            nfilters = grid.nfilters
            obs = np.full(nfilters, np.nan)
            obs_err = np.full(nfilters, np.nan)

            # Use first 3 filters
            obs[:3] = [15.0, 14.8, 14.6]
            obs_err[:3] = 0.05
            obs_mask = ~np.isnan(obs)

            # 1. Test loglike_grid with default parameters
            lnl1, ndim1, chi2_1 = fitter.loglike_grid(
                obs, obs_err, obs_mask, verbose=False
            )
            assert lnl1 is not None
            assert len(lnl1) == fitter.nmodels

            # 2. Test with subset of indices
            indices = np.arange(5)
            lnl2, ndim2, chi2_2 = fitter.loglike_grid(
                obs, obs_err, obs_mask, indices=indices, verbose=False
            )
            assert len(lnl2) == 5

            # 3. Test with different clipping threshold
            lnl3, ndim3, chi2_3 = fitter.loglike_grid(
                obs, obs_err, obs_mask, ltol=1e-1, verbose=False  # More aggressive
            )
            assert lnl3 is not None

            # 4. Test with return_vals=True
            result = fitter.loglike_grid(
                obs, obs_err, obs_mask, indices=indices, return_vals=True, verbose=False
            )
            assert isinstance(result, tuple)
            assert len(result) == 7  # lnl, ndim, chi2, scale, av, rv, icov_sar

            # 5. Test get_sed_grid method
            seds = fitter.get_sed_grid(indices=indices)
            assert seds is not None

            # 6. Test properties
            assert fitter.nmodels > 0
            assert fitter.nfilters > 0

            return  # Success

    pytest.skip("MIST grid not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
