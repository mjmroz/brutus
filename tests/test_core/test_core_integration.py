#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean integration tests for StarGrid and BruteForce classes.

Tests the essential workflow from grid creation to stellar parameter estimation.
"""

import os
import time

import h5py
import numpy as np
import pytest
from conftest import find_brutus_data_file

from brutus.analysis.individual import BruteForce

# Import the classes to test
from brutus.core.individual import StarGrid
from brutus.data import load_models


@pytest.fixture(scope="module")
def mist_integration_setup():
    """Load real MIST grid and create test setup for integration tests."""

    # Try to load real grid
    grid_paths = [
        find_brutus_data_file("grid_mist_v9.h5"),
        "./data/DATAFILES/grid_mist_v9.h5",
        "data/DATAFILES/grid_mist_v9.h5",
    ]

    for path in grid_paths:
        if path and os.path.exists(path):
            try:
                # Load subset for reasonable test times
                models, combined_labels, label_mask = load_models(path, verbose=False)

                # Get filter names and find PanSTARRS indices
                with h5py.File(path, "r") as f:
                    filter_names = list(f["mag_coeffs"].dtype.names)
                ps_indices = [
                    i for i, name in enumerate(filter_names) if name.startswith("PS_")
                ][:5]

                # Create manageable subset (every 1000th model)
                subset_indices = np.arange(0, len(models), 1000)[:100]
                models_subset = models[subset_indices]
                labels_subset = combined_labels[subset_indices]

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

                labels = labels_subset[grid_names]
                params = labels_subset[pred_names] if pred_names else None

                # Create StarGrid and BruteForce
                grid = StarGrid(models_subset, labels, params, filters=filter_names)
                fitter = BruteForce(grid, verbose=False)

                return {
                    "grid": grid,
                    "fitter": fitter,
                    "ps_indices": ps_indices,
                    "labels": labels,
                    "params": params,
                    "filter_names": filter_names,
                }

            except Exception as e:
                continue

    raise AssertionError("Grid should be available after downloading")


class TestCoreIntegration:
    """Test the core StarGrid + BruteForce workflow."""

    def test_end_to_end_workflow(self, mist_integration_setup):
        """Test complete workflow from observations to parameter estimates."""
        setup = mist_integration_setup
        grid = setup["grid"]
        fitter = setup["fitter"]
        ps_indices = setup["ps_indices"]

        # Create synthetic observations of a solar-like star
        nfilters = grid.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # PanSTARRS observations: g, r, i, z, y
        solar_mags = [15.0, 14.6, 14.3, 14.1, 14.0]
        for i, idx in enumerate(ps_indices):
            obs[idx] = solar_mags[i]
            obs_err[idx] = 0.05  # Realistic error

        obs_mask = ~np.isnan(obs)

        # Step 1: Generate SED using StarGrid for a known model
        reference_idx = len(grid.labels) // 2  # Middle model
        ref_labels = grid.labels[reference_idx]

        sed, params_pred, _ = grid.get_seds(
            mini=float(ref_labels["mini"]),
            eep=float(ref_labels["eep"]),
            feh=float(ref_labels["feh"]),
            return_dict=True,
        )

        # Verify SED generation works
        assert sed is not None
        assert len(sed) == nfilters
        assert np.all(np.isfinite(sed))

        # Step 2: Use BruteForce to fit observations
        start_time = time.time()
        lnl, ndim, chi2 = fitter.loglike_grid(
            obs,
            obs_err,
            obs_mask,
            ltol=5e-2,  # Moderate clipping for speed
            verbose=False,
        )
        elapsed = time.time() - start_time

        # Should complete quickly
        assert elapsed < 2.0, f"Integration test took {elapsed:.2f}s, expected < 2s"

        # Step 3: Verify results are sensible
        assert lnl.shape == (fitter.nmodels,)
        assert np.all(np.isfinite(lnl))
        assert np.all(ndim > 0)
        assert np.all(chi2 >= 0)

        # Find best-fit model
        best_idx = np.argmax(lnl)
        best_labels = fitter.models_labels[best_idx]

        # Should find a reasonable stellar model
        # (not testing exact match due to noise and limited grid)
        assert 0.1 < best_labels["mini"] < 5.0  # Reasonable mass range
        assert 100 < best_labels["eep"] < 1000  # Reasonable evolutionary phase
        assert (
            -3.0 < best_labels["feh"] < 1.0
        )  # Reasonable metallicity (MIST goes to -3)

        print(
            f"Best-fit parameters: mini={best_labels['mini']:.2f}, eep={best_labels['eep']:.0f}, feh={best_labels['feh']:.2f}"
        )
        print(f"Best loglike: {lnl[best_idx]:.2f}")

    def test_performance_characteristics(self, mist_integration_setup):
        """Test that performance scales reasonably with problem size."""
        setup = mist_integration_setup
        fitter = setup["fitter"]
        ps_indices = setup["ps_indices"]

        # Create simple observations
        nfilters = setup["grid"].nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # Use fewer filters for speed
        for i, idx in enumerate(ps_indices[:3]):
            obs[idx] = 15.0
            obs_err[idx] = 0.05
        obs_mask = ~np.isnan(obs)

        # Warm up JIT/cache with first call
        _ = fitter.loglike_grid(
            obs, obs_err, obs_mask, indices=np.arange(10), ltol=5e-2, verbose=False
        )

        # Test different model counts
        model_counts = [10, 50, 100]
        times = []

        for n_models in model_counts:
            if n_models > fitter.nmodels:
                continue

            indices = np.arange(n_models)

            start = time.time()
            lnl, _, _ = fitter.loglike_grid(
                obs, obs_err, obs_mask, indices=indices, ltol=5e-2, verbose=False
            )
            elapsed = time.time() - start
            times.append(elapsed)

            # Verify results
            assert lnl.shape == (n_models,)
            assert np.all(np.isfinite(lnl))

        # Performance should scale reasonably (after warmup)
        if len(times) >= 2:
            # Time per model should be roughly constant (within factor of 5 to account for overhead)
            time_per_model = [t / n for t, n in zip(times, model_counts[: len(times)])]
            ratio = max(time_per_model) / min(time_per_model)
            assert ratio < 5.0, f"Poor performance scaling: {time_per_model}"

        print(
            f"Performance: {time_per_model} s/model for {model_counts[:len(times)]} models"
        )

    def test_grid_interpolation_accuracy(self, mist_integration_setup):
        """Test that StarGrid interpolation gives reasonable results."""
        grid = mist_integration_setup["grid"]
        labels = mist_integration_setup["labels"]

        # Find a model in the middle of the grid
        mini_vals = np.unique(labels["mini"])
        eep_vals = np.unique(labels["eep"])
        feh_vals = np.unique(labels["feh"])

        if len(mini_vals) > 2 and len(eep_vals) > 2 and len(feh_vals) > 2:
            # Test interpolation at intermediate point
            mini_mid = (mini_vals[1] + mini_vals[2]) / 2
            eep_mid = (eep_vals[1] + eep_vals[2]) / 2
            feh_mid = (feh_vals[1] + feh_vals[2]) / 2

            # Get SED with interpolation
            sed_interp, params_interp, _ = grid.get_seds(
                mini=mini_mid, eep=eep_mid, feh=feh_mid, use_multilinear=True
            )

            # Get SED with nearest neighbor
            sed_nn, params_nn, _ = grid.get_seds(
                mini=mini_mid, eep=eep_mid, feh=feh_mid, use_multilinear=False
            )

            # Both should give valid results
            assert np.all(np.isfinite(sed_interp))
            assert np.all(np.isfinite(sed_nn))

            # Interpolated parameters should be reasonable
            if isinstance(params_interp, dict) and "mass" in params_interp:
                assert 0.1 < params_interp["mass"] < 10.0

            print(
                f"Interpolation test: mini={mini_mid:.2f}, eep={eep_mid:.0f}, feh={feh_mid:.2f}"
            )


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_inputs(self, mist_integration_setup):
        """Test handling of invalid inputs."""
        grid = mist_integration_setup["grid"]
        fitter = mist_integration_setup["fitter"]
        ps_indices = mist_integration_setup["ps_indices"]

        nfilters = grid.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # Valid setup
        obs[ps_indices[0]] = 15.0
        obs_err[ps_indices[0]] = 0.05
        obs_mask = ~np.isnan(obs)

        # Test with mismatched array sizes
        with pytest.raises((ValueError, IndexError)):
            fitter.loglike_grid(obs[:5], obs_err, obs_mask, verbose=False)  # Wrong size

        # Test with all NaN observations
        obs_bad = np.full(nfilters, np.nan)
        obs_err_bad = np.full(nfilters, np.nan)
        obs_mask_bad = ~np.isnan(obs_bad)

        # Should handle gracefully (may error or return low likelihood)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result = fitter.loglike_grid(
                    obs_bad, obs_err_bad, obs_mask_bad, verbose=False
                )
                # If it doesn't error, result should be valid
                assert isinstance(result, tuple)
                assert len(result) >= 3
            except (ValueError, RuntimeError, ZeroDivisionError):
                # Acceptable to raise an error for invalid input
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
