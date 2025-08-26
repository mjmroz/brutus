#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for brutus photometry utilities.

These tests cover the photometric conversion functions that are
core to the brutus functionality.
"""

import pytest
import numpy as np
import numpy.testing as npt

# Try importing from the reorganized structure first, fall back to original
try:
    from brutus.utils.photometry import (
        magnitude,
        inv_magnitude,
        luptitude,
        inv_luptitude,
    )
except ImportError:
    # Fall back to original structure during transition
    try:
        from brutus.utils import magnitude, inv_magnitude, luptitude, inv_luptitude
    except ImportError:
        pytest.skip(
            "brutus utils not available - skipping tests", allow_module_level=True
        )


class TestMagnitudeConversions:
    """Test magnitude conversion functions."""

    def test_magnitude_conversion_roundtrip(self):
        """Test that magnitude conversion round-trips correctly."""
        # Test data: flux and errors
        flux = np.array([1.0, 0.5, 0.1, 10.0])
        flux_err = np.array([0.01, 0.02, 0.005, 0.1])
        zeropoints = 1.0

        # Convert to magnitudes and back
        mag, mag_err = magnitude(flux, flux_err, zeropoints)
        flux_recovered, flux_err_recovered = inv_magnitude(mag, mag_err, zeropoints)

        # Should round-trip to original values
        npt.assert_array_almost_equal(flux, flux_recovered, decimal=10)
        npt.assert_array_almost_equal(flux_err, flux_err_recovered, decimal=10)

    def test_magnitude_basic_values(self):
        """Test magnitude conversion with known values."""
        # Flux of 1.0 should give magnitude of 0.0 with zeropoint 1.0
        flux = np.array([1.0])
        flux_err = np.array([0.01])

        mag, mag_err = magnitude(flux, flux_err, zeropoints=1.0)

        npt.assert_array_almost_equal(mag, [0.0], decimal=10)
        # mag_err = 2.5/ln(10) * flux_err/flux = 1.086 * 0.01 = 0.01086
        expected_mag_err = 2.5 / np.log(10) * 0.01
        npt.assert_array_almost_equal(mag_err, [expected_mag_err], decimal=8)

    def test_magnitude_with_zeropoints(self):
        """Test magnitude conversion with different zeropoints."""
        flux = np.array([1.0, 1.0])
        flux_err = np.array([0.01, 0.01])
        zeropoints = np.array([1.0, 10.0])

        mag, mag_err = magnitude(flux, flux_err, zeropoints)

        # First should be 0.0, second should be -2.5*log10(1/10) = 2.5
        expected_mag = np.array([0.0, 2.5])
        npt.assert_array_almost_equal(mag, expected_mag, decimal=10)

    def test_magnitude_shapes(self):
        """Test that magnitude conversion preserves array shapes."""
        # Test 1D arrays
        flux_1d = np.array([1.0, 0.5, 0.1])
        flux_err_1d = np.array([0.01, 0.02, 0.005])

        mag_1d, mag_err_1d = magnitude(flux_1d, flux_err_1d)
        assert mag_1d.shape == flux_1d.shape
        assert mag_err_1d.shape == flux_err_1d.shape

        # Test 2D arrays (multiple objects, multiple filters)
        flux_2d = np.array([[1.0, 0.5], [0.1, 2.0]])
        flux_err_2d = np.array([[0.01, 0.02], [0.005, 0.05]])

        mag_2d, mag_err_2d = magnitude(flux_2d, flux_err_2d)
        assert mag_2d.shape == flux_2d.shape
        assert mag_err_2d.shape == flux_err_2d.shape


class TestLuptitudeConversions:
    """Test luptitude (asinh magnitude) conversion functions."""

    @pytest.mark.parametrize("skynoise", [1.0, 0.1, 10.0])
    def test_luptitude_roundtrip(self, skynoise):
        """Test that luptitude conversion round-trips correctly."""
        flux = np.array([1.0, 0.5, 0.1, 10.0])
        flux_err = np.array([0.01, 0.02, 0.005, 0.1])
        zeropoints = 1.0

        # Convert to luptitudes and back
        lupt, lupt_err = luptitude(flux, flux_err, skynoise, zeropoints)
        flux_recovered, flux_err_recovered = inv_luptitude(
            lupt, lupt_err, skynoise, zeropoints
        )

        # Should round-trip to original values
        npt.assert_array_almost_equal(flux, flux_recovered, decimal=8)
        npt.assert_array_almost_equal(flux_err, flux_err_recovered, decimal=8)

    def test_luptitude_high_snr_limit(self):
        """Test that luptitudes approach magnitudes in high S/N limit."""
        # High flux relative to sky noise - should approach magnitude
        flux = np.array([100.0])  # High S/N
        flux_err = np.array([0.1])
        skynoise = 1.0
        zeropoints = 1.0

        mag, _ = magnitude(flux, flux_err, zeropoints)
        lupt, _ = luptitude(flux, flux_err, skynoise, zeropoints)

        # Should be very close in high S/N limit
        npt.assert_array_almost_equal(lupt, mag, decimal=3)

    def test_luptitude_low_snr_behavior(self):
        """Test luptitude behavior in low S/N regime."""
        # Very low flux - luptitudes should be well-behaved while magnitudes blow up
        flux = np.array([0.01])  # Very faint
        flux_err = np.array([0.005])
        skynoise = 1.0
        zeropoints = 1.0

        lupt, lupt_err = luptitude(flux, flux_err, skynoise, zeropoints)

        # Should give finite, reasonable values
        assert np.all(np.isfinite(lupt))
        assert np.all(np.isfinite(lupt_err))
        assert np.all(lupt_err > 0)


class TestPhotometryEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_flux_handling(self):
        """Test behavior with zero flux (should give infinite magnitude)."""
        flux = np.array([0.0])
        flux_err = np.array([0.01])

        # Suppress expected divide-by-zero warning
        with np.errstate(divide="ignore", invalid="ignore"):
            mag, mag_err = magnitude(flux, flux_err)

        # Should give infinite magnitude
        assert np.isinf(mag[0])
        assert np.isinf(mag_err[0])

    def test_negative_flux_handling(self):
        """Test behavior with negative flux."""
        flux = np.array([-1.0])
        flux_err = np.array([0.1])

        # Suppress expected invalid value warning
        with np.errstate(divide="ignore", invalid="ignore"):
            mag, mag_err = magnitude(flux, flux_err)

        # Magnitude should be NaN (log of negative number)
        assert np.isnan(mag[0])
        # Error might be finite (no log involved in error calculation)
        assert np.isfinite(mag_err[0])

    def test_zero_error_handling(self):
        """Test behavior with zero errors."""
        flux = np.array([1.0])
        flux_err = np.array([0.0])

        mag, mag_err = magnitude(flux, flux_err)

        # Magnitude should be finite, error should be 0
        assert np.isfinite(mag[0])
        assert mag_err[0] == 0.0


class TestPhotometryIntegration:
    """Integration tests for photometry functions."""

    def test_realistic_photometry_pipeline(self):
        """Test a realistic photometry processing pipeline."""
        # Simulate realistic survey data
        np.random.seed(42)

        # True magnitudes for a range of stars
        true_mags = np.random.uniform(14, 24, 100)
        true_fluxes = 10 ** (-0.4 * true_mags)

        # Add realistic errors (magnitude-dependent)
        snr = 100 * 10 ** (-0.4 * (true_mags - 20))  # S/N decreases with magnitude
        flux_errors = true_fluxes / snr

        # Add noise
        observed_fluxes = true_fluxes + np.random.normal(0, flux_errors)

        # Convert to magnitudes
        obs_mags, obs_mag_errs = magnitude(observed_fluxes, flux_errors)

        # Should recover input magnitudes within errors for bright stars
        bright_mask = true_mags < 20
        mag_diff = obs_mags[bright_mask] - true_mags[bright_mask]
        expected_scatter = obs_mag_errs[bright_mask]

        # Most should be within 3-sigma
        outlier_fraction = np.mean(np.abs(mag_diff) > 3 * expected_scatter)
        assert outlier_fraction < 0.05  # Less than 5% outliers
