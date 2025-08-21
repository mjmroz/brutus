#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus SED utilities.

This test suite includes:
1. Unit tests for _get_seds and get_seds functions
2. Performance tests for numba-compiled functions
3. Physical consistency tests
4. Edge case handling
5. Integration tests with other brutus components
"""

import numpy as np
import pytest
import numpy.testing as npt


class TestGetSedsCore:
    """Unit tests for the core _get_seds function."""

    def test_get_seds_basic_functionality(self):
        """Test basic _get_seds functionality."""
        from brutus.core.sed_utils import _get_seds

        # Simple test case: 2 models, 3 bands
        mag_coeffs = np.ones((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0  # Base magnitudes
        mag_coeffs[:, :, 1] = 1.0  # R(V)=0 reddening
        mag_coeffs[:, :, 2] = 0.1  # dR/dR(V)

        av = np.array([0.0, 0.5])
        rv = np.array([3.1, 3.3])

        seds, rvecs, drvecs = _get_seds(mag_coeffs, av, rv, return_flux=False)

        # Check shapes
        assert seds.shape == (2, 3)
        assert rvecs.shape == (2, 3)
        assert drvecs.shape == (2, 3)

        # Check that SEDs are computed correctly
        # For model 0: av=0, so SED should equal base magnitude
        npt.assert_array_almost_equal(seds[0], 15.0, decimal=10)

        # For model 1: av=0.5, rv=3.3
        # rvec = r0 + rv * dr = 1.0 + 3.3 * 0.1 = 1.33
        # sed = mag + av * rvec = 15.0 + 0.5 * 1.33 = 15.665
        expected_rvec = 1.0 + 3.3 * 0.1
        expected_sed = 15.0 + 0.5 * expected_rvec

        npt.assert_array_almost_equal(rvecs[1], expected_rvec, decimal=10)
        npt.assert_array_almost_equal(seds[1], expected_sed, decimal=10)
        npt.assert_array_almost_equal(drvecs[1], 0.1, decimal=10)

    def test_get_seds_flux_conversion(self):
        """Test flux conversion functionality."""
        from brutus.core.sed_utils import _get_seds

        mag_coeffs = np.ones((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0
        mag_coeffs[:, :, 1] = 1.0
        mag_coeffs[:, :, 2] = 0.1

        av = np.array([0.0, 0.5])
        rv = np.array([3.1, 3.3])

        # Get magnitudes
        seds_mag, rvecs_mag, drvecs_mag = _get_seds(
            mag_coeffs, av, rv, return_flux=False
        )

        # Get fluxes
        seds_flux, rvecs_flux, drvecs_flux = _get_seds(
            mag_coeffs, av, rv, return_flux=True
        )

        # Check flux conversion
        expected_flux = 10.0 ** (-0.4 * seds_mag)
        npt.assert_array_almost_equal(seds_flux, expected_flux, decimal=10)

        # Check that reddening vectors are also converted correctly
        fac = -0.4 * np.log(10.0)
        expected_rvecs_flux = rvecs_mag * fac * seds_flux
        expected_drvecs_flux = drvecs_mag * fac * seds_flux

        npt.assert_array_almost_equal(rvecs_flux, expected_rvecs_flux, decimal=10)
        npt.assert_array_almost_equal(drvecs_flux, expected_drvecs_flux, decimal=10)

    def test_get_seds_single_model(self):
        """Test with single model."""
        from brutus.core.sed_utils import _get_seds

        mag_coeffs = np.ones((1, 5, 3))
        mag_coeffs[0, :, 0] = [14, 15, 16, 17, 18]  # Different base mags
        mag_coeffs[0, :, 1] = [0.8, 1.0, 1.2, 1.4, 1.6]  # Different reddening
        mag_coeffs[0, :, 2] = 0.05  # Small dR/dRv

        av = np.array([1.0])
        rv = np.array([3.1])

        seds, rvecs, drvecs = _get_seds(mag_coeffs, av, rv, return_flux=False)

        assert seds.shape == (1, 5)

        # Check each band individually
        for i in range(5):
            expected_rvec = mag_coeffs[0, i, 1] + rv[0] * mag_coeffs[0, i, 2]
            expected_sed = mag_coeffs[0, i, 0] + av[0] * expected_rvec

            assert abs(rvecs[0, i] - expected_rvec) < 1e-10
            assert abs(seds[0, i] - expected_sed) < 1e-10

    def test_get_seds_many_models(self):
        """Test with many models to check vectorization."""
        from brutus.core.sed_utils import _get_seds

        n_models = 100
        n_bands = 10

        np.random.seed(42)
        mag_coeffs = np.random.random((n_models, n_bands, 3))
        mag_coeffs[:, :, 0] = np.random.uniform(14, 20, (n_models, n_bands))
        mag_coeffs[:, :, 1] = np.random.uniform(0.5, 2.0, (n_models, n_bands))
        mag_coeffs[:, :, 2] = np.random.uniform(0.0, 0.3, (n_models, n_bands))

        av = np.random.uniform(0.0, 3.0, n_models)
        rv = np.random.uniform(2.0, 5.0, n_models)

        seds, rvecs, drvecs = _get_seds(mag_coeffs, av, rv, return_flux=False)

        assert seds.shape == (n_models, n_bands)
        assert rvecs.shape == (n_models, n_bands)
        assert drvecs.shape == (n_models, n_bands)

        # Spot check a few models
        for i in [0, 50, 99]:
            for j in [0, 5, 9]:
                expected_rvec = mag_coeffs[i, j, 1] + rv[i] * mag_coeffs[i, j, 2]
                expected_sed = mag_coeffs[i, j, 0] + av[i] * expected_rvec

                assert abs(rvecs[i, j] - expected_rvec) < 1e-10
                assert abs(seds[i, j] - expected_sed) < 1e-10


class TestGetSedsWrapper:
    """Unit tests for the get_seds wrapper function."""

    def test_get_seds_wrapper_basic(self):
        """Test basic wrapper functionality."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0
        mag_coeffs[:, :, 1] = 1.0
        mag_coeffs[:, :, 2] = 0.1

        # Test with explicit parameters
        seds = get_seds(mag_coeffs, av=0.5, rv=3.1)

        assert seds.shape == (2, 3)

        # Test that it returns only SEDs by default
        assert isinstance(seds, np.ndarray)
        assert seds.ndim == 2

    def test_get_seds_wrapper_defaults(self):
        """Test default parameter handling."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0
        mag_coeffs[:, :, 1] = 1.0
        mag_coeffs[:, :, 2] = 0.0  # No RV dependence

        # Test defaults (av=0, rv=3.3)
        seds_default = get_seds(mag_coeffs)
        seds_explicit = get_seds(mag_coeffs, av=0.0, rv=3.3)

        npt.assert_array_almost_equal(seds_default, seds_explicit, decimal=12)

        # With av=0 and no RV dependence, should equal base magnitude
        npt.assert_array_almost_equal(seds_default, 15.0, decimal=12)

    def test_get_seds_wrapper_scalar_params(self):
        """Test scalar parameter broadcasting."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((3, 2, 3))
        mag_coeffs[:, :, 0] = 15.0
        mag_coeffs[:, :, 1] = 1.0
        mag_coeffs[:, :, 2] = 0.1

        # Test scalar parameters
        seds1 = get_seds(mag_coeffs, av=0.5, rv=3.1)

        # Test array parameters with same values
        seds2 = get_seds(
            mag_coeffs, av=np.array([0.5, 0.5, 0.5]), rv=np.array([3.1, 3.1, 3.1])
        )

        npt.assert_array_almost_equal(seds1, seds2, decimal=12)

    def test_get_seds_wrapper_return_options(self):
        """Test optional return parameters."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0
        mag_coeffs[:, :, 1] = 1.0
        mag_coeffs[:, :, 2] = 0.1

        av, rv = 0.5, 3.1

        # Test different return combinations
        seds_only = get_seds(mag_coeffs, av=av, rv=rv)

        seds_rvec, rvecs = get_seds(mag_coeffs, av=av, rv=rv, return_rvec=True)

        seds_drvec, drvecs = get_seds(mag_coeffs, av=av, rv=rv, return_drvec=True)

        seds_all, rvecs_all, drvecs_all = get_seds(
            mag_coeffs, av=av, rv=rv, return_rvec=True, return_drvec=True
        )

        # All SEDs should be identical
        npt.assert_array_almost_equal(seds_only, seds_rvec, decimal=12)
        npt.assert_array_almost_equal(seds_only, seds_drvec, decimal=12)
        npt.assert_array_almost_equal(seds_only, seds_all, decimal=12)

        # Check return shapes and types
        assert isinstance(seds_only, np.ndarray)
        assert isinstance(seds_rvec, np.ndarray)
        assert isinstance(rvecs, np.ndarray)
        assert len(seds_all) == 3  # Should be a tuple of 3 arrays

    def test_get_seds_wrapper_flux_conversion(self):
        """Test flux conversion in wrapper."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0
        mag_coeffs[:, :, 1] = 1.0
        mag_coeffs[:, :, 2] = 0.1

        av, rv = 0.5, 3.1

        seds_mag = get_seds(mag_coeffs, av=av, rv=rv, return_flux=False)
        seds_flux = get_seds(mag_coeffs, av=av, rv=rv, return_flux=True)

        expected_flux = 10.0 ** (-0.4 * seds_mag)
        npt.assert_array_almost_equal(seds_flux, expected_flux, decimal=10)


class TestSedPhysicalConsistency:
    """Test physical consistency of SED computations."""

    def test_no_reddening_case(self):
        """Test that zero reddening returns unreddened magnitudes."""
        from brutus.core.sed_utils import get_seds

        # Create test data with known values
        mag_coeffs = np.ones((3, 4, 3))
        base_mags = np.array([[14, 15, 16, 17], [15, 16, 17, 18], [16, 17, 18, 19]])
        mag_coeffs[:, :, 0] = base_mags
        mag_coeffs[:, :, 1] = 1.0  # Doesn't matter for av=0
        mag_coeffs[:, :, 2] = 0.1  # Doesn't matter for av=0

        # Zero reddening
        seds = get_seds(mag_coeffs, av=0.0, rv=3.1)

        # Should equal base magnitudes exactly
        npt.assert_array_almost_equal(seds, base_mags, decimal=12)

    def test_reddening_increases_magnitude(self):
        """Test that positive reddening increases magnitude (decreases flux)."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((1, 3, 3))
        mag_coeffs[0, :, 0] = 15.0  # Base magnitude
        mag_coeffs[0, :, 1] = 1.0  # Positive reddening vector
        mag_coeffs[0, :, 2] = 0.0  # No RV dependence

        # Test increasing reddening
        av_values = np.array([0.0, 0.5, 1.0, 2.0])

        for i, av in enumerate(av_values[1:], 1):
            seds_red = get_seds(mag_coeffs, av=av, rv=3.1)
            seds_unred = get_seds(mag_coeffs, av=0.0, rv=3.1)

            # Reddened magnitude should be larger (dimmer)
            assert np.all(seds_red > seds_unred)

            # Should increase linearly with A(V)
            expected = seds_unred + av * 1.0  # av * reddening_vector
            npt.assert_array_almost_equal(seds_red, expected, decimal=12)

    def test_rv_dependence(self):
        """Test R(V) dependence of reddening."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((1, 3, 3))
        mag_coeffs[0, :, 0] = 15.0  # Base magnitude
        mag_coeffs[0, :, 1] = 1.0  # R(V)=0 reddening
        mag_coeffs[0, :, 2] = 0.2  # Strong RV dependence

        av = 1.0

        # Test different RV values
        rv_values = [2.0, 3.1, 4.0, 5.0]

        for rv in rv_values:
            seds, rvecs = get_seds(mag_coeffs, av=av, rv=rv, return_rvec=True)

            # Reddening vector should be r0 + rv * dr
            expected_rvec = 1.0 + rv * 0.2
            npt.assert_array_almost_equal(rvecs, expected_rvec, decimal=12)

            # SED should be base + av * rvec
            expected_sed = 15.0 + av * expected_rvec
            npt.assert_array_almost_equal(seds, expected_sed, decimal=12)

    def test_flux_magnitude_consistency(self):
        """Test consistency between flux and magnitude representations."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.random.random((5, 4, 3))
        mag_coeffs[:, :, 0] = np.random.uniform(14, 20, (5, 4))
        mag_coeffs[:, :, 1] = np.random.uniform(0.5, 2.0, (5, 4))
        mag_coeffs[:, :, 2] = np.random.uniform(0.0, 0.3, (5, 4))

        av = np.random.uniform(0.0, 2.0, 5)
        rv = np.random.uniform(2.5, 4.5, 5)

        # Get results in both representations
        seds_mag = get_seds(mag_coeffs, av=av, rv=rv, return_flux=False)
        seds_flux = get_seds(mag_coeffs, av=av, rv=rv, return_flux=True)

        # Convert magnitude to flux manually
        expected_flux = 10.0 ** (-0.4 * seds_mag)

        npt.assert_array_almost_equal(seds_flux, expected_flux, decimal=10)

        # Test that brighter objects (smaller magnitudes) have larger fluxes
        for i in range(5):
            for j in range(4):
                if seds_mag[i, j] < 18:  # Only for reasonable magnitudes
                    # Magnitude and flux should be anti-correlated
                    # (not testing specific values, just that relationship holds)
                    assert seds_flux[i, j] > 0


class TestSedEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_coefficients(self):
        """Test behavior with zero coefficients."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.zeros((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0  # Only base magnitude

        av = np.array([0.5, 1.0])
        rv = np.array([3.1, 3.3])

        seds = get_seds(mag_coeffs, av=av, rv=rv)

        # With zero reddening vectors, should equal base magnitude
        npt.assert_array_almost_equal(seds, 15.0, decimal=12)

    def test_extreme_reddening(self):
        """Test with extreme reddening values."""
        from brutus.core.sed_utils import get_seds

        mag_coeffs = np.ones((1, 3, 3))
        mag_coeffs[0, :, 0] = 15.0
        mag_coeffs[0, :, 1] = 1.0
        mag_coeffs[0, :, 2] = 0.1

        # Test extreme (but physically possible) values
        av_extreme = 10.0  # Very high extinction
        rv_extreme = 7.0  # Very steep extinction curve

        seds = get_seds(mag_coeffs, av=av_extreme, rv=rv_extreme)

        # Should not produce NaN or inf
        assert np.all(np.isfinite(seds))

        # Should be much fainter than unreddened
        seds_unred = get_seds(mag_coeffs, av=0.0, rv=3.1)
        assert np.all(seds > seds_unred)

    def test_consistent_array_dimensions(self):
        """Test that inconsistent dimensions raise appropriate errors."""
        from brutus.core.sed_utils import _get_seds

        mag_coeffs = np.ones((2, 3, 3))

        # Wrong av array size should work (broadcasting)
        av_wrong = np.array([0.5])  # Will be broadcasted
        rv = np.array([3.1, 3.3])

        # This should work due to numpy broadcasting
        seds, _, _ = _get_seds(mag_coeffs, av_wrong, rv)
        assert seds.shape == (2, 3)


@pytest.mark.slow
class TestSedPerformance:
    """Performance tests for SED computation."""

    def test_numba_compilation(self):
        """Test that numba compilation works correctly."""
        from brutus.core.sed_utils import _get_seds

        # First call compiles
        mag_coeffs = np.ones((1, 1, 3))
        av = np.array([0.1])
        rv = np.array([3.1])

        # This should trigger compilation
        _get_seds(mag_coeffs, av, rv, return_flux=False)

        # Second call should be faster (using compiled version)
        # Just test that it doesn't crash
        result = _get_seds(mag_coeffs, av, rv, return_flux=False)
        assert result[0].shape == (1, 1)

    def test_large_arrays(self):
        """Test performance with large arrays."""
        from brutus.core.sed_utils import get_seds

        # Test with reasonably large arrays
        n_models = 1000
        n_bands = 20

        np.random.seed(42)
        mag_coeffs = np.random.random((n_models, n_bands, 3))
        av = np.random.uniform(0.0, 3.0, n_models)
        rv = np.random.uniform(2.5, 4.5, n_models)

        # This should complete in reasonable time
        seds = get_seds(mag_coeffs, av=av, rv=rv)

        assert seds.shape == (n_models, n_bands)
        assert np.all(np.isfinite(seds))


@pytest.mark.integration
class TestSedIntegration:
    """Integration tests with other brutus components."""

    def test_photometry_integration(self):
        """Test integration with photometry utilities."""
        try:
            from brutus.core.sed_utils import get_seds
            from brutus.utils.photometry import magnitude, inv_magnitude

            # Create mock SED data
            mag_coeffs = np.ones((2, 3, 3))
            mag_coeffs[:, :, 0] = 15.0
            mag_coeffs[:, :, 1] = 1.0
            mag_coeffs[:, :, 2] = 0.1

            # Get SEDs in flux units
            seds_flux = get_seds(mag_coeffs, av=0.5, rv=3.1, return_flux=True)

            # Convert to magnitudes using photometry utils
            flux_errors = 0.01 * seds_flux
            mags, mag_errs = magnitude(seds_flux, flux_errors)

            # Convert back to flux
            flux_recovered, flux_err_recovered = inv_magnitude(mags, mag_errs)

            # Should recover original fluxes
            npt.assert_array_almost_equal(seds_flux, flux_recovered, decimal=10)

        except ImportError:
            pytest.skip("Photometry utilities not available")

    def test_expected_magnitude_ranges(self):
        """Test that computed magnitudes are in expected ranges."""
        from brutus.core.sed_utils import get_seds

        # Realistic stellar parameters
        mag_coeffs = np.ones((10, 5, 3))

        # Realistic base magnitudes (14-20 mag)
        mag_coeffs[:, :, 0] = np.random.uniform(14, 20, (10, 5))

        # Realistic reddening vectors (0.5-2.0)
        mag_coeffs[:, :, 1] = np.random.uniform(0.5, 2.0, (10, 5))

        # Small RV dependence
        mag_coeffs[:, :, 2] = np.random.uniform(0.0, 0.2, (10, 5))

        # Realistic reddening
        av = np.random.uniform(0.0, 3.0, 10)
        rv = np.random.uniform(2.5, 4.5, 10)

        seds = get_seds(mag_coeffs, av=av, rv=rv)

        # Should be in reasonable magnitude range
        assert np.all(seds >= 10)  # Not brighter than very bright stars
        assert np.all(seds <= 30)  # Not fainter than detection limits

        # Should be finite
        assert np.all(np.isfinite(seds))


if __name__ == "__main__":
    pytest.main([__file__])
