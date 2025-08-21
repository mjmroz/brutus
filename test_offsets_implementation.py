#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test suite for brutus photometric offset analysis.

This combines basic functionality tests with proven parameter recovery tests
using the real get_seds function for validation.
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def create_mock_data():
    """Create realistic mock data for basic functionality tests."""
    np.random.seed(42)

    n_obj, n_filt, n_samp, n_models = 50, 5, 20, 100

    # Mock photometry
    phot = np.random.uniform(0.1, 10.0, (n_obj, n_filt))
    err = 0.05 * phot + 0.01
    mask = np.random.choice([0, 1], (n_obj, n_filt), p=[0.1, 0.9])

    # Mock models with realistic coefficients
    models = np.random.random((n_models, n_filt, 3))
    models[:, :, 0] = np.random.uniform(14, 20, (n_models, n_filt))  # Base mags
    models[:, :, 1] = np.random.uniform(0.5, 2.0, (n_models, n_filt))  # A(V) coeffs
    models[:, :, 2] = np.random.uniform(0.0, 0.3, (n_models, n_filt))  # R(V) coeffs

    # Mock fitted parameters
    idxs = np.random.randint(0, n_models, (n_obj, n_samp))
    reds = np.random.uniform(0.0, 3.0, (n_obj, n_samp))
    dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))
    dists = np.random.uniform(0.5, 10.0, (n_obj, n_samp))

    return {
        "phot": phot,
        "err": err,
        "mask": mask,
        "models": models,
        "idxs": idxs,
        "reds": reds,
        "dreds": dreds,
        "dists": dists,
    }


def setup_simple_mocks():
    """Set up simple, working mocks for basic functionality tests."""

    def mock_get_seds(models_input, av=None, rv=None, return_flux=True):
        n_requested = len(models_input) if hasattr(models_input, "__len__") else 1
        n_filt = 5
        if return_flux:
            return np.random.uniform(0.1, 10.0, (n_requested, n_filt))
        else:
            return np.random.uniform(14, 20, (n_requested, n_filt))

    def mock_phot_loglike(flux, err, mfluxes, mask=None, dim_prior=True):
        nobj, nfilt = flux.shape
        nmod = mfluxes.shape[1] if len(mfluxes.shape) > 1 else 1
        return np.random.normal(-5, 2, (nobj, nmod))

    def mock_logsumexp(x, axis=1):
        if hasattr(x, "shape") and len(x.shape) > 1:
            return np.log(
                np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1)
            ) + np.max(x, axis=1)
        else:
            return np.log(np.sum(np.exp(x - np.max(x)))) + np.max(x)

    return mock_get_seds, mock_phot_loglike, mock_logsumexp


def test_basic_functionality():
    """Test basic function execution with simple inputs."""
    print("\nTesting basic functionality...")

    try:
        mock_get_seds, mock_phot_loglike, mock_logsumexp = setup_simple_mocks()

        with patch(
            "brutus.analysis.offsets.get_seds", side_effect=mock_get_seds
        ), patch(
            "brutus.analysis.offsets.phot_loglike", side_effect=mock_phot_loglike
        ), patch(
            "brutus.analysis.offsets.logsumexp", side_effect=mock_logsumexp
        ):

            from brutus.analysis.offsets import (
                photometric_offsets,
                PhotometricOffsetsConfig,
            )

            data = create_mock_data()

            config = PhotometricOffsetsConfig(
                n_bootstrap=10,
                min_bands_used=2,
                min_bands_unused=1,
                progress_interval=0,
                validate_inputs=False,
            )

            offsets, errors, n_used = photometric_offsets(
                data["phot"],
                data["err"],
                data["mask"],
                data["models"],
                data["idxs"],
                data["reds"],
                data["dreds"],
                data["dists"],
                config=config,
                verbose=False,
            )

            # Check output shapes and types
            if offsets.shape != (5,):
                print(f"❌ Unexpected offsets shape: {offsets.shape}")
                return False

            if not np.all(np.isfinite(offsets)):
                print("❌ Offsets contain non-finite values")
                return False

            if not np.all(errors >= 0):
                print("❌ Errors must be non-negative")
                return False

            print("✅ Basic functionality tests passed")
            print(f"   Computed offsets: {offsets}")
            print(f"   Offset errors: {errors}")
            print(f"   Objects used: {n_used}")
            return True

    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_configuration_options():
    """Test different configuration options."""
    print("\nTesting configuration options...")

    try:
        mock_get_seds, mock_phot_loglike, mock_logsumexp = setup_simple_mocks()

        with patch(
            "brutus.analysis.offsets.get_seds", side_effect=mock_get_seds
        ), patch(
            "brutus.analysis.offsets.phot_loglike", side_effect=mock_phot_loglike
        ), patch(
            "brutus.analysis.offsets.logsumexp", side_effect=mock_logsumexp
        ):

            from brutus.analysis.offsets import (
                photometric_offsets,
                PhotometricOffsetsConfig,
            )

            data = create_mock_data()

            # Test different uncertainty methods
            methods = ["bootstrap_std", "bootstrap_iqr"]
            for method in methods:
                config = PhotometricOffsetsConfig(
                    n_bootstrap=15,
                    uncertainty_method=method,
                    progress_interval=0,
                    validate_inputs=False,
                )

                offsets, errors, n_used = photometric_offsets(
                    data["phot"],
                    data["err"],
                    data["mask"],
                    data["models"],
                    data["idxs"],
                    data["reds"],
                    data["dreds"],
                    data["dists"],
                    config=config,
                    verbose=False,
                )

                if not np.all(np.isfinite(offsets)) or not np.all(errors >= 0):
                    print(f"❌ Method {method} produced invalid results")
                    return False

            # Test vectorized vs loop bootstrap
            for use_vec in [True, False]:
                config = PhotometricOffsetsConfig(
                    n_bootstrap=10,
                    use_vectorized_bootstrap=use_vec,
                    progress_interval=0,
                    validate_inputs=False,
                )

                offsets, errors, n_used = photometric_offsets(
                    data["phot"],
                    data["err"],
                    data["mask"],
                    data["models"],
                    data["idxs"],
                    data["reds"],
                    data["dreds"],
                    data["dists"],
                    config=config,
                    verbose=False,
                )

                if not np.all(np.isfinite(offsets)):
                    print(f"❌ Vectorized={use_vec} produced invalid results")
                    return False

            print("✅ Configuration options test passed")
            return True

    except Exception as e:
        print(f"❌ Configuration options error: {e}")
        return False


def test_input_validation():
    """Test input validation functionality."""
    print("\nTesting input validation...")

    try:
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        data = create_mock_data()
        config = PhotometricOffsetsConfig(validate_inputs=True, n_bootstrap=5)

        # Test with mismatched array shapes
        try:
            bad_err = data["err"][:10]  # Wrong shape
            photometric_offsets(
                data["phot"],
                bad_err,
                data["mask"],
                data["models"],
                data["idxs"],
                data["reds"],
                data["dreds"],
                data["dists"],
                config=config,
                verbose=False,
            )
            print("❌ Should have caught shape mismatch")
            return False
        except ValueError:
            pass  # Expected

        # Test invalid configuration
        try:
            bad_config = PhotometricOffsetsConfig(min_bands_used=-1)
            print("❌ Should have caught invalid config")
            return False
        except ValueError:
            pass  # Expected

        print("✅ Input validation tests passed")
        return True

    except Exception as e:
        print(f"❌ Input validation error: {e}")
        return False


def test_unity_recovery_real_get_seds():
    """Test unity recovery using real get_seds function."""
    print("\nTesting unity recovery with real get_seds...")

    try:
        from brutus.analysis.offsets import (
            get_seds,
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        n_obj, n_filt = 80, 5
        n_models, n_samp = 30, 15

        # Create realistic model grid
        np.random.seed(42)
        models = np.random.random((n_models, n_filt, 3))
        models[:, :, 0] = np.random.uniform(15, 19, (n_models, n_filt))
        models[:, :, 1] = np.random.uniform(0.6, 1.8, (n_models, n_filt))
        models[:, :, 2] = np.random.uniform(0.0, 0.25, (n_models, n_filt))

        # True parameters for data generation
        true_model_idxs = np.random.randint(0, n_models, n_obj)
        true_avs = np.random.uniform(0.0, 1.2, n_obj)
        true_rvs = np.random.uniform(3.1, 3.5, n_obj)
        true_dists = np.random.uniform(0.9, 1.1, n_obj)

        # Generate observed photometry using real get_seds
        observed_phot = np.zeros((n_obj, n_filt))
        for i in range(n_obj):
            true_sed = get_seds(
                models[true_model_idxs[i : i + 1]],
                av=np.array([true_avs[i]]),
                rv=np.array([true_rvs[i]]),
                return_flux=True,
            )
            observed_phot[i] = true_sed[0] / (true_dists[i] ** 2)

        # Add realistic noise
        noise = np.random.normal(0, 0.025 * observed_phot)
        observed_phot += noise
        observed_phot = np.maximum(observed_phot, 0.1 * np.abs(observed_phot))

        err = 0.03 * observed_phot
        mask = np.ones((n_obj, n_filt), dtype=int)

        # Create samples centered on true values
        idxs = np.zeros((n_obj, n_samp), dtype=int)
        reds = np.zeros((n_obj, n_samp))
        dreds = np.zeros((n_obj, n_samp))
        dists = np.zeros((n_obj, n_samp))

        for i in range(n_obj):
            idxs[i] = true_model_idxs[i]
            reds[i] = np.random.normal(true_avs[i], 0.015, n_samp)
            dreds[i] = np.random.normal(true_rvs[i], 0.008, n_samp)
            dists[i] = np.random.normal(true_dists[i], 0.008, n_samp)

            reds[i] = np.clip(reds[i], 0.0, 2.5)
            dreds[i] = np.clip(dreds[i], 2.9, 3.9)
            dists[i] = np.clip(dists[i], 0.6, 1.5)

        # Mock only the likelihood functions
        def mock_phot_loglike(flux, err_in, mfluxes, mask=None, dim_prior=True):
            nobj = flux.shape[0]
            nmod = mfluxes.shape[1] if len(mfluxes.shape) > 1 else 1
            return np.zeros((nobj, nmod))

        def mock_logsumexp(x, axis=1):
            if hasattr(x, "shape") and len(x.shape) > 1:
                return np.zeros(x.shape[0])
            else:
                return 0.0

        # Run the algorithm
        with patch(
            "brutus.analysis.offsets.phot_loglike", side_effect=mock_phot_loglike
        ), patch("brutus.analysis.offsets.logsumexp", side_effect=mock_logsumexp):

            config = PhotometricOffsetsConfig(
                n_bootstrap=25,
                min_bands_used=3,
                min_bands_unused=2,
                validate_inputs=False,
                progress_interval=0,
            )

            offsets, errors, n_used = photometric_offsets(
                observed_phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                config=config,
                verbose=False,
            )

        # Check unity recovery
        max_deviation = np.max(np.abs(offsets - 1.0))
        max_rel_error = np.max(np.abs((offsets - 1.0) / 1.0))

        print(f"   Recovered offsets: {offsets}")
        print(f"   Max deviation: {max_deviation:.6f}")
        print(f"   Max relative error: {max_rel_error:.6f}")

        if max_rel_error < 0.02:
            print("✅ Unity recovery: EXCELLENT")
            return True
        elif max_rel_error < 0.05:
            print("✅ Unity recovery: GOOD")
            return True
        else:
            print(f"❌ Unity recovery failed: {max_rel_error:.3f} > 0.05")
            return False

    except Exception as e:
        print(f"❌ Unity recovery test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_systematic_offset_recovery():
    """Test recovery of known systematic offsets."""
    print("\nTesting systematic offset recovery...")

    try:
        from brutus.analysis.offsets import (
            get_seds,
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        # Apply known systematic offsets
        true_offsets = np.array([1.05, 0.95, 1.08, 0.92, 1.02])
        expected_corrections = 1.0 / true_offsets

        print(f"   Applied offsets: {true_offsets}")
        print(f"   Expected corrections: {expected_corrections}")

        n_obj, n_filt = 120, 5
        n_models, n_samp = 25, 12

        # Create model grid
        np.random.seed(123)
        models = np.random.random((n_models, n_filt, 3))
        models[:, :, 0] = np.random.uniform(15, 18, (n_models, n_filt))
        models[:, :, 1] = np.random.uniform(0.7, 1.6, (n_models, n_filt))
        models[:, :, 2] = np.random.uniform(0.0, 0.2, (n_models, n_filt))

        # True parameters
        true_model_idxs = np.random.randint(0, n_models, n_obj)
        true_avs = np.random.uniform(0.0, 1.0, n_obj)
        true_rvs = np.random.uniform(3.15, 3.45, n_obj)
        true_dists = np.random.uniform(0.95, 1.05, n_obj)

        # Generate photometry and apply systematic offsets
        observed_phot = np.zeros((n_obj, n_filt))
        for i in range(n_obj):
            true_sed = get_seds(
                models[true_model_idxs[i : i + 1]],
                av=np.array([true_avs[i]]),
                rv=np.array([true_rvs[i]]),
                return_flux=True,
            )
            base_flux = true_sed[0] / (true_dists[i] ** 2)
            observed_phot[i] = base_flux * true_offsets  # Apply systematic errors

        # Add noise
        noise = np.random.normal(0, 0.02 * observed_phot)
        observed_phot += noise
        observed_phot = np.maximum(observed_phot, 0.05 * np.abs(observed_phot))

        err = 0.025 * observed_phot
        mask = np.ones((n_obj, n_filt), dtype=int)

        # Create samples
        idxs = np.zeros((n_obj, n_samp), dtype=int)
        reds = np.zeros((n_obj, n_samp))
        dreds = np.zeros((n_obj, n_samp))
        dists = np.zeros((n_obj, n_samp))

        for i in range(n_obj):
            idxs[i] = true_model_idxs[i]
            reds[i] = np.random.normal(true_avs[i], 0.01, n_samp)
            dreds[i] = np.random.normal(true_rvs[i], 0.005, n_samp)
            dists[i] = np.random.normal(true_dists[i], 0.005, n_samp)

            reds[i] = np.clip(reds[i], 0.0, 2.0)
            dreds[i] = np.clip(dreds[i], 3.0, 3.8)
            dists[i] = np.clip(dists[i], 0.7, 1.3)

        # Mock likelihood functions
        def mock_phot_loglike(flux, err_in, mfluxes, mask=None, dim_prior=True):
            nobj = flux.shape[0]
            nmod = mfluxes.shape[1] if len(mfluxes.shape) > 1 else 1
            return np.zeros((nobj, nmod))

        def mock_logsumexp(x, axis=1):
            if hasattr(x, "shape") and len(x.shape) > 1:
                return np.zeros(x.shape[0])
            else:
                return 0.0

        # Run algorithm
        with patch(
            "brutus.analysis.offsets.phot_loglike", side_effect=mock_phot_loglike
        ), patch("brutus.analysis.offsets.logsumexp", side_effect=mock_logsumexp):

            config = PhotometricOffsetsConfig(
                n_bootstrap=30,
                min_bands_used=3,
                min_bands_unused=2,
                validate_inputs=False,
                progress_interval=0,
            )

            recovered_corrections, errors, n_used = photometric_offsets(
                observed_phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                config=config,
                verbose=False,
            )

        # Check recovery
        abs_errors = np.abs(recovered_corrections - expected_corrections)
        rel_errors = abs_errors / expected_corrections
        max_rel_error = np.max(rel_errors)

        print(f"   Recovered corrections: {recovered_corrections}")
        print(f"   Absolute errors: {abs_errors}")
        print(f"   Max relative error: {max_rel_error:.6f}")

        if max_rel_error < 0.05:
            print("✅ Systematic offset recovery: EXCELLENT")
            return True
        elif max_rel_error < 0.15:
            print("✅ Systematic offset recovery: GOOD")
            return True
        else:
            print(f"❌ Systematic offset recovery failed: {max_rel_error:.3f}")
            return False

    except Exception as e:
        print(f"❌ Systematic offset recovery failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prior_application():
    """Test Gaussian prior application."""
    print("\nTesting prior application...")

    try:
        mock_get_seds, mock_phot_loglike, mock_logsumexp = setup_simple_mocks()

        with patch(
            "brutus.analysis.offsets.get_seds", side_effect=mock_get_seds
        ), patch(
            "brutus.analysis.offsets.phot_loglike", side_effect=mock_phot_loglike
        ), patch(
            "brutus.analysis.offsets.logsumexp", side_effect=mock_logsumexp
        ):

            from brutus.analysis.offsets import (
                photometric_offsets,
                PhotometricOffsetsConfig,
            )

            data = create_mock_data()

            config = PhotometricOffsetsConfig(
                n_bootstrap=15, progress_interval=0, validate_inputs=False
            )

            # Test without priors
            offsets_no_prior, errors_no_prior, _ = photometric_offsets(
                data["phot"],
                data["err"],
                data["mask"],
                data["models"],
                data["idxs"],
                data["reds"],
                data["dreds"],
                data["dists"],
                config=config,
                verbose=False,
            )

            # Test with priors
            prior_mean = np.ones(5) * 1.05
            prior_std = np.ones(5) * 0.02

            offsets_with_prior, errors_with_prior, _ = photometric_offsets(
                data["phot"],
                data["err"],
                data["mask"],
                data["models"],
                data["idxs"],
                data["reds"],
                data["dreds"],
                data["dists"],
                prior_mean=prior_mean,
                prior_std=prior_std,
                config=config,
                verbose=False,
            )

            # Both should produce finite results
            if not (
                np.all(np.isfinite(offsets_no_prior))
                and np.all(np.isfinite(offsets_with_prior))
            ):
                print("❌ Prior application produced non-finite results")
                return False

            if not (np.all(errors_no_prior >= 0) and np.all(errors_with_prior >= 0)):
                print("❌ Prior application produced negative errors")
                return False

            print("✅ Prior application test passed")
            return True

    except Exception as e:
        print(f"❌ Prior application error: {e}")
        return False


def main():
    """Run comprehensive photometric offsets test suite."""
    print("=" * 60)
    print("COMPREHENSIVE PHOTOMETRIC OFFSETS TEST SUITE")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_configuration_options,
        test_input_validation,
        test_unity_recovery_real_get_seds,
        test_systematic_offset_recovery,
        test_prior_application,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)

    test_names = [
        "Basic Functionality",
        "Configuration Options",
        "Input Validation",
        "Unity Recovery (Real get_seds)",
        "Systematic Offset Recovery",
        "Prior Application",
    ]

    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<45} {status}")

    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print("\n🎉 All tests passed! Photometric offsets implementation is robust.")
        print("\nNext steps:")
        print("1. Integration test with full brutus workflows")
        print("2. Performance benchmarking with large datasets")
        print("3. Comparison with published offset catalogs")
        return 0
    else:
        print(
            f"\n❌ {total_count - success_count} test(s) failed. Implementation needs attention."
        )
        print("\nTroubleshooting:")
        print("1. Check that all brutus dependencies are properly imported")
        print("2. Verify model grid format matches expected structure")
        print("3. Test with realistic astrophysical parameter ranges")
        return 1


if __name__ == "__main__":
    sys.exit(main())
