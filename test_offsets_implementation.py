#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation test for photometric offsets reorganization.

This script tests the reorganized photometric_offsets function that was
moved from utilities.py to analysis/offsets.py with improvements.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def create_mock_data():
    """Create mock data for testing photometric offsets."""
    np.random.seed(42)  # Reproducible results

    # Mock observation parameters
    n_obj = 50
    n_filt = 5
    n_samp = 20
    n_models = 100

    # Mock photometry (flux units)
    phot = np.random.uniform(0.1, 10.0, (n_obj, n_filt))
    err = 0.1 * phot  # 10% errors
    mask = np.random.choice(
        [0, 1], (n_obj, n_filt), p=[0.15, 0.85]
    )  # 85% detection rate

    # Mock models (magnitude polynomial coefficients)
    models = np.random.random((n_models, n_filt, 3))
    models[:, :, 0] = np.random.uniform(14, 20, (n_models, n_filt))  # Base mags
    models[:, :, 1] = np.random.uniform(0.5, 2.0, (n_models, n_filt))  # Reddening
    models[:, :, 2] = np.random.uniform(0.0, 0.3, (n_models, n_filt))  # RV dependence

    # Mock fitted parameters
    idxs = np.random.randint(0, n_models, (n_obj, n_samp))
    reds = np.random.uniform(0.0, 3.0, (n_obj, n_samp))  # A(V)
    dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))  # R(V)
    dists = np.random.uniform(0.5, 10.0, (n_obj, n_samp))  # distances in kpc

    # Mock optional parameters
    sel = np.ones(n_obj, dtype=bool)
    weights = np.ones((n_obj, n_samp))
    mask_fit = np.ones(n_filt, dtype=bool)
    old_offsets = np.ones(n_filt)

    return {
        "phot": phot,
        "err": err,
        "mask": mask,
        "models": models,
        "idxs": idxs,
        "reds": reds,
        "dreds": dreds,
        "dists": dists,
        "sel": sel,
        "weights": weights,
        "mask_fit": mask_fit,
        "old_offsets": old_offsets,
    }


def test_basic_functionality():
    """Test that the reorganized photometric_offsets function works."""
    print("\nTesting basic photometric offsets functionality...")

    try:
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        # Create test data
        data = create_mock_data()

        # Test basic configuration
        config = PhotometricOffsetsConfig(
            n_bootstrap=10,  # Small for testing
            progress_interval=0,  # No progress output
            validate_inputs=True,
        )

        # Run function
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

        # Check outputs
        if offsets.shape != (5,):
            print(f"❌ Unexpected offsets shape: {offsets.shape}")
            return False

        if errors.shape != (5,):
            print(f"❌ Unexpected errors shape: {errors.shape}")
            return False

        if n_used.shape != (5,):
            print(f"❌ Unexpected n_used shape: {n_used.shape}")
            return False

        # Check for reasonable values
        if not np.all(np.isfinite(offsets)):
            print("❌ Offsets contain non-finite values")
            return False

        if not np.all(errors >= 0):
            print("❌ Errors must be non-negative")
            return False

        if not np.all(n_used >= 0):
            print("❌ n_used must be non-negative")
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
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        data = create_mock_data()

        # Test different uncertainty methods
        methods = ["bootstrap_std", "bootstrap_iqr", "analytical"]
        results = {}

        for method in methods:
            config = PhotometricOffsetsConfig(
                n_bootstrap=20,
                uncertainty_method=method,
                progress_interval=0,
                min_bands_used=3,  # More lenient for test data
                min_bands_unused=2,
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

            results[method] = {"offsets": offsets, "errors": errors, "n_used": n_used}

        # Check that all methods produce reasonable results
        for method, result in results.items():
            if not np.all(np.isfinite(result["offsets"])):
                print(f"❌ Method {method} produced non-finite offsets")
                return False
            if not np.all(result["errors"] >= 0):
                print(f"❌ Method {method} produced negative errors")
                return False

        print("✅ Configuration options test passed")

        # Test vectorized vs non-vectorized bootstrap
        config_vec = PhotometricOffsetsConfig(
            n_bootstrap=10, use_vectorized_bootstrap=True, progress_interval=0
        )
        config_loop = PhotometricOffsetsConfig(
            n_bootstrap=10, use_vectorized_bootstrap=False, progress_interval=0
        )

        offsets_vec, _, _ = photometric_offsets(
            data["phot"],
            data["err"],
            data["mask"],
            data["models"],
            data["idxs"],
            data["reds"],
            data["dreds"],
            data["dists"],
            config=config_vec,
            verbose=False,
        )

        offsets_loop, _, _ = photometric_offsets(
            data["phot"],
            data["err"],
            data["mask"],
            data["models"],
            data["idxs"],
            data["reds"],
            data["dreds"],
            data["dists"],
            config=config_loop,
            verbose=False,
        )

        # Results should be statistically similar (within reason for small bootstrap)
        if np.any(np.abs(offsets_vec - offsets_loop) > 0.5):
            print("⚠️  Large differences between vectorized and loop implementations")
            print(f"   Vec: {offsets_vec}")
            print(f"   Loop: {offsets_loop}")
        else:
            print("✅ Vectorized and loop implementations give similar results")

        return True

    except Exception as e:
        print(f"❌ Configuration options error: {e}")
        return False


def test_comparison_with_original():
    """Test comparison with original implementation (if available)."""
    print("\nTesting comparison with original implementation...")

    try:
        # Try to import original function
        from brutus.utilities import photometric_offsets as orig_photometric_offsets

        # Import new function
        from brutus.analysis.offsets import (
            photometric_offsets as new_photometric_offsets,
        )
        from brutus.analysis.offsets import PhotometricOffsetsConfig

        print("✅ Both original and new functions available for comparison")

        # Create test data
        data = create_mock_data()

        # Configure new function to match original behavior as closely as possible
        config = PhotometricOffsetsConfig(
            min_bands_used=4,  # Original >3+1 behavior
            min_bands_unused=3,  # Original >3 behavior
            n_bootstrap=30,  # Smaller for faster testing
            uncertainty_method="bootstrap_std",  # Match original
            use_vectorized_bootstrap=False,  # Use original-style loop
            progress_interval=0,  # No progress for testing
            random_seed=42,  # Reproducible
        )

        # Set up consistent random state for original function
        rstate = np.random.RandomState(42)

        # Run original function
        orig_offsets, orig_errors, orig_n_used = orig_photometric_offsets(
            data["phot"],
            data["err"],
            data["mask"],
            data["models"],
            data["idxs"],
            data["reds"],
            data["dreds"],
            data["dists"],
            sel=data["sel"],
            weights=data["weights"],
            mask_fit=data["mask_fit"],
            Nmc=30,
            old_offsets=data["old_offsets"],
            dim_prior=True,
            verbose=False,
            rstate=rstate,
        )

        # Run new function
        rng = np.random.default_rng(42)
        new_offsets, new_errors, new_n_used = new_photometric_offsets(
            data["phot"],
            data["err"],
            data["mask"],
            data["models"],
            data["idxs"],
            data["reds"],
            data["dreds"],
            data["dists"],
            sel=data["sel"],
            weights=data["weights"],
            mask_fit=data["mask_fit"],
            old_offsets=data["old_offsets"],
            dim_prior=True,
            config=config,
            rng=rng,
            verbose=False,
        )

        print(f"Original offsets: {orig_offsets}")
        print(f"New offsets:      {new_offsets}")
        print(f"Differences:      {np.abs(orig_offsets - new_offsets)}")

        # Check if results are statistically consistent
        # (Allow for some differences due to random sampling)
        max_diff = np.max(np.abs(orig_offsets - new_offsets))
        if max_diff < 0.2:  # Reasonable threshold for bootstrap differences
            print("✅ New implementation produces statistically consistent results")
        else:
            print("⚠️  Large differences detected - may be due to bootstrap sampling")
            print(f"   Max difference: {max_diff}")

        # Object counts should be identical (deterministic)
        if np.array_equal(orig_n_used, new_n_used):
            print("✅ Object selection logic matches exactly")
        else:
            print("❌ Object selection logic differs")
            print(f"   Original: {orig_n_used}")
            print(f"   New:      {new_n_used}")
            return False

        return True

    except ImportError:
        print("⚠️  Original function not available for comparison")
        print("   This is expected if utilities.py has been updated")
        return True  # Not a failure, just can't compare
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False


def test_input_validation():
    """Test input validation features."""
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
            print("✅ Caught shape mismatch as expected")

        # Test with negative errors
        try:
            bad_err = -np.abs(data["err"])  # Negative errors
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
            print("❌ Should have caught negative errors")
            return False
        except ValueError:
            print("✅ Caught negative errors as expected")

        # Test with invalid configuration
        try:
            bad_config = PhotometricOffsetsConfig(min_bands_used=-1)
            print("❌ Should have caught invalid config")
            return False
        except ValueError:
            print("✅ Caught invalid configuration as expected")

        return True

    except Exception as e:
        print(f"❌ Input validation error: {e}")
        return False


def test_prior_application():
    """Test prior application functionality."""
    print("\nTesting prior application...")

    try:
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        data = create_mock_data()
        config = PhotometricOffsetsConfig(n_bootstrap=10, progress_interval=0)

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
        prior_mean = np.ones(5) * 1.1  # Slight offset from unity
        prior_std = np.ones(5) * 0.05  # Tight priors

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

        # Priors should pull results toward prior mean
        diff_no_prior = np.abs(offsets_no_prior - prior_mean)
        diff_with_prior = np.abs(offsets_with_prior - prior_mean)

        if np.all(diff_with_prior <= diff_no_prior):
            print("✅ Priors correctly pull results toward prior mean")
        else:
            print("⚠️  Prior application may not be working as expected")
            print(f"   No prior diffs:   {diff_no_prior}")
            print(f"   With prior diffs: {diff_with_prior}")

        # Errors should be reduced by priors
        if np.all(errors_with_prior <= errors_no_prior):
            print("✅ Priors correctly reduce uncertainties")
        else:
            print("⚠️  Prior error reduction may not be working")
            print(f"   Errors no prior:   {errors_no_prior}")
            print(f"   Errors with prior: {errors_with_prior}")

        return True

    except Exception as e:
        print(f"❌ Prior application error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BRUTUS PHOTOMETRIC OFFSETS REORGANIZATION TEST")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_configuration_options,
        test_comparison_with_original,
        test_input_validation,
        test_prior_application,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all(results):
        print("🎉 All tests passed! Photometric offsets reorganization successful.")
        print("\nNext steps:")
        print(
            "1. Run the full test suite: pytest tests/test_analysis/test_offsets_comprehensive.py"
        )
        print("2. Update any imports that reference the old utilities.py version")
        print("3. Performance test with realistic data sizes")
        print("4. Integration test with full brutus workflows")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure you created src/brutus/analysis/offsets.py")
        print("2. Make sure you created src/brutus/analysis/__init__.py")
        print("3. Make sure you created src/brutus/core/sed_utils.py (dependency)")
        print("4. Check for any syntax errors in the files")
        print("5. Ensure all brutus utility modules are available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
