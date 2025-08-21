#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the sampling utilities module reorganization.

Run this script after creating the new sampling.py file to verify
that our reorganization is working correctly.
"""

import sys
import numpy as np


def test_basic_functionality():
    """Test basic sampling functionality."""
    print("Testing basic sampling functionality...")

    try:
        # Test imports
        from brutus.utils.sampling import quantile, draw_sar, sample_multivariate_normal

        print("✅ Successfully imported sampling functions")

        # Test quantile computation
        x = np.array([1, 2, 3, 4, 5])
        q = np.array([0.25, 0.5, 0.75])

        result = quantile(x, q)
        expected = np.percentile(x, q * 100)

        if np.allclose(result, expected):
            print("✅ Quantile computation working correctly")
        else:
            print("❌ Quantile computation issue")
            print(f"   Got: {result}")
            print(f"   Expected: {expected}")
            return False

        # Test weighted quantiles
        weights = np.array([1, 1, 1, 1, 10])  # Last sample heavily weighted
        weighted_result = quantile(x, 0.5, weights=weights)

        if weighted_result > 3.0:  # Should be > 3 due to heavy weighting on 5
            print("✅ Weighted quantile computation working correctly")
        else:
            print(f"❌ Weighted quantile issue: got {weighted_result}, expected > 3")
            return False

        # Test sample_multivariate_normal (single distribution)
        np.random.seed(42)
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])

        samples = sample_multivariate_normal(
            mean, cov, size=100, rstate=np.random.RandomState(42)
        )

        if samples.shape == (2, 100):
            print("✅ Single multivariate normal sampling working correctly")
        else:
            print(
                f"❌ Single MVN sampling shape issue: got {samples.shape}, expected (2, 100)"
            )
            return False

        # Test sample_multivariate_normal (multiple distributions)
        means = np.array([[1.0, 2.0], [3.0, 4.0]])
        covs = np.array([[[1.0, 0.2], [0.2, 1.0]], [[2.0, 0.5], [0.5, 2.0]]])

        multi_samples = sample_multivariate_normal(
            means, covs, size=50, rstate=np.random.RandomState(42)
        )

        if multi_samples.shape == (2, 50, 2):
            print("✅ Multiple multivariate normal sampling working correctly")
        else:
            print(
                f"❌ Multiple MVN sampling shape issue: got {multi_samples.shape}, expected (2, 50, 2)"
            )
            return False

        # Test draw_sar
        scales = np.array([1.0, 1.1])
        avs = np.array([0.1, 0.2])
        rvs = np.array([3.1, 3.3])
        covs_sar = np.array(
            [
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
            ]
        )

        sdraws, adraws, rdraws = draw_sar(
            scales, avs, rvs, covs_sar, ndraws=100, rstate=np.random.RandomState(42)
        )

        if (
            sdraws.shape == (2, 100)
            and adraws.shape == (2, 100)
            and rdraws.shape == (2, 100)
        ):
            print("✅ draw_sar working correctly")
        else:
            print(f"❌ draw_sar shape issue")
            print(
                f"   sdraws: {sdraws.shape}, adraws: {adraws.shape}, rdraws: {rdraws.shape}"
            )
            return False

        # Test that draw_sar respects bounds
        if (
            np.all(sdraws >= 0)
            and np.all(adraws >= 0)
            and np.all(adraws <= 6)
            and np.all(rdraws >= 1)
            and np.all(rdraws <= 8)
        ):
            print("✅ draw_sar respects bounds correctly")
        else:
            print("❌ draw_sar bounds issue")
            return False

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you've created src/brutus/utils/sampling.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_comparison_with_original():
    """Test comparison with original utilities.py functions if available."""
    print("\nTesting comparison with original functions...")

    try:
        # Try to import original functions
        from brutus.utilities import (
            quantile as orig_quantile,
            draw_sar as orig_draw_sar,
            sample_multivariate_normal as orig_sample_mvn,
        )

        # Import new functions
        from brutus.utils.sampling import (
            quantile as new_quantile,
            draw_sar as new_draw_sar,
            sample_multivariate_normal as new_sample_mvn,
        )

        print("✅ Both original and new functions available for comparison")

        # Test quantile comparison
        np.random.seed(42)
        x = np.random.random(50)
        q = np.array([0.25, 0.5, 0.75])
        weights = np.random.random(50)

        # Unweighted comparison
        orig_quant = orig_quantile(x, q)
        new_quant = new_quantile(x, q)

        if np.allclose(orig_quant, new_quant, rtol=1e-12):
            print("✅ quantile functions match exactly (unweighted)")
        else:
            print("❌ quantile functions don't match (unweighted)")
            return False

        # Weighted comparison
        orig_weighted = orig_quantile(x, q, weights=weights)
        new_weighted = new_quantile(x, q, weights=weights)

        if np.allclose(orig_weighted, new_weighted, rtol=1e-12):
            print("✅ quantile functions match exactly (weighted)")
        else:
            print("❌ quantile functions don't match (weighted)")
            return False

        # Test draw_sar comparison
        scales = np.array([1.0, 1.1])
        avs = np.array([0.1, 0.2])
        rvs = np.array([3.1, 3.3])
        covs_sar = np.array(
            [
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
            ]
        )

        # Use identical random states
        rstate1 = np.random.RandomState(42)
        rstate2 = np.random.RandomState(42)

        orig_results = orig_draw_sar(
            scales, avs, rvs, covs_sar, ndraws=50, rstate=rstate1
        )
        new_results = new_draw_sar(
            scales, avs, rvs, covs_sar, ndraws=50, rstate=rstate2
        )

        match = True
        for orig, new in zip(orig_results, new_results):
            if not np.allclose(orig, new, rtol=1e-12):
                match = False
                break

        if match:
            print("✅ draw_sar functions match exactly")
        else:
            print("❌ draw_sar functions don't match")
            return False

        # Test sample_multivariate_normal comparison
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])

        rstate1 = np.random.RandomState(42)
        rstate2 = np.random.RandomState(42)

        orig_mvn = orig_sample_mvn(mean, cov, size=10, rstate=rstate1)
        new_mvn = new_sample_mvn(mean, cov, size=10, rstate=rstate2)

        if np.allclose(orig_mvn, new_mvn, rtol=1e-12):
            print("✅ sample_multivariate_normal functions match exactly")
        else:
            print("❌ sample_multivariate_normal functions don't match")
            return False

        return True

    except ImportError:
        print("⚠️  Original functions not available for comparison")
        print("   This is expected if you haven't installed the package yet")
        return True  # Not a failure, just can't compare
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False


def test_import_structure():
    """Test that the import structure works as expected."""
    print("\nTesting import structure...")

    try:
        # Test importing from utils package
        from brutus.utils import quantile, draw_sar, sample_multivariate_normal

        print("✅ Can import sampling functions from brutus.utils")

        # Test importing directly from sampling module
        from brutus.utils.sampling import quantile as sampling_quantile

        print("✅ Can import directly from brutus.utils.sampling")

        # Test that they're the same function
        assert quantile is sampling_quantile
        print("✅ Import paths point to same function")

        return True

    except Exception as e:
        print(f"❌ Import structure error: {e}")
        return False


def test_statistical_properties():
    """Test statistical properties of sampling functions."""
    print("\nTesting statistical properties...")

    try:
        from brutus.utils.sampling import quantile, sample_multivariate_normal

        # Test that quantiles are in correct order
        np.random.seed(42)
        x = np.random.random(1000)
        q = np.array([0.1, 0.5, 0.9])

        result = quantile(x, q)

        if result[0] <= result[1] <= result[2]:
            print("✅ Quantiles in correct ascending order")
        else:
            print(f"❌ Quantiles not in order: {result}")
            return False

        # Test that multivariate normal has correct mean
        mean = np.array([5.0, 10.0])
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])

        samples = sample_multivariate_normal(
            mean, cov, size=2000, rstate=np.random.RandomState(42)
        )

        sample_mean = np.mean(samples, axis=1)

        if np.allclose(
            sample_mean, mean, atol=0.2
        ):  # Should be close with 2000 samples
            print("✅ Multivariate normal sample mean correct")
        else:
            print(f"❌ Sample mean issue: got {sample_mean}, expected {mean}")
            return False

        # Test that sample covariance is approximately correct
        sample_cov = np.cov(samples)

        if np.allclose(sample_cov, cov, atol=0.2):
            print("✅ Multivariate normal sample covariance correct")
        else:
            print(f"❌ Sample covariance issue")
            print(f"   Got: {sample_cov}")
            print(f"   Expected: {cov}")
            return False

        return True

    except Exception as e:
        print(f"❌ Statistical properties error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BRUTUS SAMPLING UTILITIES REORGANIZATION TEST")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_comparison_with_original,
        test_import_structure,
        test_statistical_properties,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all(results):
        print("🎉 All tests passed! Sampling utilities reorganization successful.")
        print("\nNext steps:")
        print(
            "1. Run the full test suite: pytest tests/test_utils/test_sampling_comprehensive.py"
        )
        print("2. Move on to data functions reorganization")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure you created src/brutus/utils/sampling.py")
        print("2. Make sure you updated src/brutus/utils/__init__.py")
        print("3. Check for any syntax errors in the files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
