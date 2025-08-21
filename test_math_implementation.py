#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the mathematical utilities module reorganization.

Run this script after creating the new math.py file to verify
that our reorganization is working correctly.
"""

import sys
import numpy as np

from brutus.utils.math import inverse3


def test_basic_functionality():
    """Test basic mathematical functionality."""
    print("Testing basic mathematical functionality...")

    try:
        # Test imports
        from brutus.utils.math import (
            adjoint3,
            inverse3,
            dot3,
            isPSD,
            chisquare_logpdf,
            truncnorm_pdf,
            _function_wrapper,
        )

        print("✅ Successfully imported mathematical functions")

        # Test 3x3 matrix operations
        A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=float)

        # Test matrix inversion
        A_inv = inverse3(A)
        product = np.dot(A, A_inv)
        identity = np.eye(3)

        if np.allclose(product, identity, rtol=1e-12):
            print("✅ Matrix inversion working correctly")
        else:
            print("❌ Matrix inversion accuracy issue")
            print(f"   Product: {product}")
            print(f"   Expected: {identity}")
            return False

        # Test adjoint computation
        adj_A = adjoint3(A)
        det_A = np.linalg.det(A)
        expected_product = det_A * identity
        actual_product = np.dot(A, adj_A)

        if np.allclose(actual_product, expected_product, rtol=1e-10):
            print("✅ Adjoint computation working correctly")
        else:
            print("❌ Adjoint computation issue")
            return False

        # Test dot product
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = dot3(a, b)
        expected = np.dot(a, b)

        if result == expected:
            print("✅ Dot product working correctly")
        else:
            print(f"❌ Dot product issue: got {result}, expected {expected}")
            return False

        # Test positive semidefinite check
        I = np.eye(3)
        if isPSD(I):
            print("✅ Positive semidefinite check working correctly")
        else:
            print("❌ PSD check failed for identity matrix")
            return False

        # Test chi-square log-PDF
        x = 2.0
        df = 3
        logpdf = chisquare_logpdf(x, df)

        if np.isfinite(logpdf) and logpdf < 0:
            print("✅ Chi-square log-PDF working correctly")
        else:
            print(f"❌ Chi-square log-PDF issue: got {logpdf}")
            return False

        # Test truncated normal PDF
        x = 0.0
        a, b = -2.0, 2.0
        pdf = truncnorm_pdf(x, a, b)

        if np.isfinite(pdf) and pdf > 0:
            print("✅ Truncated normal PDF working correctly")
        else:
            print(f"❌ Truncated normal PDF issue: got {pdf}")
            return False

        # Test function wrapper
        def test_func(x, a):
            return x * a

        wrapped = _function_wrapper(test_func, args=(2,), kwargs={})
        result = wrapped(5)

        if result == 10:
            print("✅ Function wrapper working correctly")
        else:
            print(f"❌ Function wrapper issue: got {result}, expected 10")
            return False

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you've created src/brutus/utils/math.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_comparison_with_original():
    """Test comparison with original utils.py functions if available."""
    print("\nTesting comparison with original functions...")

    try:
        # Try to import original functions
        from brutus.utilities import (
            _adjoint3 as orig_adjoint3,
            _inverse3 as orig_inverse3,
            _chisquare_logpdf as orig_chisquare_logpdf,
            _truncnorm_pdf as orig_truncnorm_pdf,
        )

        # Import new functions
        from brutus.utils.math import (
            adjoint3 as new_adjoint3,
            inverse3 as new_inverse3,
            chisquare_logpdf as new_chisquare_logpdf,
            truncnorm_pdf as new_truncnorm_pdf,
        )

        print("✅ Both original and new functions available for comparison")

        # Test adjoint computation
        np.random.seed(42)
        A = np.random.random((2, 3, 3))

        orig_adj = orig_adjoint3(A)
        new_adj = new_adjoint3(A)

        if np.allclose(orig_adj, new_adj, rtol=1e-12):
            print("✅ _adjoint3 functions match exactly")
        else:
            print("❌ _adjoint3 functions don't match")
            return False

        # Test matrix inversion (create invertible matrices)
        A_inv_test = np.random.random((2, 3, 3))
        A_inv_test = (
            A_inv_test + np.transpose(A_inv_test, (0, 2, 1)) + 3 * np.eye(3)[None, :, :]
        )

        orig_inv = orig_inverse3(A_inv_test)
        new_inv = new_inverse3(A_inv_test)

        if np.allclose(orig_inv, new_inv, rtol=1e-12):
            print("✅ _inverse3 functions match exactly")
        else:
            print("❌ _inverse3 functions don't match")
            return False

        # Test chi-square log-PDF
        x = np.array([1.0, 2.0, 5.0])
        df = 3

        orig_chi2 = orig_chisquare_logpdf(x, df)
        new_chi2 = new_chisquare_logpdf(x, df)

        if np.allclose(orig_chi2, new_chi2, rtol=1e-12):
            print("✅ _chisquare_logpdf functions match exactly")
        else:
            print("❌ _chisquare_logpdf functions don't match")
            return False

        # Test truncated normal PDF
        x = np.array([-1.0, 0.0, 1.0])
        a, b = -2.0, 2.0

        orig_truncnorm = orig_truncnorm_pdf(x, a, b)
        new_truncnorm = new_truncnorm_pdf(x, a, b)

        if np.allclose(orig_truncnorm, new_truncnorm, rtol=1e-12):
            print("✅ _truncnorm_pdf functions match exactly")
        else:
            print("❌ _truncnorm_pdf functions don't match")
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
        from brutus.utils import inverse3

        print("✅ Can import mathematical functions from brutus.utils")

        # Test importing directly from math module
        from brutus.utils.math import inverse3 as math_inverse3

        print("✅ Can import directly from brutus.utils.math")

        # Test that they're the same function
        assert inverse3 is math_inverse3
        print("✅ Import paths point to same function")

        return True

    except Exception as e:
        print(f"❌ Import structure error: {e}")
        return False


def test_against_scipy():
    """Test against scipy reference implementations."""
    print("\nTesting against scipy reference implementations...")

    try:
        import scipy.stats
        from brutus.utils.math import (
            chisquare_logpdf,
            truncnorm_pdf,
            truncnorm_logpdf,
        )

        # Test chi-square log-PDF against scipy
        x = np.array([0.5, 1.0, 2.0, 5.0])
        df = 3

        custom_result = chisquare_logpdf(x, df)
        scipy_result = scipy.stats.chi2.logpdf(x, df)

        if np.allclose(custom_result, scipy_result, rtol=1e-10):
            print("✅ Chi-square log-PDF matches scipy")
        else:
            print("❌ Chi-square log-PDF doesn't match scipy")
            print(f"   Custom: {custom_result}")
            print(f"   Scipy:  {scipy_result}")
            return False

        # Test truncated normal PDF against scipy
        x = np.array([-1.0, 0.0, 1.0])
        a, b = -2.0, 2.0

        custom_pdf = truncnorm_pdf(x, a, b)
        scipy_pdf = scipy.stats.truncnorm.pdf(x, a, b)

        if np.allclose(custom_pdf, scipy_pdf, rtol=1e-10):
            print("✅ Truncated normal PDF matches scipy")
        else:
            print("❌ Truncated normal PDF doesn't match scipy")
            return False

        # Test truncated normal log-PDF against scipy
        custom_logpdf = truncnorm_logpdf(x, a, b)
        scipy_logpdf = scipy.stats.truncnorm.logpdf(x, a, b)

        if np.allclose(custom_logpdf, scipy_logpdf, rtol=1e-10):
            print("✅ Truncated normal log-PDF matches scipy")
        else:
            print("❌ Truncated normal log-PDF doesn't match scipy")
            return False

        return True

    except ImportError:
        print("⚠️  scipy not available for reference comparison")
        return True  # Not a failure
    except Exception as e:
        print(f"❌ Scipy comparison error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BRUTUS MATHEMATICAL UTILITIES REORGANIZATION TEST")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_comparison_with_original,
        test_import_structure,
        test_against_scipy,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all(results):
        print("🎉 All tests passed! Mathematical utilities reorganization successful.")
        print("\nNext steps:")
        print(
            "1. Run the full test suite: pytest tests/test_utils/test_math_comprehensive.py"
        )
        print("2. Move on to sampling utilities reorganization")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure you created src/brutus/utils/math.py")
        print("2. Make sure you updated src/brutus/utils/__init__.py")
        print("3. Check for any syntax errors in the files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
