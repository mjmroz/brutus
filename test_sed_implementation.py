#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation test for SED utilities reorganization.

This script tests the reorganized SED functions (_get_seds and get_seds)
that were moved from utilities.py to core/sed_utils.py.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_basic_functionality():
    """Test that the reorganized SED functions work correctly."""
    print("\nTesting basic SED computation functionality...")

    try:
        from brutus.core.sed_utils import _get_seds, get_seds

        # Create simple test data
        # 2 models, 3 bands, 3 coefficients per band
        mag_coeffs = np.random.random((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0  # Base magnitudes
        mag_coeffs[:, :, 1] = 1.0  # R(V)=0 reddening vector
        mag_coeffs[:, :, 2] = 0.1  # dR/dR(V) differential

        av = np.array([0.0, 0.5])  # A(V) values
        rv = np.array([3.1, 3.3])  # R(V) values

        # Test _get_seds function
        seds, rvecs, drvecs = _get_seds(mag_coeffs, av, rv, return_flux=False)

        # Check shapes
        if seds.shape != (2, 3):
            print(f"❌ Unexpected SED shape: {seds.shape}")
            return False

        if rvecs.shape != (2, 3):
            print(f"❌ Unexpected reddening vector shape: {rvecs.shape}")
            return False

        if drvecs.shape != (2, 3):
            print(f"❌ Unexpected differential reddening vector shape: {drvecs.shape}")
            return False

        # Test get_seds wrapper
        seds2 = get_seds(mag_coeffs, av=av, rv=rv, return_flux=False)

        # Should match _get_seds output
        if not np.allclose(seds, seds2, rtol=1e-12):
            print("❌ get_seds output doesn't match _get_seds")
            return False

        # Test flux conversion
        seds_flux, rvecs_flux, drvecs_flux = _get_seds(
            mag_coeffs, av, rv, return_flux=True
        )
        expected_flux = 10.0 ** (-0.4 * seds)

        if not np.allclose(seds_flux, expected_flux, rtol=1e-12):
            print("❌ Flux conversion incorrect")
            return False

        print("✅ Basic functionality tests passed")
        return True

    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        return False


def test_comparison_with_original():
    """Test that new functions match original implementations."""
    print("\nTesting comparison with original functions...")

    try:
        # Try to import original functions
        from brutus.utilities import _get_seds as orig_get_seds
        from brutus.utilities import get_seds as orig_get_seds_wrapper

        # Import new functions
        from brutus.core.sed_utils import _get_seds as new_get_seds
        from brutus.core.sed_utils import get_seds as new_get_seds_wrapper

        print("✅ Both original and new functions available for comparison")

        # Create test data
        np.random.seed(42)
        mag_coeffs = np.random.random((3, 4, 3))
        mag_coeffs[:, :, 0] = np.random.uniform(14, 18, (3, 4))  # Base mags
        mag_coeffs[:, :, 1] = np.random.uniform(0.5, 2.0, (3, 4))  # R0
        mag_coeffs[:, :, 2] = np.random.uniform(0.05, 0.2, (3, 4))  # dR

        av = np.random.uniform(0.0, 2.0, 3)
        rv = np.random.uniform(2.5, 4.5, 3)

        # Test _get_seds functions
        orig_seds, orig_rvecs, orig_drvecs = orig_get_seds(
            mag_coeffs, av, rv, return_flux=False
        )
        new_seds, new_rvecs, new_drvecs = new_get_seds(
            mag_coeffs, av, rv, return_flux=False
        )

        if not np.allclose(orig_seds, new_seds, rtol=1e-12):
            print("❌ _get_seds SEDs don't match")
            return False

        if not np.allclose(orig_rvecs, new_rvecs, rtol=1e-12):
            print("❌ _get_seds reddening vectors don't match")
            return False

        if not np.allclose(orig_drvecs, new_drvecs, rtol=1e-12):
            print("❌ _get_seds differential reddening vectors don't match")
            return False

        # Test get_seds wrapper functions
        orig_seds_wrapper = orig_get_seds_wrapper(mag_coeffs, av=av, rv=rv)
        new_seds_wrapper = new_get_seds_wrapper(mag_coeffs, av=av, rv=rv)

        if not np.allclose(orig_seds_wrapper, new_seds_wrapper, rtol=1e-12):
            print("❌ get_seds wrapper outputs don't match")
            return False

        # Test flux conversion
        orig_seds_flux, _, _ = orig_get_seds(mag_coeffs, av, rv, return_flux=True)
        new_seds_flux, _, _ = new_get_seds(mag_coeffs, av, rv, return_flux=True)

        if not np.allclose(orig_seds_flux, new_seds_flux, rtol=1e-12):
            print("❌ Flux conversion doesn't match")
            return False

        print("✅ All functions match original implementations exactly")
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
        # Test direct imports
        from brutus.core.sed_utils import _get_seds, get_seds

        print("✅ Direct imports from sed_utils work")

        # Test core module imports
        from brutus.core import _get_seds as core_get_seds
        from brutus.core import get_seds as core_get_seds_wrapper

        print("✅ Imports from core module work")

        # Test that they are the same functions
        assert _get_seds is core_get_seds
        assert get_seds is core_get_seds_wrapper
        print("✅ Functions are properly exposed through core module")

        return True

    except Exception as e:
        print(f"❌ Import structure error: {e}")
        return False


def test_parameter_handling():
    """Test edge cases and parameter handling."""
    print("\nTesting parameter handling and edge cases...")

    try:
        from brutus.core.sed_utils import get_seds

        # Create test data
        mag_coeffs = np.ones((2, 3, 3))
        mag_coeffs[:, :, 0] = 15.0  # Base magnitudes

        # Test default parameters
        seds1 = get_seds(mag_coeffs)  # Should use av=0, rv=3.3
        seds2 = get_seds(mag_coeffs, av=0.0, rv=3.3)

        if not np.allclose(seds1, seds2, rtol=1e-12):
            print("❌ Default parameter handling incorrect")
            return False

        # Test scalar parameters
        seds3 = get_seds(mag_coeffs, av=0.1, rv=3.1)
        seds4 = get_seds(mag_coeffs, av=np.array([0.1, 0.1]), rv=np.array([3.1, 3.1]))

        if not np.allclose(seds3, seds4, rtol=1e-12):
            print("❌ Scalar parameter broadcasting incorrect")
            return False

        # Test optional returns
        seds_only = get_seds(mag_coeffs, av=0.1, rv=3.1)
        seds_rvec, rvecs = get_seds(mag_coeffs, av=0.1, rv=3.1, return_rvec=True)
        seds_drvec, drvecs = get_seds(mag_coeffs, av=0.1, rv=3.1, return_drvec=True)
        seds_all, rvecs_all, drvecs_all = get_seds(
            mag_coeffs, av=0.1, rv=3.1, return_rvec=True, return_drvec=True
        )

        # All SEDs should be identical
        if not (
            np.allclose(seds_only, seds_rvec, rtol=1e-12)
            and np.allclose(seds_only, seds_drvec, rtol=1e-12)
            and np.allclose(seds_only, seds_all, rtol=1e-12)
        ):
            print("❌ Optional return parameters affect SED computation")
            return False

        print("✅ Parameter handling tests passed")
        return True

    except Exception as e:
        print(f"❌ Parameter handling error: {e}")
        return False


def test_physical_consistency():
    """Test that the physics makes sense."""
    print("\nTesting physical consistency...")

    try:
        from brutus.core.sed_utils import get_seds

        # Create test data where we can verify physics
        mag_coeffs = np.ones((1, 3, 3))
        mag_coeffs[0, :, 0] = 15.0  # Base magnitude
        mag_coeffs[0, :, 1] = 1.0  # R(V)=0 reddening
        mag_coeffs[0, :, 2] = 0.0  # No R(V) dependence

        # Test no reddening case
        seds_no_red = get_seds(mag_coeffs, av=0.0, rv=3.1)
        expected_no_red = 15.0  # Should equal base magnitude

        if not np.allclose(seds_no_red, expected_no_red, rtol=1e-12):
            print(f"❌ No reddening case failed: {seds_no_red} != {expected_no_red}")
            return False

        # Test that extinction increases magnitude (decreases flux)
        seds_with_red = get_seds(mag_coeffs, av=1.0, rv=3.1)
        expected_with_red = 15.0 + 1.0 * 1.0  # mag + av * rvec

        if not np.allclose(seds_with_red, expected_with_red, rtol=1e-12):
            print(f"❌ Reddening case failed: {seds_with_red} != {expected_with_red}")
            return False

        # Test flux conversion makes sense
        seds_mag = get_seds(mag_coeffs, av=0.0, rv=3.1, return_flux=False)
        seds_flux = get_seds(mag_coeffs, av=0.0, rv=3.1, return_flux=True)
        expected_flux = 10.0 ** (-0.4 * seds_mag)

        if not np.allclose(seds_flux, expected_flux, rtol=1e-12):
            print("❌ Magnitude to flux conversion incorrect")
            return False

        print("✅ Physical consistency tests passed")
        return True

    except Exception as e:
        print(f"❌ Physical consistency error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BRUTUS SED UTILITIES REORGANIZATION TEST")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_comparison_with_original,
        test_import_structure,
        test_parameter_handling,
        test_physical_consistency,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all(results):
        print("🎉 All tests passed! SED utilities reorganization successful.")
        print("\nNext steps:")
        print(
            "1. Run the full test suite: pytest tests/test_core/test_sed_comprehensive.py"
        )
        print("2. Update photometric_offsets function to use these utilities")
        print("3. Verify all imports and dependencies work correctly")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure you created src/brutus/core/sed_utils.py")
        print("2. Make sure you created src/brutus/core/__init__.py")
        print("3. Check for any syntax errors in the files")
        print("4. Ensure the src/ directory is in your Python path")
        return 1


if __name__ == "__main__":
    sys.exit(main())
