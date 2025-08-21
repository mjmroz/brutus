#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the photometry module reorganization.

Run this script after creating the new photometry.py file to verify
that our reorganization is working correctly.
"""

import sys
import numpy as np


def test_basic_functionality():
    """Test basic photometry functionality."""
    print("Testing basic photometry functionality...")

    try:
        # Test imports
        from brutus.utils.photometry import magnitude, inv_magnitude, add_mag

        print("✅ Successfully imported photometry functions")

        # Test magnitude conversion
        flux = np.array([[1.0, 10.0, 0.1]])
        flux_err = np.array([[0.1, 1.0, 0.01]])

        mag, mag_err = magnitude(flux, flux_err)
        print(f"✅ magnitude conversion: flux {flux} -> mag {mag}")

        # Test inverse magnitude conversion
        recovered_flux, recovered_flux_err = inv_magnitude(mag, mag_err)
        print(f"✅ inverse magnitude: mag {mag} -> flux {recovered_flux}")

        # Test roundtrip accuracy
        if np.allclose(recovered_flux, flux, rtol=1e-12):
            print("✅ Roundtrip conversion accurate to machine precision")
        else:
            print("❌ Roundtrip conversion accuracy issue")
            print(f"   Original flux: {flux}")
            print(f"   Recovered flux: {recovered_flux}")
            print(f"   Difference: {np.abs(recovered_flux - flux)}")
            return False

        # Test magnitude addition
        mag1, mag2 = 0.0, 0.0
        combined = add_mag(mag1, mag2)
        expected = -2.5 * np.log10(2.0)  # Two equal fluxes
        if np.isclose(combined, expected):
            print("✅ Magnitude addition working correctly")
        else:
            print(f"❌ Magnitude addition issue: got {combined}, expected {expected}")
            return False

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you've created src/brutus/utils/photometry.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_comparison_with_original():
    """Test comparison with original utils.py functions if available."""
    print("\nTesting comparison with original functions...")

    try:
        # Try to import original functions
        from brutus.utilities import magnitude as orig_magnitude
        from brutus.utilities import inv_magnitude as orig_inv_magnitude
        from brutus.utilities import add_mag as orig_add_mag

        # Import new functions
        from brutus.utils.photometry import magnitude as new_magnitude
        from brutus.utils.photometry import inv_magnitude as new_inv_magnitude
        from brutus.utils.photometry import add_mag as new_add_mag

        print("✅ Both original and new functions available for comparison")

        # Test data
        np.random.seed(42)
        flux = np.random.uniform(0.1, 100.0, (5, 3))
        flux_err = 0.1 * flux
        mag = np.random.uniform(15, 25, (5, 3))
        mag_err = np.random.uniform(0.01, 0.1, (5, 3))

        # Compare magnitude conversion
        orig_mag, orig_mag_err = orig_magnitude(flux, flux_err)
        new_mag, new_mag_err = new_magnitude(flux, flux_err)

        if np.allclose(orig_mag, new_mag, rtol=1e-12) and np.allclose(
            orig_mag_err, new_mag_err, rtol=1e-12
        ):
            print("✅ magnitude functions match exactly")
        else:
            print("❌ magnitude functions don't match")
            return False

        # Compare inverse magnitude conversion
        orig_flux, orig_flux_err = orig_inv_magnitude(mag, mag_err)
        new_flux, new_flux_err = new_inv_magnitude(mag, mag_err)

        if np.allclose(orig_flux, new_flux, rtol=1e-12) and np.allclose(
            orig_flux_err, new_flux_err, rtol=1e-12
        ):
            print("✅ inv_magnitude functions match exactly")
        else:
            print("❌ inv_magnitude functions don't match")
            return False

        # Compare magnitude addition
        mag1 = np.random.uniform(15, 25, 10)
        mag2 = np.random.uniform(15, 25, 10)

        orig_combined = orig_add_mag(mag1, mag2)
        new_combined = new_add_mag(mag1, mag2)

        if np.allclose(orig_combined, new_combined, rtol=1e-12):
            print("✅ add_mag functions match exactly")
        else:
            print("❌ add_mag functions don't match")
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
        from brutus.utils import magnitude, inv_magnitude

        print("✅ Can import photometry functions from brutus.utils")

        # Test importing directly from photometry module
        from brutus.utils.photometry import magnitude as photo_magnitude

        print("✅ Can import directly from brutus.utils.photometry")

        # Test that they're the same function
        assert magnitude is photo_magnitude
        print("✅ Import paths point to same function")

        return True

    except Exception as e:
        print(f"❌ Import structure error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BRUTUS PHOTOMETRY REORGANIZATION TEST")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_comparison_with_original,
        test_import_structure,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all(results):
        print("🎉 All tests passed! Photometry reorganization successful.")
        print("\nNext steps:")
        print(
            "1. Run the full test suite: pytest tests/test_utils/test_photometry_comprehensive.py"
        )
        print("2. Move on to mathematical utilities reorganization")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure you created src/brutus/utils/photometry.py")
        print("2. Make sure you updated src/brutus/utils/__init__.py")
        print("3. Check for any syntax errors in the files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
