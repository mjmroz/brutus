#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the data functions module reorganization.

Run this script after creating the new data modules to verify
that our reorganization is working correctly.
"""

import sys
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_basic_functionality():
    """Test basic data functionality."""
    print("Testing basic data functionality...")

    try:
        # Test imports
        from brutus.data.download import (
            fetch_isos,
            fetch_tracks,
            fetch_dustmaps,
            fetch_grids,
            fetch_offsets,
            fetch_nns,
        )
        from brutus.data.loader import load_models, load_offsets

        print("✅ Successfully imported data functions")

        # Test that all functions are callable
        download_funcs = [
            fetch_isos,
            fetch_tracks,
            fetch_dustmaps,
            fetch_grids,
            fetch_offsets,
            fetch_nns,
        ]
        loader_funcs = [load_models, load_offsets]

        for func in download_funcs + loader_funcs:
            if not callable(func):
                print(f"❌ {func.__name__} is not callable")
                return False

        print("✅ All data functions are callable")

        # Test download function parameter validation
        try:
            fetch_isos(iso="invalid_iso")
            print("❌ fetch_isos should have raised ValueError for invalid iso")
            return False
        except ValueError as e:
            if "does not exist" in str(e):
                print("✅ fetch_isos correctly validates iso parameter")
            else:
                print(f"❌ fetch_isos unexpected error: {e}")
                return False

        try:
            fetch_grids(grid="invalid_grid")
            print("❌ fetch_grids should have raised ValueError for invalid grid")
            return False
        except ValueError as e:
            if "does not exist" in str(e):
                print("✅ fetch_grids correctly validates grid parameter")
            else:
                print(f"❌ fetch_grids unexpected error: {e}")
                return False

        # Test download functions with mocking
        with patch("brutus.data.download._fetch") as mock_fetch:
            mock_fetch.return_value = Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5")

            result = fetch_isos(target_dir="/tmp")

            if result == Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5"):
                print("✅ fetch_isos works with mocked _fetch")
            else:
                print(f"❌ fetch_isos returned unexpected result: {result}")
                return False

            # Check that _fetch was called with correct parameters
            mock_fetch.assert_called_once_with("MIST_1.2_iso_vvcrit0.0.h5", "/tmp")

        # Test load_offsets with mocking
        with patch("numpy.loadtxt") as mock_loadtxt:
            mock_loadtxt.return_value = (
                np.array(["g", "r", "i"], dtype="str"),
                np.array(["1.02", "0.98", "1.01"], dtype="str"),
            )

            with patch("brutus.data.loader.sys.stderr"):
                offsets = load_offsets(
                    "/fake/path.txt", filters=["g", "r", "i"], verbose=False
                )

            expected = np.array([1.02, 0.98, 1.01])
            if np.allclose(offsets, expected):
                print("✅ load_offsets works correctly with mocked data")
            else:
                print(f"❌ load_offsets returned unexpected result: {offsets}")
                return False

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you've created the data module files")
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
            fetch_isos as orig_fetch_isos,
            fetch_grids as orig_fetch_grids,
            load_offsets as orig_load_offsets,
        )

        # Import new functions
        from brutus.data.download import (
            fetch_isos as new_fetch_isos,
            fetch_grids as new_fetch_grids,
        )
        from brutus.data.loader import load_offsets as new_load_offsets

        print("✅ Both original and new functions available for comparison")

        # Test fetch_isos comparison with mocking
        with patch("brutus.data.download._fetch") as new_mock_fetch:
            with patch("brutus.utilities._fetch") as orig_mock_fetch:
                mock_path = Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5")
                new_mock_fetch.return_value = mock_path
                orig_mock_fetch.return_value = mock_path

                orig_result = orig_fetch_isos(target_dir="/tmp")
                new_result = new_fetch_isos(target_dir="/tmp")

                if orig_result == new_result:
                    print("✅ fetch_isos functions return identical results")
                else:
                    print("❌ fetch_isos functions return different results")
                    return False

        # Test load_offsets comparison with mocking
        with patch("numpy.loadtxt") as mock_loadtxt:
            mock_loadtxt.return_value = (
                np.array(["g", "r", "i"], dtype="str"),
                np.array(["1.02", "0.98", "1.01"], dtype="str"),
            )

            filters = ["g", "r", "i"]

            with patch("brutus.data.loader.sys.stderr"):
                with patch("brutus.utilities.sys.stderr"):
                    orig_result = orig_load_offsets(
                        "/fake/path.txt", filters=filters, verbose=False
                    )
                    new_result = new_load_offsets(
                        "/fake/path.txt", filters=filters, verbose=False
                    )

            if np.allclose(orig_result, new_result, rtol=1e-12):
                print("✅ load_offsets functions match exactly")
            else:
                print("❌ load_offsets functions don't match")
                print(f"   Original: {orig_result}")
                print(f"   New: {new_result}")
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
        # Test importing from data package
        from brutus.data import fetch_isos, load_models

        print("✅ Can import data functions from brutus.data")

        # Test importing directly from submodules
        from brutus.data.download import fetch_isos as download_fetch_isos
        from brutus.data.loader import load_models as loader_load_models

        print("✅ Can import directly from data submodules")

        # Test that they're the same function
        assert fetch_isos is download_fetch_isos
        assert load_models is loader_load_models
        print("✅ Import paths point to same functions")

        # Test __all__ is properly defined
        import brutus.data as data_module

        if hasattr(data_module, "__all__"):
            print("✅ data module has __all__ defined")

            # Check that all listed functions are available
            for name in data_module.__all__:
                if not hasattr(data_module, name):
                    print(f"❌ Function {name} in __all__ but not available")
                    return False
                if not callable(getattr(data_module, name)):
                    print(f"❌ {name} is not callable")
                    return False
            print("✅ All functions in __all__ are available and callable")
        else:
            print("❌ data module missing __all__")
            return False

        return True

    except Exception as e:
        print(f"❌ Import structure error: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\nTesting error handling...")

    try:
        from brutus.data.download import fetch_dustmaps, fetch_nns
        from brutus.data.loader import load_models

        # Test invalid parameters for all download functions
        invalid_params = [
            (fetch_dustmaps, {"dustmap": "invalid"}),
            (fetch_nns, {"model": "invalid"}),
        ]

        for func, kwargs in invalid_params:
            try:
                func(**kwargs)
                print(f"❌ {func.__name__} should have raised ValueError")
                return False
            except ValueError as e:
                if "does not exist" in str(e):
                    print(f"✅ {func.__name__} correctly validates parameters")
                else:
                    print(f"❌ {func.__name__} unexpected error: {e}")
                    return False

        # Test load_models with non-existent file
        try:
            load_models("/completely/non/existent/file.h5")
            print("❌ load_models should have raised error for non-existent file")
            return False
        except (FileNotFoundError, OSError):
            print("✅ load_models correctly handles non-existent files")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_mock_workflow():
    """Test a realistic workflow with mocked functions."""
    print("\nTesting mock workflow...")

    try:
        from brutus.data import fetch_grids, fetch_offsets, load_models, load_offsets

        # Mock a complete workflow
        with patch("brutus.data.download._fetch") as mock_fetch:
            # Mock downloading grid and offsets
            mock_fetch.side_effect = [
                Path("/fake/grid_mist_v9.h5"),
                Path("/fake/offsets_mist_v9.txt"),
            ]

            # Download files
            grid_path = fetch_grids(target_dir="/data", grid="mist_v9")
            offset_path = fetch_offsets(target_dir="/data", grid="mist_v9")

            print("✅ Successfully downloaded (mocked) grid and offset files")

            # Verify paths
            if grid_path != Path("/fake/grid_mist_v9.h5"):
                print(f"❌ Unexpected grid path: {grid_path}")
                return False
            if offset_path != Path("/fake/offsets_mist_v9.txt"):
                print(f"❌ Unexpected offset path: {offset_path}")
                return False

            print("✅ Download paths are correct")

        # Mock loading the files
        with patch("h5py.File") as mock_h5:
            with patch("numpy.loadtxt") as mock_loadtxt:
                # Mock offset file content
                mock_loadtxt.return_value = (
                    np.array(["g", "r", "i"], dtype="str"),
                    np.array(["1.00", "1.00", "1.00"], dtype="str"),
                )

                with patch("brutus.data.loader.sys.stderr"):
                    offsets = load_offsets(
                        "/fake/offsets_mist_v9.txt",
                        filters=["g", "r", "i"],
                        verbose=False,
                    )

                if len(offsets) == 3 and np.allclose(offsets, 1.0):
                    print("✅ Successfully loaded (mocked) offset data")
                else:
                    print(f"❌ Unexpected offset data: {offsets}")
                    return False

        return True

    except Exception as e:
        print(f"❌ Mock workflow error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BRUTUS DATA FUNCTIONS REORGANIZATION TEST")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_comparison_with_original,
        test_import_structure,
        test_error_handling,
        test_mock_workflow,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all(results):
        print("🎉 All tests passed! Data functions reorganization successful.")
        print("\nNext steps:")
        print(
            "1. Run the full test suite: pytest tests/test_data/test_data_comprehensive.py"
        )
        print("2. Move on to SED utilities reorganization (final step!)")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure you created src/brutus/data/download.py")
        print("2. Make sure you created src/brutus/data/loader.py")
        print("3. Make sure you created src/brutus/data/__init__.py")
        print("4. Check for any syntax errors in the files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
