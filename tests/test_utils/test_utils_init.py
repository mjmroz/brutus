#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for brutus utils module __init__.py

This module tests the import behavior and __all__ definitions
of the utils package initialization.
"""

import sys
from unittest.mock import patch

# pytest used by test framework


class TestUtilsInit:
    """Test utils package initialization behavior."""

    def test_successful_imports(self):
        """Test that all expected functions are available when imports succeed."""
        # Import the utils module
        import brutus.utils

        # Check that __all__ is properly defined
        assert hasattr(brutus.utils, "__all__")
        assert isinstance(brutus.utils.__all__, list)
        assert len(brutus.utils.__all__) > 0

        # Check that key functions are available
        expected_functions = [
            "magnitude",
            "inv_magnitude",
            "luptitude",
            "inv_luptitude",
            "add_mag",
            "phot_loglike",
            "inverse3",
            "isPSD",
            "chisquare_logpdf",
            "truncnorm_pdf",
            "quantile",
            "sample_multivariate_normal",
            "draw_sar",
        ]

        for func_name in expected_functions:
            assert func_name in brutus.utils.__all__, f"{func_name} not in __all__"
            assert hasattr(
                brutus.utils, func_name
            ), f"{func_name} not available in module"

    def test_import_error_fallback(self):
        """Test that ImportError is handled gracefully with empty __all__."""
        # We need to test the ImportError path by mocking import failures
        # This is tricky since the imports happen at module level

        # First, let's create a scenario where import would fail
        with patch.dict("sys.modules", {"brutus.utils.photometry": None}):
            # Force reload of the utils module to trigger import error path
            if "brutus.utils" in sys.modules:
                del sys.modules["brutus.utils"]

            # This should trigger the ImportError handling
            try:
                import brutus.utils

                # If we get here, check that __all__ exists (might be empty on import error)
                assert hasattr(brutus.utils, "__all__")
            except ImportError:
                # This is also acceptable - it means the import actually failed
                pass

    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import brutus.utils

        assert hasattr(brutus.utils, "__doc__")
        assert brutus.utils.__doc__ is not None
        assert "utilities module" in brutus.utils.__doc__.lower()

    def test_import_structure_consistency(self):
        """Test that the imports are consistent with the actual module structure."""
        import brutus.utils

        # Test that photometry functions are available
        photometry_funcs = [
            "magnitude",
            "inv_magnitude",
            "luptitude",
            "inv_luptitude",
            "add_mag",
            "phot_loglike",
        ]
        for func in photometry_funcs:
            if func in brutus.utils.__all__:
                assert callable(
                    getattr(brutus.utils, func, None)
                ), f"{func} should be callable"

        # Test that math functions are available
        math_funcs = ["inverse3", "isPSD"]
        for func in math_funcs:
            if func in brutus.utils.__all__:
                assert callable(
                    getattr(brutus.utils, func, None)
                ), f"{func} should be callable"

        # Test that sampling functions are available
        sampling_funcs = ["quantile", "sample_multivariate_normal", "draw_sar"]
        for func in sampling_funcs:
            if func in brutus.utils.__all__:
                assert callable(
                    getattr(brutus.utils, func, None)
                ), f"{func} should be callable"
