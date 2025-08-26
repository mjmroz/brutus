#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for brutus.core.__init__.py module.

This test suite covers the core package initialization, including:
1. Import behavior and error handling for all submodules
2. Dynamic __all__ list construction based on available imports
3. Integration with core submodules (sed_utils, neural_nets, etc.)
"""

import pytest
import sys
from unittest.mock import patch
import importlib


class TestCorePackageInit:
    """Test core package initialization behavior."""

    def test_successful_import(self):
        """Test that core package imports successfully."""
        # Direct import should work
        import brutus.core

        # Check basic module structure
        assert hasattr(brutus.core, "__all__")
        assert isinstance(brutus.core.__all__, list)

    def test_sed_utils_imports(self):
        """Test that SED utilities are correctly imported."""
        import brutus.core

        # Check if SED functions are available
        if "_get_seds" in brutus.core.__all__:
            assert hasattr(brutus.core, "_get_seds")
            assert hasattr(brutus.core, "get_seds")
            assert callable(brutus.core._get_seds)
            assert callable(brutus.core.get_seds)
        else:
            # If not in __all__, should not be available or should be None
            assert (
                not hasattr(brutus.core, "_get_seds")
                or getattr(brutus.core, "_get_seds", None) is None
            )

    def test_neural_net_imports(self):
        """Test that neural network classes are correctly imported."""
        import brutus.core

        # Check if neural net classes are available
        if "FastNN" in brutus.core.__all__:
            assert hasattr(brutus.core, "FastNN")
            assert hasattr(brutus.core, "FastNNPredictor")
            assert brutus.core.FastNN is not None
            assert brutus.core.FastNNPredictor is not None
        else:
            # If not in __all__, should not be available or should be None
            assert (
                not hasattr(brutus.core, "FastNN")
                or getattr(brutus.core, "FastNN", None) is None
            )

    def test_tracks_imports(self):
        """Test that tracks classes are correctly imported."""
        import brutus.core

        # Check if tracks classes are available
        if "EEPTracks" in brutus.core.__all__:
            assert hasattr(brutus.core, "EEPTracks")
            assert brutus.core.EEPTracks is not None
        else:
            # If not in __all__, should not be available or should be None
            assert (
                not hasattr(brutus.core, "EEPTracks")
                or getattr(brutus.core, "EEPTracks", None) is None
            )

    def test_isochrones_imports(self):
        """Test that isochrones classes are correctly imported."""
        import brutus.core

        # Check if isochrone classes are available
        if "Isochrone" in brutus.core.__all__:
            assert hasattr(brutus.core, "Isochrone")
            assert brutus.core.Isochrone is not None
        else:
            # If not in __all__, should not be available or should be None
            assert (
                not hasattr(brutus.core, "Isochrone")
                or getattr(brutus.core, "Isochrone", None) is None
            )

    def test_dynamic_all_construction(self):
        """Test that __all__ is dynamically constructed based on available imports."""
        import brutus.core

        # __all__ should only contain items that are actually available
        for item in brutus.core.__all__:
            assert hasattr(
                brutus.core, item
            ), f"Item '{item}' in __all__ but not accessible"
            assert getattr(brutus.core, item) is not None, f"Item '{item}' is None"


class TestCoreImportErrorHandling:
    """Test import error handling for various submodules."""

    def test_sed_utils_import_failure(self):
        """Test behavior when sed_utils import fails."""
        # Mock the sed_utils import to fail
        with patch.dict("sys.modules", {"brutus.core.sed_utils": None}):
            # Force reload to trigger the ImportError
            if "brutus.core" in sys.modules:
                del sys.modules["brutus.core"]

            import brutus.core

            # When import fails, the variables should be None
            assert getattr(brutus.core, "_get_seds", "not_found") in [None, "not_found"]
            assert getattr(brutus.core, "get_seds", "not_found") in [None, "not_found"]

            # These should not be in __all__
            assert "_get_seds" not in brutus.core.__all__
            assert "get_seds" not in brutus.core.__all__

    def test_neural_nets_import_failure(self):
        """Test behavior when neural_nets import fails."""
        # Mock the neural_nets import to fail
        with patch.dict("sys.modules", {"brutus.core.neural_nets": None}):
            # Force reload to trigger the ImportError
            if "brutus.core" in sys.modules:
                del sys.modules["brutus.core"]

            import brutus.core

            # When import fails, the variables should be None
            assert getattr(brutus.core, "FastNN", "not_found") in [None, "not_found"]
            assert getattr(brutus.core, "FastNNPredictor", "not_found") in [
                None,
                "not_found",
            ]

            # These should not be in __all__
            assert "FastNN" not in brutus.core.__all__
            assert "FastNNPredictor" not in brutus.core.__all__

    def test_individual_import_failure(self):
        """Test behavior when individual import fails."""
        # Mock the individual import to fail
        with patch.dict("sys.modules", {"brutus.core.individual": None}):
            # Force reload to trigger the ImportError
            if "brutus.core" in sys.modules:
                del sys.modules["brutus.core"]

            import brutus.core

            # When import fails, the variable should be None
            assert getattr(brutus.core, "EEPTracks", "not_found") in [None, "not_found"]

            # This should not be in __all__
            assert "EEPTracks" not in brutus.core.__all__

    def test_populations_import_failure(self):
        """Test behavior when populations import fails."""
        # Mock the populations import to fail
        with patch.dict("sys.modules", {"brutus.core.populations": None}):
            # Force reload to trigger the ImportError
            if "brutus.core" in sys.modules:
                del sys.modules["brutus.core"]

            import brutus.core

            # When import fails, the variable should be None
            assert getattr(brutus.core, "Isochrone", "not_found") in [None, "not_found"]

            # This should not be in __all__
            assert "Isochrone" not in brutus.core.__all__


class TestCoreIntegration:
    """Test integration with other brutus components."""

    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import brutus.core

        assert brutus.core.__doc__ is not None
        assert len(brutus.core.__doc__.strip()) > 0
        assert "core module" in brutus.core.__doc__.lower()
        assert "stellar evolution modeling" in brutus.core.__doc__.lower()

    def test_import_from_core(self):
        """Test importing specific items from core."""
        import brutus.core

        # Test that we can import available items
        for item in brutus.core.__all__:
            # Dynamic import test
            exec(f"from brutus.core import {item}")
            # Should not raise an exception

    def test_all_list_consistency(self):
        """Test that __all__ list is consistent with actual attributes."""
        import brutus.core

        # Every item in __all__ should be accessible
        for item in brutus.core.__all__:
            assert hasattr(
                brutus.core, item
            ), f"Item '{item}' in __all__ but not available"
            assert getattr(brutus.core, item) is not None, f"Item '{item}' is None"

        # __all__ should be a list and contain no duplicates
        assert isinstance(brutus.core.__all__, list)
        assert len(brutus.core.__all__) == len(
            set(brutus.core.__all__)
        ), "Duplicates in __all__"


if __name__ == "__main__":
    pytest.main([__file__])
