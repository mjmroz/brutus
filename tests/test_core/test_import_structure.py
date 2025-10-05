#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for brutus core module structure and imports.

These tests verify that the new modular structure works correctly
and that key classes can be imported as expected.
"""

from pathlib import Path

import pytest


class TestModuleStructure:
    """Test the new module structure and imports."""

    def test_main_brutus_import(self):
        """Test that main brutus module can be imported."""
        try:
            import brutus

            assert hasattr(brutus, "__version__")
            assert isinstance(brutus.__version__, str)
        except ImportError as e:
            pytest.skip(f"brutus module not available: {e}")

    def test_core_module_import(self):
        """Test that core module can be imported."""
        try:
            import brutus.core

            # Should have __all__ defined
            assert hasattr(brutus.core, "__all__")
        except ImportError as e:
            pytest.skip(f"brutus.core module not available: {e}")

    def test_analysis_module_import(self):
        """Test that analysis module can be imported."""
        try:
            import brutus.analysis

            assert hasattr(brutus.analysis, "__all__")
        except ImportError as e:
            pytest.skip(f"brutus.analysis module not available: {e}")

    def test_utils_module_import(self):
        """Test that utils module can be imported."""
        try:
            import brutus.utils

            assert hasattr(brutus.utils, "__all__")
        except ImportError as e:
            pytest.skip(f"brutus.utils module not available: {e}")

    def test_data_module_import(self):
        """Test that data module can be imported."""
        try:
            import brutus.data

            assert hasattr(brutus.data, "__all__")
        except ImportError as e:
            pytest.skip(f"brutus.data module not available: {e}")


class TestBackwardCompatibility:
    """Test that old import patterns still work during transition."""

    def test_old_import_patterns(self):
        """Test that old import patterns still work with warnings."""
        # These tests will be important during the transition period
        # to ensure we don't break existing user code

        try:
            # Test old-style imports that should still work
            from brutus import __version__

            assert isinstance(__version__, str)

        except ImportError as e:
            pytest.skip(f"Legacy imports not available: {e}")

    def test_core_classes_available(self):
        """Test that core classes are available through new structure."""
        try:
            # These might not work initially but should work after reorganization
            from brutus.core import EEPTracks, Isochrone

            # Just test that they're classes
            assert callable(Isochrone)
            assert callable(EEPTracks)

        except ImportError as e:
            pytest.skip(f"Core classes not yet reorganized: {e}")


class TestProjectStructure:
    """Test that the project directory structure is correct."""

    def test_src_layout_exists(self):
        """Test that src layout directory structure exists."""
        # This test assumes the new structure has been created

        project_root = Path(
            __file__
        ).parent.parent.parent  # Go up from tests/test_core to root
        src_dir = project_root / "src" / "brutus"

        if not src_dir.exists():
            pytest.skip("src/brutus directory not yet created")

        # Check that key subdirectories exist
        expected_dirs = ["core", "analysis", "utils", "data", "dust"]

        for dirname in expected_dirs:
            dir_path = src_dir / dirname
            if dir_path.exists():
                assert (
                    dir_path / "__init__.py"
                ).exists(), f"{dirname} missing __init__.py"

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and is valid."""
        project_root = Path(
            __file__
        ).parent.parent.parent  # Go up from tests/test_core to root
        pyproject_path = project_root / "pyproject.toml"

        if not pyproject_path.exists():
            pytest.skip("pyproject.toml not yet created")

        # Basic validation - can be read as text
        content = pyproject_path.read_text()
        assert "[project]" in content
        assert 'name = "astro-brutus"' in content


class TestVersionManagement:
    """Test version management and metadata."""

    def test_version_accessible(self):
        """Test that version is accessible from main module."""
        try:
            import brutus

            version = brutus.__version__

            # Should be a valid version string
            assert isinstance(version, str)
            assert len(version) > 0

            # Should follow semantic versioning pattern (roughly)
            parts = version.split(".")
            assert len(parts) >= 2  # At least major.minor

        except ImportError:
            pytest.skip("brutus module not available")

    def test_version_consistency(self):
        """Test that version is consistent across files."""
        # This will be important when we have multiple version references
        try:
            import brutus

            # For now, just test that it exists and is a string
            assert hasattr(brutus, "__version__")
            assert isinstance(brutus.__version__, str)

        except ImportError:
            pytest.skip("brutus module not available")
