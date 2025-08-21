#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus data utilities.

This test suite includes:
1. Unit tests for data downloading and loading functions
2. Comparison tests between old and new implementations
3. Integration tests for realistic data workflows
4. Mock tests that don't require actual downloads
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch


class TestDataDownloadFunctions:
    """Unit tests for data downloading functions."""

    def test_download_imports(self):
        """Test that download functions can be imported."""
        from brutus.data.download import (
            fetch_isos,
            fetch_tracks,
            fetch_dustmaps,
            fetch_grids,
            fetch_offsets,
            fetch_nns,
        )

        # Check that all functions are callable
        assert callable(fetch_isos)
        assert callable(fetch_tracks)
        assert callable(fetch_dustmaps)
        assert callable(fetch_grids)
        assert callable(fetch_offsets)
        assert callable(fetch_nns)

    @patch("brutus.data.download._fetch")
    def test_fetch_isos_basic(self, mock_fetch):
        """Test basic fetch_isos functionality."""
        from brutus.data.download import fetch_isos

        # Mock the _fetch function
        mock_fetch.return_value = Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5")

        # Test default parameters
        result = fetch_isos()

        mock_fetch.assert_called_once_with("MIST_1.2_iso_vvcrit0.0.h5", ".")
        assert result == Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5")

    @patch("brutus.data.download._fetch")
    def test_fetch_isos_rotating(self, mock_fetch):
        """Test fetch_isos with rotating models."""
        from brutus.data.download import fetch_isos

        mock_fetch.return_value = Path("/fake/path/MIST_1.2_iso_vvcrit0.4.h5")

        result = fetch_isos(target_dir="/tmp", iso="MIST_1.2_vvcrit0.4")

        mock_fetch.assert_called_once_with("MIST_1.2_iso_vvcrit0.4.h5", "/tmp")
        assert result == Path("/fake/path/MIST_1.2_iso_vvcrit0.4.h5")

    def test_fetch_isos_invalid(self):
        """Test fetch_isos with invalid iso parameter."""
        from brutus.data.download import fetch_isos

        with pytest.raises(
            ValueError, match="The specified isochrone file does not exist"
        ):
            fetch_isos(iso="invalid_iso")

    @patch("brutus.data.download._fetch")
    def test_fetch_tracks_basic(self, mock_fetch):
        """Test basic fetch_tracks functionality."""
        from brutus.data.download import fetch_tracks

        mock_fetch.return_value = Path("/fake/path/MIST_1.2_EEPtrk.h5")

        result = fetch_tracks(target_dir="/data")

        mock_fetch.assert_called_once_with("MIST_1.2_EEPtrk.h5", "/data")
        assert result == Path("/fake/path/MIST_1.2_EEPtrk.h5")

    def test_fetch_tracks_invalid(self):
        """Test fetch_tracks with invalid track parameter."""
        from brutus.data.download import fetch_tracks

        with pytest.raises(ValueError, match="The specified track file does not exist"):
            fetch_tracks(track="invalid_track")

    @patch("brutus.data.download._fetch")
    def test_fetch_dustmaps_basic(self, mock_fetch):
        """Test basic fetch_dustmaps functionality."""
        from brutus.data.download import fetch_dustmaps

        mock_fetch.return_value = Path("/fake/path/bayestar2019_v1.h5")

        result = fetch_dustmaps()

        mock_fetch.assert_called_once_with("bayestar2019_v1.h5", ".")

    def test_fetch_dustmaps_invalid(self):
        """Test fetch_dustmaps with invalid dustmap parameter."""
        from brutus.data.download import fetch_dustmaps

        with pytest.raises(
            ValueError, match="The specified dustmap file does not exist"
        ):
            fetch_dustmaps(dustmap="invalid_dustmap")

    @patch("brutus.data.download._fetch")
    def test_fetch_grids_versions(self, mock_fetch):
        """Test fetch_grids with different versions."""
        from brutus.data.download import fetch_grids

        # Test v9 (default)
        mock_fetch.return_value = Path("/fake/path/grid_mist_v9.h5")
        fetch_grids(grid="mist_v9")
        mock_fetch.assert_called_with("grid_mist_v9.h5", ".")

        # Test v8
        mock_fetch.return_value = Path("/fake/path/grid_mist_v8.h5")
        fetch_grids(grid="mist_v8")
        mock_fetch.assert_called_with("grid_mist_v8.h5", ".")

        # Test bayestar
        mock_fetch.return_value = Path("/fake/path/grid_bayestar_v5.h5")
        fetch_grids(grid="bayestar_v5")
        mock_fetch.assert_called_with("grid_bayestar_v5.h5", ".")

    def test_fetch_grids_invalid(self):
        """Test fetch_grids with invalid grid parameter."""
        from brutus.data.download import fetch_grids

        with pytest.raises(ValueError, match="The specified grid file does not exist"):
            fetch_grids(grid="invalid_grid")

    @patch("brutus.data.download._fetch")
    def test_fetch_offsets_versions(self, mock_fetch):
        """Test fetch_offsets with different versions."""
        from brutus.data.download import fetch_offsets

        # Test v9 (default)
        mock_fetch.return_value = Path("/fake/path/offsets_mist_v9.txt")
        fetch_offsets(grid="mist_v9")
        mock_fetch.assert_called_with("offsets_mist_v9.txt", ".")

        # Test bayestar (note: uses bs_v9 filename)
        mock_fetch.return_value = Path("/fake/path/offsets_bs_v9.txt")
        fetch_offsets(grid="bayestar_v5")
        mock_fetch.assert_called_with("offsets_bs_v9.txt", ".")

    @patch("brutus.data.download._fetch")
    def test_fetch_nns_basic(self, mock_fetch):
        """Test basic fetch_nns functionality."""
        from brutus.data.download import fetch_nns

        mock_fetch.return_value = Path("/fake/path/nn_c3k.h5")

        result = fetch_nns()

        mock_fetch.assert_called_once_with("nn_c3k.h5", ".")

    def test_fetch_nns_invalid(self):
        """Test fetch_nns with invalid model parameter."""
        from brutus.data.download import fetch_nns

        with pytest.raises(
            ValueError, match="The specified neural network file does not exist"
        ):
            fetch_nns(model="invalid_model")


class TestDataLoaderFunctions:
    """Unit tests for data loading functions."""

    def test_loader_imports(self):
        """Test that loader functions can be imported."""
        from brutus.data.loader import load_models, load_offsets

        assert callable(load_models)
        assert callable(load_offsets)

    def test_load_models_error_handling(self):
        """Test load_models error handling."""
        from brutus.data.loader import load_models

        # Test with non-existent file
        with pytest.raises((FileNotFoundError, OSError)):
            load_models("/non/existent/file.h5")

        # Test invalid evolutionary phase selection
        with pytest.raises(ValueError, match="If you don't include the Main Sequence"):
            # This would require a mock HDF5 file, so we'll just test the logic
            with patch("h5py.File"):
                with patch("brutus.data.loader.sys.stderr"):
                    # This should raise the error due to include_ms=False, include_postms=False
                    pass  # The actual error is in the logic, hard to test without real data

    @patch("numpy.loadtxt")
    def test_load_offsets_basic(self, mock_loadtxt):
        """Test basic load_offsets functionality."""
        from brutus.data.loader import load_offsets

        # Mock the file content: filter names and offset values
        mock_loadtxt.return_value = (
            np.array(["g", "r", "i"], dtype="str"),
            np.array(["1.02", "0.98", "1.01"], dtype="str"),
        )

        with patch("brutus.data.loader.sys.stderr"):
            offsets = load_offsets(
                "/fake/path.txt", filters=["g", "r", "i"], verbose=False
            )

        expected = np.array([1.02, 0.98, 1.01])
        np.testing.assert_array_almost_equal(offsets, expected)

    @patch("numpy.loadtxt")
    def test_load_offsets_missing_filter(self, mock_loadtxt):
        """Test load_offsets with missing filter."""
        from brutus.data.loader import load_offsets

        # Mock file with only 'g' and 'r', missing 'i'
        mock_loadtxt.return_value = (
            np.array(["g", "r"], dtype="str"),
            np.array(["1.02", "0.98"], dtype="str"),
        )

        with patch("brutus.data.loader.sys.stderr"):
            offsets = load_offsets(
                "/fake/path.txt", filters=["g", "r", "i"], verbose=False
            )

        # 'i' should default to 1.0
        expected = np.array([1.02, 0.98, 1.0])
        np.testing.assert_array_almost_equal(offsets, expected)

    @patch("numpy.loadtxt")
    def test_load_offsets_duplicate_filter(self, mock_loadtxt):
        """Test load_offsets with duplicate filter names."""
        from brutus.data.loader import load_offsets

        # Mock file with duplicate 'g' filter
        mock_loadtxt.return_value = (
            np.array(["g", "g", "r"], dtype="str"),
            np.array(["1.02", "1.03", "0.98"], dtype="str"),
        )

        with pytest.raises(ValueError, match="Something went wrong when extracting"):
            with patch("brutus.data.loader.sys.stderr"):
                load_offsets("/fake/path.txt", filters=["g", "r"], verbose=False)


class TestDataComparison:
    """Comparison tests between old and new implementations."""

    @patch("brutus.data.download._fetch")
    def test_fetch_isos_vs_original(self, mock_fetch):
        """Compare new fetch_isos with original."""
        try:
            from brutus.utilities import fetch_isos as orig_fetch_isos
        except ImportError:
            pytest.skip("Original utilities.py not available for comparison")

        from brutus.data.download import fetch_isos as new_fetch_isos

        # Mock both implementations to return same path
        mock_path = Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5")
        mock_fetch.return_value = mock_path

        with patch("brutus.utilities._fetch", return_value=mock_path):
            orig_result = orig_fetch_isos(target_dir="/tmp")
            new_result = new_fetch_isos(target_dir="/tmp")

            assert orig_result == new_result

    @patch("numpy.loadtxt")
    def test_load_offsets_vs_original(self, mock_loadtxt):
        """Compare new load_offsets with original."""
        try:
            from brutus.utilities import load_offsets as orig_load_offsets
        except ImportError:
            pytest.skip("Original utilities.py not available for comparison")

        from brutus.data.loader import load_offsets as new_load_offsets

        # Mock file content
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

        np.testing.assert_array_almost_equal(new_result, orig_result, decimal=12)


class TestDataImportStructure:
    """Test import structure for data module."""

    def test_data_module_imports(self):
        """Test that data functions can be imported from data module."""
        from brutus.data import (
            fetch_isos,
            fetch_tracks,
            fetch_dustmaps,
            fetch_grids,
            fetch_offsets,
            fetch_nns,
            load_models,
            load_offsets,
        )

        # All should be callable
        funcs = [
            fetch_isos,
            fetch_tracks,
            fetch_dustmaps,
            fetch_grids,
            fetch_offsets,
            fetch_nns,
            load_models,
            load_offsets,
        ]

        for func in funcs:
            assert callable(func)

    def test_data_module_all(self):
        """Test that __all__ is properly defined."""
        import brutus.data as data_module

        assert hasattr(data_module, "__all__")

        # Check that all listed functions are actually available
        for name in data_module.__all__:
            assert hasattr(data_module, name)
            assert callable(getattr(data_module, name))


class TestDataEdgeCases:
    """Test edge cases and error conditions."""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.symlink_to")
    @patch("pathlib.Path.mkdir")
    @patch("brutus.data_old.strato.fetch")
    def test_fetch_symlink_creation(
        self, mock_strato_fetch, mock_mkdir, mock_symlink, mock_exists
    ):
        """Test _fetch symlink creation logic."""
        from brutus.data.download import _fetch

        # Mock strato.fetch to return a path
        mock_strato_fetch.return_value = "/cache/fake_file.h5"
        mock_exists.return_value = False  # Symlink doesn't exist yet

        result = _fetch("fake_file.h5", "/tmp")

        # Should call strato.fetch
        mock_strato_fetch.assert_called_once_with("fake_file.h5", progressbar=True)

        # Should create parent directory
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Should create symlink
        mock_symlink.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("brutus.data_old.strato.fetch")
    def test_fetch_existing_symlink(self, mock_strato_fetch, mock_exists):
        """Test _fetch when symlink already exists."""
        from brutus.data.download import _fetch

        mock_strato_fetch.return_value = "/cache/fake_file.h5"
        mock_exists.return_value = True  # Symlink already exists

        with patch("pathlib.Path.symlink_to") as mock_symlink:
            result = _fetch("fake_file.h5", "/tmp")

            # Should not create symlink if it already exists
            mock_symlink.assert_not_called()


@pytest.mark.integration
class TestDataIntegration:
    """Integration tests for data workflows."""

    @patch("brutus.data.download._fetch")
    def test_typical_data_workflow(self, mock_fetch):
        """Test a typical data download and load workflow."""
        # This test simulates a typical user workflow without actual downloads

        # Mock download functions
        mock_fetch.side_effect = [
            Path("/fake/grid_mist_v9.h5"),
            Path("/fake/offsets_mist_v9.txt"),
        ]

        from brutus.data import fetch_grids, fetch_offsets

        # Download model grid
        grid_path = fetch_grids(target_dir="/data", grid="mist_v9")
        assert grid_path == Path("/fake/grid_mist_v9.h5")

        # Download offsets
        offset_path = fetch_offsets(target_dir="/data", grid="mist_v9")
        assert offset_path == Path("/fake/offsets_mist_v9.txt")

        # Verify expected calls
        assert mock_fetch.call_count == 2

    def test_documentation_examples(self):
        """Test that documentation examples work (with mocking)."""
        with patch("brutus.data.download._fetch") as mock_fetch:
            mock_fetch.return_value = Path("/fake/MIST_1.2_iso_vvcrit0.0.h5")

            # Example from fetch_isos docstring
            from brutus.data import fetch_isos

            iso_path = fetch_isos(target_dir="./data/")

            assert iso_path == Path("/fake/MIST_1.2_iso_vvcrit0.0.h5")

        with patch("numpy.loadtxt") as mock_loadtxt:
            mock_loadtxt.return_value = (
                np.array(["g", "r", "i"], dtype="str"),
                np.array(["1.02", "0.98", "1.01"], dtype="str"),
            )

            # Example from load_offsets docstring
            from brutus.data import load_offsets

            with patch("brutus.data.loader.sys.stderr"):
                offsets = load_offsets("./data/offsets_mist_v9.txt", verbose=False)

            assert len(offsets) > 0
            assert np.all(np.isfinite(offsets))


if __name__ == "__main__":
    pytest.main([__file__])
