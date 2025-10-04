#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for grid generation module.

This module tests the GridGenerator class that creates pre-computed
stellar model grids with reddening coefficients.
"""

import pytest
import numpy as np
import h5py
import tempfile
from pathlib import Path

from brutus.core.individual import EEPTracks, StarEvolTrack, StarGrid
from brutus.core.grid_generation import GridGenerator
from brutus.data.loader import load_models


class TestGridGenerator:
    """Test suite for GridGenerator class."""

    def test_init(self):
        """Test GridGenerator initialization."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        assert gen.tracks is tracks
        assert len(gen.filters) == 3
        assert all(f in ["SDSS_g", "SDSS_r", "SDSS_i"] for f in gen.filters)
        assert isinstance(gen.star_track, StarEvolTrack)

    def test_init_default_filters(self):
        """Test GridGenerator with default filters."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, verbose=False)

        # Should have many default filters
        assert len(gen.filters) > 3

    def test_make_grid_minimal(self):
        """Test grid generation with minimal parameters."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g", "SDSS_r"], verbose=False)

        # Create very small grid for testing
        mini_grid = np.array([1.0])
        eep_grid = np.array([350.0])
        feh_grid = np.array([0.0])
        afe_grid = np.array([0.0])
        smf_grid = np.array([0.0])

        gen.make_grid(
            mini_grid=mini_grid,
            eep_grid=eep_grid,
            feh_grid=feh_grid,
            afe_grid=afe_grid,
            smf_grid=smf_grid,
            verbose=False,
        )

        # Check outputs
        assert hasattr(gen, "grid_labels")
        assert hasattr(gen, "grid_seds")
        assert hasattr(gen, "grid_params")
        assert hasattr(gen, "grid_sel")

        # Should have 1 model
        assert len(gen.grid_labels) == 1
        assert len(gen.grid_seds) == 1
        assert len(gen.grid_params) == 1

    def test_grid_structure(self):
        """Test structure of generated grid."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        # Create small grid
        mini_grid = np.array([0.8, 1.0, 1.2])
        eep_grid = np.array([300.0, 400.0])
        feh_grid = np.array([0.0])
        afe_grid = np.array([0.0])
        smf_grid = np.array([0.0])

        gen.make_grid(
            mini_grid=mini_grid,
            eep_grid=eep_grid,
            feh_grid=feh_grid,
            afe_grid=afe_grid,
            smf_grid=smf_grid,
            verbose=False,
        )

        # Should have 3*2*1*1*1 = 6 models
        assert len(gen.grid_labels) == 6

        # Check labels structure
        assert "mini" in gen.grid_labels.dtype.names
        assert "eep" in gen.grid_labels.dtype.names
        assert "feh" in gen.grid_labels.dtype.names

        # Check SEDs structure (structured array with filter names)
        assert "SDSS_g" in gen.grid_seds.dtype.names
        assert "SDSS_r" in gen.grid_seds.dtype.names
        assert "SDSS_i" in gen.grid_seds.dtype.names

        # Each filter should have 3 coefficients
        assert gen.grid_seds["SDSS_g"].shape == (6, 3)

    def test_reference_distance(self):
        """Test that grid is generated at 1 kpc reference distance."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            dist=1000.0,  # Explicitly set to 1 kpc
            verbose=False,
        )

        # The base magnitude (first coefficient) should correspond to 1 kpc
        # We can verify this by comparing to direct StarEvolTrack call
        star_track = StarEvolTrack(tracks, filters=["SDSS_g"], verbose=False)
        sed_direct, _, _ = star_track.get_seds(
            mini=1.0,
            eep=350.0,
            feh=0.0,
            afe=0.0,
            av=0.0,
            rv=3.3,
            dist=1000.0,
            return_dict=False,
        )

        # Base coefficient should match direct evaluation
        base_mag = gen.grid_seds["SDSS_g"][0, 0]
        np.testing.assert_allclose(base_mag, sed_direct[0], rtol=1e-3)

    def test_reddening_coefficients(self):
        """Test that reddening coefficients are plausible."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            verbose=False,
        )

        # Check that we have 3 coefficients per filter
        for filt in ["SDSS_g", "SDSS_r", "SDSS_i"]:
            coeffs = gen.grid_seds[filt][0]
            assert len(coeffs) == 3

            # Base magnitude should be reasonable (roughly 0-10 at 1 kpc)
            assert -5 < coeffs[0] < 15

            # Av coefficient should be positive (extinction makes stars fainter)
            assert coeffs[1] > 0

            # Rv coefficient typically small
            assert abs(coeffs[2]) < 2.0

    def test_invalid_models_flagged(self):
        """Test that invalid models are properly flagged."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Include some parameters that should produce invalid models
        mini_grid = np.array([0.1, 1.0])  # 0.1 Msun too low
        eep_grid = np.array([350.0])
        feh_grid = np.array([0.0])

        gen.make_grid(
            mini_grid=mini_grid,
            eep_grid=eep_grid,
            feh_grid=feh_grid,
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            mini_bound=0.5,  # Should exclude 0.1 Msun
            verbose=False,
        )

        # Should have flagged some models as invalid
        assert not all(gen.grid_sel)

    def test_save_and_load(self):
        """Test saving grid to HDF5 and loading it back."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        # Generate small test grid
        mini_grid = np.array([0.9, 1.0, 1.1])
        eep_grid = np.array([350.0, 400.0])
        feh_grid = np.array([0.0])

        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate and save
            gen.make_grid(
                mini_grid=mini_grid,
                eep_grid=eep_grid,
                feh_grid=feh_grid,
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                output_file=tmp_path,
                verbose=False,
            )

            # Load with h5py to check structure
            with h5py.File(tmp_path, "r") as f:
                assert "mag_coeffs" in f
                assert "labels" in f
                assert "parameters" in f

                # Check attributes
                assert "reference_distance_pc" in f.attrs
                assert f.attrs["reference_distance_pc"] == 1000.0

                # Check dimensions
                assert f["mag_coeffs"].shape[0] == 6  # 3*2*1*1*1
                assert f["labels"].shape[0] == 6

            # Load with load_models
            models, labels, label_mask = load_models(
                tmp_path, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
            )

            assert len(models) > 0
            assert models.shape[1] == 3  # 3 filters
            assert models.shape[2] == 3  # 3 coefficients

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_grid_compatible_with_stargrid(self):
        """Test that generated grid can be loaded by StarGrid."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g", "SDSS_r"], verbose=False)

        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate small grid
            gen.make_grid(
                mini_grid=np.array([0.9, 1.0, 1.1]),
                eep_grid=np.array([350.0, 400.0]),
                feh_grid=np.array([0.0]),
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                output_file=tmp_path,
                verbose=False,
            )

            # Load with load_models
            models, labels, label_mask = load_models(
                tmp_path, filters=["SDSS_g", "SDSS_r"], verbose=False
            )

            # Create StarGrid instance
            grid = StarGrid(models, labels, filters=["SDSS_g", "SDSS_r"], verbose=False)

            # Test that we can get predictions
            preds = grid.get_predictions(mini=1.0, eep=375.0, feh=0.0)

            # Should return valid predictions (dict or structured array)
            assert preds is not None
            # Grid was successfully created and loaded - functional test passed

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_different_dist_warning(self):
        """Test that using non-standard distance is preserved."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate with non-standard distance
            gen.make_grid(
                mini_grid=np.array([1.0]),
                eep_grid=np.array([350.0]),
                feh_grid=np.array([0.0]),
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                dist=500.0,  # Non-standard!
                output_file=tmp_path,
                verbose=False,
            )

            # Check that distance is recorded
            with h5py.File(tmp_path, "r") as f:
                assert f.attrs["reference_distance_pc"] == 500.0

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_grid_params_structure(self):
        """Test that grid parameters match track predictions."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            verbose=False,
        )

        # Check parameter names match tracks
        param_names = gen.grid_params.dtype.names
        track_predictions = tracks.predictions

        for pred in track_predictions:
            assert pred in param_names

    def test_binary_handling(self):
        """Test grid generation with binary stars."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Generate grid with and without binaries
        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0, 0.5]),  # Single + binary
            verbose=False,
        )

        # Should have 2 models
        assert len(gen.grid_labels) == 2

        # Binary should be brighter (smaller magnitude)
        mag_single = gen.grid_seds["SDSS_g"][0, 0]
        mag_binary = gen.grid_seds["SDSS_g"][1, 0]

        # Binary should be brighter (if both valid)
        if np.isfinite(mag_single) and np.isfinite(mag_binary):
            assert mag_binary < mag_single


class TestGridGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_default_grids(self):
        """Test that default grid parameters are used when not specified.

        This test starts grid generation with all defaults (~300k models)
        but terminates early after verifying it's working correctly.
        """
        import signal
        import time

        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Patch make_grid to terminate after a few successful models
        original_get_seds = gen.star_track.get_seds
        call_count = [0]

        def limited_get_seds(*args, **kwargs):
            call_count[0] += 1
            result = original_get_seds(*args, **kwargs)
            # After 5 successful calls, we've verified defaults work
            if call_count[0] >= 5:
                raise KeyboardInterrupt("Test verified - defaults working")
            return result

        gen.star_track.get_seds = limited_get_seds

        try:
            # This will use all default grids (~300k models)
            # but will be interrupted after 5 successful model generations
            gen.make_grid(
                mini_grid=None,  # Use default
                eep_grid=None,  # Use default
                feh_grid=None,  # Use default
                afe_grid=None,  # Use default
                smf_grid=None,  # Use default
                verbose=False,
            )
        except KeyboardInterrupt:
            # Expected - we interrupted after verifying defaults work
            pass

        # Verify that defaults were applied and models were generated
        assert call_count[0] == 5, "Should have generated 5 models before interrupting"

        # Restore original method
        gen.star_track.get_seds = original_get_seds

    def test_empty_filter_list(self):
        """Test behavior with empty filter list."""
        tracks = EEPTracks(verbose=False)

        # Should use default filters
        gen = GridGenerator(tracks, filters=None, verbose=False)
        assert len(gen.filters) > 0

    def test_single_point_grid(self):
        """Test grid with single point in each dimension."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            verbose=False,
        )

        assert len(gen.grid_labels) == 1

    def test_grid_without_save(self):
        """Test generating grid without saving to file."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Generate without output_file
        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            output_file=None,  # Don't save
            verbose=False,
        )

        # Should still have results in memory
        assert hasattr(gen, "grid_seds")
        assert len(gen.grid_seds) == 1

    def test_verbose_output(self):
        """Test that verbose mode produces output."""
        import io
        import sys

        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=True)

        # Capture stderr
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            # Generate small grid with verbose=True
            gen.make_grid(
                mini_grid=np.array([0.9, 1.0]),
                eep_grid=np.array([350.0, 400.0]),
                feh_grid=np.array([0.0]),
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                verbose=True,
            )

            # Check that some output was produced
            output = sys.stderr.getvalue()
            assert "Generating grid" in output or "Grid generation" in output
        finally:
            sys.stderr = old_stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
