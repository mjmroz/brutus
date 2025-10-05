#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for photometric offset plotting functions.

This test suite provides comprehensive coverage for the photometric_offsets
and photometric_offsets_2d functions.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
from unittest.mock import patch

import matplotlib.pyplot as plt

from brutus.plotting.offsets import photometric_offsets, photometric_offsets_2d


class TestPhotometricOffsets:
    """Comprehensive tests for photometric_offsets function."""

    @pytest.fixture
    def mock_photometry_data(self):
        """Generate mock photometry data for testing."""
        np.random.seed(42)
        nobj, nfilt = 20, 4

        # Create realistic photometry with errors and masks
        phot = np.random.uniform(1e-14, 1e-12, (nobj, nfilt))  # Flux values
        err = phot * np.random.uniform(0.05, 0.15, (nobj, nfilt))  # 5-15% errors
        mask = np.random.choice(
            [True, False], (nobj, nfilt), p=[0.8, 0.2]
        )  # 80% detection rate

        return phot, err, mask

    @pytest.fixture
    def mock_models(self):
        """Generate mock model coefficients."""
        np.random.seed(42)
        nmodels, nfilt, ncoeff = 30, 4, 3

        models = np.random.uniform(10, 20, (nmodels, nfilt, ncoeff))
        # Add some structure - first coefficient is base magnitude
        models[:, :, 0] += np.random.uniform(-2, 2, (nmodels, nfilt))

        return models

    @pytest.fixture
    def mock_posterior_samples(self, mock_photometry_data, mock_models):
        """Generate mock posterior samples."""
        np.random.seed(42)
        phot, err, mask = mock_photometry_data
        nobj = phot.shape[0]
        nmodels = mock_models.shape[0]
        nsamps = 25

        # Create sample indices and parameters
        idxs = np.random.choice(nmodels, (nobj, nsamps), replace=True)
        reds = np.random.exponential(0.3, (nobj, nsamps))  # Av samples
        dreds = np.random.normal(3.1, 0.2, nsamps)  # Rv samples (shared)
        dists = np.random.lognormal(
            np.log(1.5), 0.4, (nobj, nsamps)
        )  # Distance samples

        return idxs, reds, dreds, dists

    def test_photometric_offsets_basic(
        self, mock_photometry_data, mock_models, mock_posterior_samples
    ):
        """Test basic photometric_offsets functionality."""
        phot, err, mask = mock_photometry_data
        idxs, reds, dreds, dists = mock_posterior_samples

        # Mock get_seds and related functions
        nobj, nsamps, nfilt = idxs.shape[0], idxs.shape[1], phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )

        # Mock phot_loglike to return proper shape for selected objects
        def mock_phot_loglike_func(mo, me, mt, mp, dim_prior=True):
            # Return loglike values for each sample (length nsamps)
            return np.random.uniform(-10, -1, nsamps)

        # Mock logsumexp to return evidence for selected objects
        def mock_logsumexp_func(lnl, axis=1):
            # lnl should have shape (n_selected_objects, nsamps)
            # Return evidence for each selected object
            return np.random.uniform(-5, 0, lnl.shape[0])

        # Mock quantile to return reasonable bounds
        def mock_quantile_func(x, q, weights=None):
            if len(x) == 0:
                return [15.0, 20.0]  # Default bounds
            return [np.min(x), np.max(x)]

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch(
                "brutus.plotting.offsets.phot_loglike",
                side_effect=mock_phot_loglike_func,
            ),
            patch("brutus.plotting.offsets.logsumexp", side_effect=mock_logsumexp_func),
            patch("brutus.plotting.offsets.quantile", side_effect=mock_quantile_func),
        ):

            fig, axes = photometric_offsets(
                phot, err, mask, mock_models, idxs, reds, dreds, dists
            )

        # Should return figure and axes
        assert fig is not None
        assert axes is not None
        assert hasattr(axes, "shape")

        plt.close(fig)

    def test_photometric_offsets_with_x_values(
        self, mock_photometry_data, mock_models, mock_posterior_samples
    ):
        """Test photometric_offsets with custom x values."""
        phot, err, mask = mock_photometry_data
        idxs, reds, dreds, dists = mock_posterior_samples
        nobj = phot.shape[0]

        # Create custom x values
        x = np.random.uniform(15, 20, nobj)  # Magnitudes for x-axis

        nobj, nsamps, nfilt = idxs.shape[0], idxs.shape[1], phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-10, -1, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-5, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[15.0, 20.0]),
        ):

            fig, axes = photometric_offsets(
                phot, err, mask, mock_models, idxs, reds, dreds, dists, x=x
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_magnitude_mode(
        self, mock_photometry_data, mock_models, mock_posterior_samples
    ):
        """Test photometric_offsets with magnitude input."""
        phot, err, mask = mock_photometry_data
        idxs, reds, dreds, dists = mock_posterior_samples

        # Convert to magnitude-like values
        mag_phot = np.random.uniform(15, 20, phot.shape)
        mag_err = np.random.uniform(0.05, 0.15, phot.shape)

        nobj, nsamps, nfilt = idxs.shape[0], idxs.shape[1], phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_loglike = np.random.uniform(-10, -1, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-5, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[15.0, 20.0]),
        ):

            fig, axes = photometric_offsets(
                mag_phot,
                mag_err,
                mask,
                mock_models,
                idxs,
                reds,
                dreds,
                dists,
                flux=False,  # Input is already in magnitudes
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_with_weights(
        self, mock_photometry_data, mock_models, mock_posterior_samples
    ):
        """Test photometric_offsets with importance weights."""
        phot, err, mask = mock_photometry_data
        idxs, reds, dreds, dists = mock_posterior_samples

        # Generate weights
        nobj, nsamps = idxs.shape
        weights = np.random.exponential(1.0, (nobj, nsamps))

        nfilt = phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-10, -1, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-5, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[15.0, 20.0]),
        ):

            fig, axes = photometric_offsets(
                phot, err, mask, mock_models, idxs, reds, dreds, dists, weights=weights
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_custom_parameters(
        self, mock_photometry_data, mock_models, mock_posterior_samples
    ):
        """Test photometric_offsets with custom parameters."""
        phot, err, mask = mock_photometry_data
        idxs, reds, dreds, dists = mock_posterior_samples
        nfilt = phot.shape[1]

        # Custom parameters
        bins = [25, 30, 20, 35]  # Different bins per filter
        offset = np.random.uniform(0.95, 1.05, nfilt)  # ±5% offsets
        titles = [f"Filter {i}" for i in range(nfilt)]
        xlabel = "Magnitude"

        nobj, nsamps = idxs.shape
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-10, -1, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-5, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[15.0, 20.0]),
        ):

            fig, axes = photometric_offsets(
                phot,
                err,
                mask,
                mock_models,
                idxs,
                reds,
                dreds,
                dists,
                bins=bins,
                offset=offset,
                titles=titles,
                xlabel=xlabel,
                cmap="plasma",
                plot_thresh=5.0,
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_custom_figure(
        self, mock_photometry_data, mock_models, mock_posterior_samples
    ):
        """Test photometric_offsets with custom figure."""
        phot, err, mask = mock_photometry_data
        idxs, reds, dreds, dists = mock_posterior_samples
        nfilt = phot.shape[1]

        # Create custom figure
        ncols = 2
        nrows = (nfilt - 1) // ncols + 1
        custom_fig, custom_axes = plt.subplots(nrows, ncols, figsize=(8, 6))

        nobj, nsamps = idxs.shape
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-10, -1, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-5, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[15.0, 20.0]),
        ):

            fig, axes = photometric_offsets(
                phot,
                err,
                mask,
                mock_models,
                idxs,
                reds,
                dreds,
                dists,
                fig=(custom_fig, custom_axes),
            )

        # Should return the same figure and axes
        assert fig is custom_fig
        assert axes is custom_axes

        plt.close(fig)

    def test_photometric_offsets_return_format(
        self, mock_photometry_data, mock_models, mock_posterior_samples
    ):
        """Test photometric_offsets return format."""
        phot, err, mask = mock_photometry_data
        idxs, reds, dreds, dists = mock_posterior_samples

        nobj, nsamps, nfilt = idxs.shape[0], idxs.shape[1], phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-10, -1, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-5, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[15.0, 20.0]),
        ):

            result = photometric_offsets(
                phot, err, mask, mock_models, idxs, reds, dreds, dists
            )

        # Should return 2-tuple: (fig, axes)
        assert len(result) == 2
        fig, axes = result

        # Check types
        assert hasattr(fig, "subplots_adjust")  # Figure-like object
        assert hasattr(axes, "shape")  # Array-like object

        plt.close(fig)


class TestPhotometricOffsets2D:
    """Comprehensive tests for photometric_offsets_2d function."""

    @pytest.fixture
    def mock_2d_data(self):
        """Generate mock 2D positioning data."""
        np.random.seed(42)
        nobj = 30

        # Generate 2D coordinates (e.g., sky positions, color-magnitude, etc.)
        x = np.random.uniform(-2, 2, nobj)  # RA-like
        y = np.random.uniform(-1, 1, nobj)  # Dec-like

        return x, y

    @pytest.fixture
    def mock_photometry_data_2d(self):
        """Generate mock photometry data for 2D tests."""
        np.random.seed(42)
        nobj, nfilt = 30, 3

        phot = np.random.uniform(1e-14, 1e-12, (nobj, nfilt))
        err = phot * np.random.uniform(0.05, 0.12, (nobj, nfilt))
        mask = np.random.choice([True, False], (nobj, nfilt), p=[0.85, 0.15])

        return phot, err, mask

    @pytest.fixture
    def mock_models_2d(self):
        """Generate mock models for 2D tests."""
        np.random.seed(42)
        nmodels, nfilt, ncoeff = 25, 3, 3

        models = np.random.uniform(10, 20, (nmodels, nfilt, ncoeff))
        models[:, :, 0] += np.random.uniform(-1.5, 1.5, (nmodels, nfilt))

        return models

    @pytest.fixture
    def mock_posterior_samples_2d(self, mock_photometry_data_2d, mock_models_2d):
        """Generate mock posterior samples for 2D tests."""
        np.random.seed(42)
        phot, err, mask = mock_photometry_data_2d
        nobj = phot.shape[0]
        nmodels = mock_models_2d.shape[0]
        nsamps = 20

        idxs = np.random.choice(nmodels, (nobj, nsamps), replace=True)
        reds = np.random.exponential(0.25, (nobj, nsamps))
        dreds = np.random.normal(3.2, 0.15, nsamps)
        dists = np.random.lognormal(np.log(1.2), 0.35, (nobj, nsamps))

        return idxs, reds, dreds, dists

    def test_photometric_offsets_2d_basic(
        self,
        mock_2d_data,
        mock_photometry_data_2d,
        mock_models_2d,
        mock_posterior_samples_2d,
    ):
        """Test basic photometric_offsets_2d functionality."""
        x, y = mock_2d_data
        phot, err, mask = mock_photometry_data_2d
        idxs, reds, dreds, dists = mock_posterior_samples_2d

        nobj, nsamps, nfilt = idxs.shape[0], idxs.shape[1], phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-8, -2, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-4, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[0.0]),
        ):

            fig, axes = photometric_offsets_2d(
                phot, err, mask, mock_models_2d, idxs, reds, dreds, dists, x, y
            )

        assert fig is not None
        assert axes is not None

        plt.close(fig)

    def test_photometric_offsets_2d_with_offsets(
        self,
        mock_2d_data,
        mock_photometry_data_2d,
        mock_models_2d,
        mock_posterior_samples_2d,
    ):
        """Test photometric_offsets_2d with photometric offsets."""
        x, y = mock_2d_data
        phot, err, mask = mock_photometry_data_2d
        idxs, reds, dreds, dists = mock_posterior_samples_2d
        nfilt = phot.shape[1]

        # Create offsets
        offset = np.array([0.98, 1.02, 0.99])  # ±2% offsets

        nobj, nsamps = idxs.shape
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-8, -2, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-4, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[0.0]),
        ):

            fig, axes = photometric_offsets_2d(
                phot,
                err,
                mask,
                mock_models_2d,
                idxs,
                reds,
                dreds,
                dists,
                x,
                y,
                offset=offset,
                show_off=True,
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_2d_custom_parameters(
        self,
        mock_2d_data,
        mock_photometry_data_2d,
        mock_models_2d,
        mock_posterior_samples_2d,
    ):
        """Test photometric_offsets_2d with custom parameters."""
        x, y = mock_2d_data
        phot, err, mask = mock_photometry_data_2d
        idxs, reds, dreds, dists = mock_posterior_samples_2d
        nfilt = phot.shape[1]

        # Custom parameters
        bins = [15, 20, 18]  # Different bins per filter
        titles = ["g-band", "r-band", "i-band"]
        xlabel = "RA (deg)"
        ylabel = "Dec (deg)"
        clims = (-0.1, 0.1)

        nobj, nsamps = idxs.shape
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-8, -2, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-4, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[0.0]),
        ):

            fig, axes = photometric_offsets_2d(
                phot,
                err,
                mask,
                mock_models_2d,
                idxs,
                reds,
                dreds,
                dists,
                x,
                y,
                bins=bins,
                titles=titles,
                xlabel=xlabel,
                ylabel=ylabel,
                clims=clims,
                cmap="RdBu_r",
                plot_thresh=5,
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_2d_magnitude_mode(
        self,
        mock_2d_data,
        mock_photometry_data_2d,
        mock_models_2d,
        mock_posterior_samples_2d,
    ):
        """Test photometric_offsets_2d in magnitude mode."""
        x, y = mock_2d_data
        phot, err, mask = mock_photometry_data_2d
        idxs, reds, dreds, dists = mock_posterior_samples_2d

        # Convert to magnitude-like data
        mag_phot = np.random.uniform(15, 20, phot.shape)
        mag_err = np.random.uniform(0.05, 0.15, phot.shape)

        nobj, nsamps, nfilt = idxs.shape[0], idxs.shape[1], phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_loglike = np.random.uniform(-8, -2, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-4, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[0.0]),
        ):

            fig, axes = photometric_offsets_2d(
                mag_phot,
                mag_err,
                mask,
                mock_models_2d,
                idxs,
                reds,
                dreds,
                dists,
                x,
                y,
                flux=False,
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_2d_return_format(
        self,
        mock_2d_data,
        mock_photometry_data_2d,
        mock_models_2d,
        mock_posterior_samples_2d,
    ):
        """Test photometric_offsets_2d return format."""
        x, y = mock_2d_data
        phot, err, mask = mock_photometry_data_2d
        idxs, reds, dreds, dists = mock_posterior_samples_2d

        nobj, nsamps, nfilt = idxs.shape[0], idxs.shape[1], phot.shape[1]
        mock_seds = np.random.uniform(15, 20, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 20, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-8, -2, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-4, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[0.0]),
        ):

            result = photometric_offsets_2d(
                phot, err, mask, mock_models_2d, idxs, reds, dreds, dists, x, y
            )

        # Should return 2-tuple: (fig, axes)
        assert len(result) == 2
        fig, axes = result

        # Check types
        assert hasattr(fig, "subplots_adjust")  # Figure-like object
        assert hasattr(axes, "shape")  # Array-like object

        plt.close(fig)


class TestPhotometricOffsetsIntegration:
    """Integration tests for photometric offset functions."""

    def test_photometric_offsets_realistic_scenario(self):
        """Test photometric_offsets with realistic astronomical data."""
        np.random.seed(42)

        # Realistic survey-like data
        nobj, nfilt, nsamps = 50, 5, 30
        nmodels, ncoeff = 40, 4

        # Generate realistic photometry (flux + errors)
        phot = np.random.lognormal(np.log(1e-13), 0.5, (nobj, nfilt))
        err = phot * np.random.uniform(0.03, 0.20, (nobj, nfilt))  # 3-20% errors

        # Realistic detection mask (brighter = higher detection probability)
        snr = phot / err
        detection_prob = 1 / (1 + np.exp(-(snr - 3) / 2))  # Sigmoid around SNR=3
        mask = np.random.random((nobj, nfilt)) < detection_prob

        # Generate models and posterior samples
        models = np.random.uniform(8, 22, (nmodels, nfilt, ncoeff))
        idxs = np.random.choice(nmodels, (nobj, nsamps), replace=True)

        # Realistic astrophysical parameter distributions
        reds = np.random.exponential(0.2, (nobj, nsamps))  # Low extinction
        dreds = np.random.normal(3.1, 0.1, nsamps)  # Standard Rv
        dists = np.random.lognormal(np.log(2.0), 0.6, (nobj, nsamps))  # ~2 kpc

        # Mock dependencies
        mock_seds = np.random.uniform(12, 19, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(12, 19, (nobj, nfilt)),
            np.random.uniform(0.03, 0.20, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-15, -3, nsamps)

        # Mock functions with proper shapes
        def mock_phot_loglike_func(mo, me, mt, mp, dim_prior=True):
            return np.random.uniform(-15, -3, nsamps)

        def mock_logsumexp_func(lnl, axis=1):
            return np.random.uniform(-8, 0, lnl.shape[0])

        def mock_quantile_func(x, q, weights=None):
            if len(x) == 0:
                return [12.0, 19.0]
            return [np.min(x), np.max(x)]

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch(
                "brutus.plotting.offsets.phot_loglike",
                side_effect=mock_phot_loglike_func,
            ),
            patch("brutus.plotting.offsets.logsumexp", side_effect=mock_logsumexp_func),
            patch("brutus.plotting.offsets.quantile", side_effect=mock_quantile_func),
        ):

            fig, axes = photometric_offsets(
                phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                bins=25,
                titles=["u", "g", "r", "i", "z"],
            )

        # Should handle realistic scenario without issues
        assert fig is not None
        assert axes.size >= nfilt  # Should have at least nfilt subplots

        plt.close(fig)

    def test_photometric_offsets_2d_realistic_scenario(self):
        """Test photometric_offsets_2d with realistic 2D data."""
        np.random.seed(42)

        nobj, nfilt, nsamps = 40, 3, 25
        nmodels, ncoeff = 30, 3

        # Generate sky coordinates (simulating a field)
        ra = np.random.uniform(150, 152, nobj)  # 2-degree field
        dec = np.random.uniform(1, 3, nobj)

        # Generate photometry with spatial correlation (e.g., extinction)
        base_extinction = 0.1 + 0.05 * np.sin(ra * np.pi / 180) * np.cos(
            dec * np.pi / 180
        )
        phot = np.random.lognormal(np.log(1e-13), 0.4, (nobj, nfilt))
        # Apply extinction-like effect
        for i in range(nobj):
            phot[i] *= np.exp(
                -base_extinction[i] * np.array([1.2, 1.0, 0.8])
            )  # wavelength dependent

        err = phot * np.random.uniform(0.05, 0.15, (nobj, nfilt))
        mask = np.random.choice([True, False], (nobj, nfilt), p=[0.9, 0.1])

        # Models and samples
        models = np.random.uniform(10, 18, (nmodels, nfilt, ncoeff))
        idxs = np.random.choice(nmodels, (nobj, nsamps), replace=True)
        reds = np.random.exponential(0.15, (nobj, nsamps))
        dreds = np.random.normal(3.1, 0.1, nsamps)
        dists = np.random.lognormal(np.log(1.8), 0.4, (nobj, nsamps))

        # Mock dependencies
        mock_seds = np.random.uniform(13, 17, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(13, 17, (nobj, nfilt)),
            np.random.uniform(0.05, 0.15, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-12, -2, nsamps)

        with (
            patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
            patch(
                "brutus.plotting.offsets.magnitude", return_value=mock_magnitude_result
            ),
            patch("brutus.plotting.offsets.phot_loglike", return_value=mock_loglike),
            patch(
                "brutus.plotting.offsets.logsumexp",
                return_value=np.random.uniform(-6, 0, nobj),
            ),
            patch("brutus.plotting.offsets.quantile", return_value=[0.0]),
        ):

            fig, axes = photometric_offsets_2d(
                phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                ra,
                dec,
                xlabel="RA (deg)",
                ylabel="Dec (deg)",
                titles=["g", "r", "i"],
                bins=12,
            )

        assert fig is not None
        plt.close(fig)

    def test_photometric_offsets_edge_cases(self):
        """Test edge cases for photometric offset functions."""
        # Very small dataset
        np.random.seed(42)
        nobj, nfilt, nsamps = 5, 2, 8
        nmodels, ncoeff = 10, 2

        phot = np.random.uniform(1e-13, 1e-12, (nobj, nfilt))
        err = phot * 0.1
        mask = np.ones((nobj, nfilt), dtype=bool)  # All detected

        models = np.random.uniform(15, 18, (nmodels, nfilt, ncoeff))
        idxs = np.random.choice(nmodels, (nobj, nsamps), replace=True)
        reds = np.random.uniform(0, 0.5, (nobj, nsamps))
        dreds = np.random.uniform(2.8, 3.5, nsamps)
        dists = np.random.uniform(0.5, 3.0, (nobj, nsamps))

        mock_seds = np.random.uniform(15, 18, (nobj * nsamps, nfilt))
        mock_magnitude_result = (
            np.random.uniform(15, 18, (nobj, nfilt)),
            np.random.uniform(0.08, 0.12, (nobj, nfilt)),
        )
        mock_loglike = np.random.uniform(-10, -5, nsamps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May produce warnings for small dataset
            with (
                patch("brutus.plotting.offsets.get_seds", return_value=mock_seds),
                patch(
                    "brutus.plotting.offsets.magnitude",
                    return_value=mock_magnitude_result,
                ),
                patch(
                    "brutus.plotting.offsets.phot_loglike", return_value=mock_loglike
                ),
                patch(
                    "brutus.plotting.offsets.logsumexp",
                    return_value=np.random.uniform(-5, 0, nobj),
                ),
                patch("brutus.plotting.offsets.quantile", return_value=[15.0, 18.0]),
            ):

                fig, axes = photometric_offsets(
                    phot, err, mask, models, idxs, reds, dreds, dists
                )

        assert fig is not None
        plt.close(fig)
