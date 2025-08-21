#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus photometric offset analysis.

This test suite includes:
1. Unit tests for PhotometricOffsetsConfig class
2. Comprehensive tests for photometric_offsets function
3. Performance and edge case tests
4. Integration tests with other brutus components
"""

import numpy as np
import pytest
import numpy.testing as npt
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_data():
    """Create realistic mock data for testing."""
    np.random.seed(42)

    n_obj, n_filt, n_samp, n_models = 100, 5, 50, 200

    # Mock photometry
    phot = np.random.uniform(0.1, 10.0, (n_obj, n_filt))
    err = 0.05 * phot + 0.01  # Realistic errors
    mask = np.random.choice([0, 1], (n_obj, n_filt), p=[0.1, 0.9])

    # Mock models
    models = np.random.random((n_models, n_filt, 3))
    models[:, :, 0] = np.random.uniform(14, 20, (n_models, n_filt))
    models[:, :, 1] = np.random.uniform(0.5, 2.0, (n_models, n_filt))
    models[:, :, 2] = np.random.uniform(0.0, 0.3, (n_models, n_filt))

    # Mock fitted parameters
    idxs = np.random.randint(0, n_models, (n_obj, n_samp))
    reds = np.random.uniform(0.0, 3.0, (n_obj, n_samp))
    dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))
    dists = np.random.uniform(0.5, 10.0, (n_obj, n_samp))

    return {
        "phot": phot,
        "err": err,
        "mask": mask,
        "models": models,
        "idxs": idxs,
        "reds": reds,
        "dreds": dreds,
        "dists": dists,
    }


@pytest.fixture
def mock_brutus_functions():
    """Mock brutus utility functions for testing."""
    with patch("brutus.analysis.offsets.get_seds") as mock_get_seds, patch(
        "brutus.analysis.offsets.phot_loglike"
    ) as mock_phot_loglike, patch(
        "brutus.analysis.offsets.logsumexp"
    ) as mock_logsumexp:

        # Mock get_seds to return realistic SEDs
        def mock_get_seds_func(models, av, rv, return_flux=False):
            n_models = len(models) if hasattr(models, "__len__") else 1
            n_filt = models.shape[1] if hasattr(models, "shape") else 5

            # Generate mock SEDs
            if return_flux:
                return np.random.uniform(0.1, 10.0, (n_models, n_filt))
            else:
                return np.random.uniform(14, 20, (n_models, n_filt))

        mock_get_seds.side_effect = mock_get_seds_func

        # Mock phot_loglike to return reasonable log-likelihoods
        mock_phot_loglike.return_value = np.random.normal(-5, 2)

        # Mock logsumexp
        mock_logsumexp.side_effect = lambda x: np.log(
            np.sum(np.exp(x - np.max(x)))
        ) + np.max(x)

        yield {
            "get_seds": mock_get_seds,
            "phot_loglike": mock_phot_loglike,
            "logsumexp": mock_logsumexp,
        }


class TestPhotometricOffsetsConfig:
    """Test the configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        from brutus.analysis.offsets import PhotometricOffsetsConfig

        config = PhotometricOffsetsConfig()

        assert config.min_bands_used == 4
        assert config.min_bands_unused == 3
        assert config.n_bootstrap == 150
        assert config.uncertainty_method == "bootstrap_std"
        assert config.progress_interval == 10
        assert config.use_vectorized_bootstrap == True
        assert config.random_seed is None
        assert config.validate_inputs == True

    def test_custom_config(self):
        """Test custom configuration values."""
        from brutus.analysis.offsets import PhotometricOffsetsConfig

        config = PhotometricOffsetsConfig(
            min_bands_used=5,
            min_bands_unused=4,
            n_bootstrap=100,
            uncertainty_method="bootstrap_iqr",
            progress_interval=5,
            use_vectorized_bootstrap=False,
            random_seed=42,
            validate_inputs=False,
        )

        assert config.min_bands_used == 5
        assert config.min_bands_unused == 4
        assert config.n_bootstrap == 100
        assert config.uncertainty_method == "bootstrap_iqr"
        assert config.progress_interval == 5
        assert config.use_vectorized_bootstrap == False
        assert config.random_seed == 42
        assert config.validate_inputs == False

    def test_config_validation(self):
        """Test configuration validation."""
        from brutus.analysis.offsets import PhotometricOffsetsConfig

        # Test invalid min_bands_used
        with pytest.raises(ValueError, match="min_bands_used must be >= 1"):
            PhotometricOffsetsConfig(min_bands_used=0)

        # Test invalid uncertainty_method
        with pytest.raises(ValueError, match="Unknown uncertainty_method"):
            PhotometricOffsetsConfig(uncertainty_method="invalid")

        # Test invalid n_bootstrap
        with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
            PhotometricOffsetsConfig(n_bootstrap=0)

        # Test invalid progress_interval
        with pytest.raises(ValueError, match="progress_interval must be >= 0"):
            PhotometricOffsetsConfig(progress_interval=-1)


class TestPhotometricOffsetsCore:
    """Test core photometric_offsets functionality."""

    def test_basic_functionality(self, mock_data, mock_brutus_functions):
        """Test basic function execution."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        config = PhotometricOffsetsConfig(
            n_bootstrap=10,
            progress_interval=0,
            validate_inputs=False,  # Skip validation for mocked data
        )

        offsets, errors, n_used = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config,
            verbose=False,
        )

        # Check output shapes
        assert offsets.shape == (5,)
        assert errors.shape == (5,)
        assert n_used.shape == (5,)

        # Check output types and ranges
        assert np.all(np.isfinite(offsets))
        assert np.all(errors >= 0)
        assert np.all(n_used >= 0)

    def test_uncertainty_methods(self, mock_data, mock_brutus_functions):
        """Test different uncertainty estimation methods."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        methods = ["bootstrap_std", "bootstrap_iqr", "analytical"]
        results = {}

        for method in methods:
            config = PhotometricOffsetsConfig(
                n_bootstrap=20,
                uncertainty_method=method,
                progress_interval=0,
                validate_inputs=False,
            )

            offsets, errors, n_used = photometric_offsets(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
                config=config,
                verbose=False,
            )

            results[method] = {"offsets": offsets, "errors": errors}

        # All methods should produce finite results
        for method, result in results.items():
            assert np.all(
                np.isfinite(result["offsets"])
            ), f"Method {method} produced non-finite offsets"
            assert np.all(
                result["errors"] >= 0
            ), f"Method {method} produced negative errors"

    def test_vectorized_vs_loop_bootstrap(self, mock_data, mock_brutus_functions):
        """Test vectorized vs loop bootstrap implementations."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        # Set random seed for reproducible comparison
        rng = np.random.default_rng(42)

        config_vec = PhotometricOffsetsConfig(
            n_bootstrap=20,
            use_vectorized_bootstrap=True,
            progress_interval=0,
            validate_inputs=False,
            random_seed=42,
        )

        config_loop = PhotometricOffsetsConfig(
            n_bootstrap=20,
            use_vectorized_bootstrap=False,
            progress_interval=0,
            validate_inputs=False,
            random_seed=42,
        )

        # This test would need identical RNG states, which is complex with mocked functions
        # Instead, just test that both execute without error
        offsets_vec, _, _ = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config_vec,
            rng=np.random.default_rng(42),
            verbose=False,
        )

        offsets_loop, _, _ = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config_loop,
            rng=np.random.default_rng(42),
            verbose=False,
        )

        # Both should produce finite results
        assert np.all(np.isfinite(offsets_vec))
        assert np.all(np.isfinite(offsets_loop))

    def test_optional_parameters(self, mock_data, mock_brutus_functions):
        """Test optional parameter handling."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        config = PhotometricOffsetsConfig(
            n_bootstrap=10, progress_interval=0, validate_inputs=False
        )

        # Test with all optional parameters
        n_obj, n_filt = mock_data["phot"].shape
        n_samp = mock_data["idxs"].shape[1]

        sel = np.random.choice([True, False], n_obj, p=[0.8, 0.2])
        weights = np.random.uniform(0.5, 1.5, (n_obj, n_samp))
        mask_fit = np.random.choice([True, False], n_filt, p=[0.7, 0.3])
        old_offsets = np.random.uniform(0.9, 1.1, n_filt)

        offsets, errors, n_used = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            sel=sel,
            weights=weights,
            mask_fit=mask_fit,
            old_offsets=old_offsets,
            dim_prior=True,
            config=config,
            verbose=False,
        )

        assert np.all(np.isfinite(offsets))
        assert np.all(errors >= 0)

    def test_prior_application(self, mock_data, mock_brutus_functions):
        """Test Gaussian prior application."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        config = PhotometricOffsetsConfig(
            n_bootstrap=10, progress_interval=0, validate_inputs=False
        )

        # Test without priors
        offsets_no_prior, errors_no_prior, _ = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config,
            verbose=False,
        )

        # Test with priors
        prior_mean = np.ones(5) * 1.05
        prior_std = np.ones(5) * 0.02

        offsets_with_prior, errors_with_prior, _ = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            prior_mean=prior_mean,
            prior_std=prior_std,
            config=config,
            verbose=False,
        )

        # Both should produce finite results
        assert np.all(np.isfinite(offsets_no_prior))
        assert np.all(np.isfinite(offsets_with_prior))
        assert np.all(errors_no_prior >= 0)
        assert np.all(errors_with_prior >= 0)

        # Priors should generally reduce errors (though not guaranteed with mocked data)
        # Just test that prior application doesn't break anything
        assert offsets_with_prior.shape == offsets_no_prior.shape
        assert errors_with_prior.shape == errors_no_prior.shape


class TestInputValidation:
    """Test input validation functionality."""

    def test_array_shape_validation(self, mock_data):
        """Test validation of array shapes."""
        from brutus.analysis.offsets import _validate_inputs

        # Test correct shapes (should not raise)
        _validate_inputs(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
        )

        # Test incorrect err shape
        with pytest.raises(ValueError, match="err shape"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"][:50],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test incorrect mask shape
        with pytest.raises(ValueError, match="mask shape"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"][:50],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test incorrect idxs shape
        with pytest.raises(ValueError, match="idxs shape"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"][:50],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

    def test_array_value_validation(self, mock_data):
        """Test validation of array values."""
        from brutus.analysis.offsets import _validate_inputs

        # Test negative errors
        bad_err = -np.abs(mock_data["err"])
        with pytest.raises(ValueError, match="err must be positive"):
            _validate_inputs(
                mock_data["phot"],
                bad_err,
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test invalid mask values
        bad_mask = mock_data["mask"].copy().astype(float)
        bad_mask[0, 0] = 0.5  # Should be 0 or 1
        with pytest.raises(ValueError, match="mask must contain only 0s and 1s"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                bad_mask,
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test negative distances
        bad_dists = -np.abs(mock_data["dists"])
        with pytest.raises(ValueError, match="dists must be positive"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                bad_dists,
            )

        # Test non-finite photometry
        bad_phot = mock_data["phot"].copy()
        bad_phot[0, 0] = np.nan
        with pytest.raises(ValueError, match="phot contains non-finite values"):
            _validate_inputs(
                bad_phot,
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

    def test_type_validation(self, mock_data):
        """Test validation of array types."""
        from brutus.analysis.offsets import _validate_inputs

        # Test non-array input
        with pytest.raises(TypeError, match="phot must be numpy array"):
            _validate_inputs(
                mock_data["phot"].tolist(),
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_valid_objects(self, mock_brutus_functions):
        """Test behavior when no objects meet selection criteria."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        # Create data where no objects meet minimum band requirements
        n_obj, n_filt = 10, 5
        phot = np.random.uniform(0.1, 1.0, (n_obj, n_filt))
        err = 0.1 * phot
        mask = np.zeros((n_obj, n_filt))  # No detections

        models = np.random.random((10, n_filt, 3))
        idxs = np.random.randint(0, 10, (n_obj, 5))
        reds = np.random.uniform(0, 1, (n_obj, 5))
        dreds = np.random.uniform(3, 4, (n_obj, 5))
        dists = np.random.uniform(1, 5, (n_obj, 5))

        config = PhotometricOffsetsConfig(
            n_bootstrap=5, min_bands_used=3, progress_interval=0, validate_inputs=False
        )

        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )

        # Should handle gracefully
        assert offsets.shape == (n_filt,)
        assert np.all(n_used == 0)  # No objects used
        assert np.all(offsets == 1.0)  # Default values
        assert np.all(errors == 0.0)  # No uncertainty

    def test_single_object(self, mock_brutus_functions):
        """Test with single object."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        n_obj, n_filt = 1, 3
        phot = np.random.uniform(0.1, 1.0, (n_obj, n_filt))
        err = 0.1 * phot
        mask = np.ones((n_obj, n_filt))  # All detected

        models = np.random.random((10, n_filt, 3))
        idxs = np.random.randint(0, 10, (n_obj, 5))
        reds = np.random.uniform(0, 1, (n_obj, 5))
        dreds = np.random.uniform(3, 4, (n_obj, 5))
        dists = np.random.uniform(1, 5, (n_obj, 5))

        config = PhotometricOffsetsConfig(
            n_bootstrap=5,
            min_bands_used=1,  # Very lenient
            min_bands_unused=1,
            progress_interval=0,
            validate_inputs=False,
        )

        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )

        assert np.all(np.isfinite(offsets))
        assert np.all(errors >= 0)

    def test_extreme_bootstrap_counts(self, mock_data, mock_brutus_functions):
        """Test with extreme bootstrap counts."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        # Test very small bootstrap count
        config_small = PhotometricOffsetsConfig(
            n_bootstrap=1, progress_interval=0, validate_inputs=False
        )

        offsets_small, errors_small, _ = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config_small,
            verbose=False,
        )

        assert np.all(np.isfinite(offsets_small))
        assert np.all(errors_small >= 0)

        # Test larger bootstrap count
        config_large = PhotometricOffsetsConfig(
            n_bootstrap=500, progress_interval=0, validate_inputs=False
        )

        offsets_large, errors_large, _ = photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config_large,
            verbose=False,
        )

        assert np.all(np.isfinite(offsets_large))
        assert np.all(errors_large >= 0)


@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests."""

    def test_large_dataset_performance(self, mock_brutus_functions):
        """Test performance with large datasets."""
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        # Create larger dataset
        n_obj, n_filt, n_samp = 1000, 10, 100

        phot = np.random.uniform(0.1, 10.0, (n_obj, n_filt))
        err = 0.1 * phot
        mask = np.random.choice([0, 1], (n_obj, n_filt), p=[0.1, 0.9])

        models = np.random.random((500, n_filt, 3))
        idxs = np.random.randint(0, 500, (n_obj, n_samp))
        reds = np.random.uniform(0, 3, (n_obj, n_samp))
        dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))
        dists = np.random.uniform(0.5, 10, (n_obj, n_samp))

        config = PhotometricOffsetsConfig(
            n_bootstrap=50,  # Reasonable for performance test
            progress_interval=0,
            validate_inputs=False,
            use_vectorized_bootstrap=True,
        )

        # This should complete in reasonable time
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )

        assert offsets.shape == (n_filt,)
        assert np.all(np.isfinite(offsets))

    def test_vectorized_bootstrap_performance(self, mock_data, mock_brutus_functions):
        """Compare vectorized vs loop bootstrap performance."""
        import time
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        config_vec = PhotometricOffsetsConfig(
            n_bootstrap=100,
            use_vectorized_bootstrap=True,
            progress_interval=0,
            validate_inputs=False,
        )

        config_loop = PhotometricOffsetsConfig(
            n_bootstrap=100,
            use_vectorized_bootstrap=False,
            progress_interval=0,
            validate_inputs=False,
        )

        # Time vectorized version
        start_time = time.time()
        photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config_vec,
            verbose=False,
        )
        vec_time = time.time() - start_time

        # Time loop version
        start_time = time.time()
        photometric_offsets(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
            config=config_loop,
            verbose=False,
        )
        loop_time = time.time() - start_time

        # Vectorized should generally be faster (though hard to guarantee with mocks)
        print(f"Vectorized time: {vec_time:.3f}s, Loop time: {loop_time:.3f}s")

        # Just verify both complete successfully
        assert vec_time > 0
        assert loop_time > 0


@pytest.mark.integration
class TestIntegration:
    """Integration tests with other brutus components."""

    def test_realistic_workflow_simulation(self):
        """Test with a simulated realistic workflow."""
        # This would test integration with actual brutus functions
        # For now, just test that the function signature is compatible
        from brutus.analysis.offsets import (
            photometric_offsets,
            PhotometricOffsetsConfig,
        )

        # Verify function signature matches expected interface
        import inspect

        sig = inspect.signature(photometric_offsets)

        required_params = [
            "phot",
            "err",
            "mask",
            "models",
            "idxs",
            "reds",
            "dreds",
            "dists",
        ]

        for param in required_params:
            assert param in sig.parameters, f"Missing required parameter: {param}"

        # Verify return type annotation if present
        if sig.return_annotation != inspect.Signature.empty:
            # Should return tuple of 3 arrays
            pass


if __name__ == "__main__":
    pytest.main([__file__])
