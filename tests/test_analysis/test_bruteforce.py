#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for BruteForce class.

Tests the BruteForce class for grid-based Bayesian stellar parameter estimation.
"""

import pytest
import numpy as np
import warnings
import time
import tempfile
import os

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# Import classes to test
from src.brutus.core.individual import StarGrid
from src.brutus.analysis.individual import BruteForce
from src.brutus.data import load_models


# ============================================================================
# Module-level fixtures
# ============================================================================

@pytest.fixture(scope="module")
def mist_grid():
    """Load MIST v9 grid once for all tests."""
    import os
    
    # Try different paths for the MIST grid file
    grid_paths = [
        '/mnt/d/Dropbox/GitHub/brutus/data/DATAFILES/grid_mist_v9.h5',
        './data/DATAFILES/grid_mist_v9.h5',
        'data/DATAFILES/grid_mist_v9.h5',
    ]
    
    for path in grid_paths:
        if os.path.exists(path):
            try:
                models, combined_labels, label_mask = load_models(path)
                
                # Separate grid parameters from predictions using label_mask
                grid_names = [name for name, is_grid in zip(combined_labels.dtype.names, label_mask[0]) if is_grid]
                pred_names = [name for name, is_grid in zip(combined_labels.dtype.names, label_mask[0]) if not is_grid]
                
                # Extract grid parameters
                if grid_names:
                    labels = combined_labels[grid_names]
                else:
                    # Fallback to standard grid parameters
                    grid_names = ['mini', 'eep', 'feh']
                    labels = combined_labels[grid_names]
                
                # Extract predictions if available
                if pred_names:
                    params = combined_labels[pred_names]
                else:
                    params = None
                    
                return StarGrid(models, labels, params)
            except Exception as e:
                continue
    
    # If real grid unavailable, skip tests that require it
    pytest.skip("MIST grid file not found in expected locations")


@pytest.fixture(scope="module") 
def real_grid_subset():
    """Create a subset of the real MIST grid for testing."""
    import os
    import h5py
    
    # Try to load real grid
    grid_paths = [
        '/mnt/d/Dropbox/GitHub/brutus/data/DATAFILES/grid_mist_v9.h5',
        './data/DATAFILES/grid_mist_v9.h5',
        'data/DATAFILES/grid_mist_v9.h5',
    ]
    
    for path in grid_paths:
        if os.path.exists(path):
            try:
                # Load full grid
                models, combined_labels, label_mask = load_models(path, verbose=False)
                
                # Get filter names 
                with h5py.File(path, 'r') as f:
                    filter_names = list(f['mag_coeffs'].dtype.names)
                
                # Find PanSTARRS filters for focused testing
                ps_indices = [i for i, name in enumerate(filter_names) if name.startswith('PS_')][:5]
                
                # Create subset of models for reasonable test times
                # Select every 1000th model for diversity while keeping it manageable
                subset_indices = np.arange(0, len(models), 1000)[:200]  # ~200 models
                
                models_subset = models[subset_indices]
                labels_subset = combined_labels[subset_indices]
                
                # Separate grid parameters from predictions
                grid_names = [name for name, is_grid in zip(combined_labels.dtype.names, label_mask[0]) if is_grid]
                pred_names = [name for name, is_grid in zip(combined_labels.dtype.names, label_mask[0]) if not is_grid]
                
                labels = labels_subset[grid_names]
                params = labels_subset[pred_names] if pred_names else None
                
                return StarGrid(models_subset, labels, params, filters=filter_names), ps_indices
            except Exception as e:
                continue
    
    pytest.skip("Real MIST grid not available for subset testing")


@pytest.fixture(scope="module")
def mock_grid():
    """Create a small mock grid for testing."""
    np.random.seed(42)
    
    # Create small grid (3x3x3 = 27 models)
    nmodels = 27
    nfilters = 5
    
    # Create models with realistic structure
    models = np.zeros((nmodels, nfilters, 3))
    
    # Separate grid parameters from predictions
    labels_dtype = [('mini', 'f4'), ('eep', 'f4'), ('feh', 'f4')]
    labels = np.zeros(nmodels, dtype=labels_dtype)
    
    params_dtype = [('mass', 'f4'), ('radius', 'f4'), ('logg', 'f4'),
                    ('Teff', 'f4'), ('Mr', 'f4')]
    params = np.zeros(nmodels, dtype=params_dtype)
    
    # Fill grid
    idx = 0
    for i, m in enumerate([0.5, 1.0, 2.0]):
        for j, e in enumerate([200, 350, 450]):
            for k, z in enumerate([-1.0, 0.0, 0.5]):
                # Base magnitudes (brighter for more massive stars)
                base_mag = 15.0 - 2.5 * np.log10(m)
                models[idx, :, 0] = base_mag + np.array([0, -0.2, -0.3, -0.4, -0.45])
                models[idx, :, 1] = np.array([1.5, 1.2, 1.0, 0.8, 0.7])  # Reddening
                models[idx, :, 2] = np.array([0.15, 0.12, 0.10, 0.08, 0.07])  # dR/dRv
                
                # Grid parameters 
                labels[idx] = (m, e, z)
                
                # Predictions
                params[idx] = (m * (1 - 0.05*j),  # mass
                              m**0.8 * (1 + 0.05*j),  # radius
                              4.4 - 0.1*j,  # logg
                              5777 * m**0.4,  # Teff
                              5.0 - 2.5*np.log10(m))  # Mr
                idx += 1
    
    filters = ['g', 'r', 'i', 'z', 'y']
    
    return StarGrid(models, labels, params, filters=filters)


@pytest.fixture(scope="module")
def bruteforce_fitter(mock_grid):
    """Create BruteForce instance with mock grid."""
    return BruteForce(mock_grid, verbose=False)


@pytest.fixture
def synthetic_observation(mock_grid):
    """Create synthetic observation from known model."""
    np.random.seed(42)
    
    # Use model index 13 (middle of grid)
    true_idx = 13
    true_model = mock_grid.models[true_idx, :, 0]  # Base magnitudes
    
    # Convert to flux and add noise
    true_flux = 10**(-0.4 * true_model)
    noise_level = 0.05  # 5% noise
    flux = true_flux * (1 + np.random.normal(0, noise_level, len(true_flux)))
    flux_err = true_flux * noise_level
    
    # All bands observed
    mask = np.ones(len(flux), dtype=bool)
    
    return flux, flux_err, mask, true_idx


# ============================================================================
# BruteForce Tests
# ============================================================================

class TestBruteForceInitialization:
    """Test BruteForce initialization."""
    
    def test_initialization_with_stargrid(self, mock_grid):
        """Test proper initialization with StarGrid."""
        fitter = BruteForce(mock_grid, verbose=False)
        
        assert fitter.NMODEL == 27
        assert fitter.NDIM == 5
        assert fitter.NCOEF == 3
        assert hasattr(fitter, 'star_grid')
        assert hasattr(fitter, 'labels_mask')
        
    def test_initialization_invalid_input(self):
        """Test initialization with invalid input."""
        with pytest.raises(TypeError):
            BruteForce("not a grid")
            
        with pytest.raises(TypeError):
            BruteForce(None)
            
    def test_labels_mask_generation(self, bruteforce_fitter):
        """Test automatic labels mask generation."""
        mask = bruteforce_fitter.labels_mask
        
        # Grid parameters should be True
        assert mask['mini'] == True
        assert mask['eep'] == True
        assert mask['feh'] == True
        
        # Predictions should be False
        assert mask['mass'] == False
        assert mask['radius'] == False
        assert mask['logg'] == False
        
    def test_verbose_output(self, mock_grid, capsys):
        """Test verbose initialization output."""
        fitter = BruteForce(mock_grid, verbose=True)
        captured = capsys.readouterr()
        
        assert "BruteForce initialized" in captured.out
        assert "27" in captured.out  # Number of models
        assert "Grid parameters" in captured.out
        assert "Predictions" in captured.out


class TestBruteForceGetSedGrid:
    """Test batch SED computation."""
    
    def test_get_sed_grid_all_models(self, bruteforce_fitter):
        """Test SED computation for all models."""
        seds, rvecs, drvecs = bruteforce_fitter.get_sed_grid()
        
        assert seds.shape == (27, 5)
        assert rvecs.shape == (27, 5)
        assert drvecs.shape == (27, 5)
        
        # Check values are finite
        assert np.all(np.isfinite(seds))
        
    def test_get_sed_grid_subset(self, bruteforce_fitter):
        """Test SED computation for model subset."""
        indices = [0, 5, 10, 15, 20]
        seds, rvecs, drvecs = bruteforce_fitter.get_sed_grid(indices=indices)
        
        assert seds.shape == (5, 5)
        assert np.all(np.isfinite(seds))
        
    def test_get_sed_grid_with_extinction(self, bruteforce_fitter):
        """Test SED computation with extinction."""
        # Without extinction
        seds_no_ext, _, _ = bruteforce_fitter.get_sed_grid(av=0.0, rv=3.3)
        
        # With extinction
        seds_ext, _, _ = bruteforce_fitter.get_sed_grid(av=0.5, rv=3.1)
        
        # Should be different (extinction makes fainter = larger mags)
        assert not np.allclose(seds_no_ext, seds_ext)
        assert np.mean(seds_ext) > np.mean(seds_no_ext)
        
    def test_get_sed_grid_return_flux(self, bruteforce_fitter):
        """Test returning flux vs magnitudes."""
        seds_mag, _, _ = bruteforce_fitter.get_sed_grid(return_flux=False)
        seds_flux, _, _ = bruteforce_fitter.get_sed_grid(return_flux=True)
        
        # Convert and compare
        flux_from_mag = 10**(-0.4 * seds_mag)
        np.testing.assert_allclose(flux_from_mag, seds_flux, rtol=1e-6)


class TestBruteForceLikelihood:
    """Test likelihood computation."""
    
    def test_loglike_grid_basic(self, bruteforce_fitter, synthetic_observation):
        """Test basic likelihood computation."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            return_vals=False
        )
        
        assert len(lnl) == 27
        assert ndim == 5  # All bands observed
        assert len(chi2) == 27
        assert np.all(np.isfinite(lnl))
        
    def test_loglike_finds_true_model(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood peaks near true model."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            return_vals=False
        )
        
        # Best models should include true model (within top 5)
        best_indices = np.argsort(lnl)[-5:]
        assert true_idx in best_indices
        
    def test_loglike_with_optimization(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood with extinction optimization."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            avlim=(0., 2.0),
            rvlim=(2.5, 4.0),
            av_gauss=(0.0, 1.0),
            rv_gauss=(3.3, 0.2),
            return_vals=True
        )
        
        lnl, ndim, chi2, scale, av, rv, icov = results
        
        assert len(scale) == 27
        assert len(av) == 27
        assert len(rv) == 27
        assert icov.shape == (27, 3, 3)
        
        # Check optimized values are in bounds
        assert np.all((av >= 0.0) & (av <= 2.0))
        assert np.all((rv >= 2.5) & (rv <= 4.0))
        
    def test_loglike_with_parallax(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood with parallax constraint."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        # Add parallax (1 mas = 1 kpc distance)
        parallax = 1.0
        parallax_err = 0.1
        
        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            parallax=parallax,
            parallax_err=parallax_err,
            return_vals=False
        )
        
        assert np.all(np.isfinite(lnl))
        
    def test_loglike_masked_bands(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood with some bands masked."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        # Mask some bands
        mask_partial = mask.copy()
        mask_partial[2:4] = False
        
        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask_partial,
            return_vals=False
        )
        
        assert ndim == 3  # Only 3 bands observed
        assert np.all(np.isfinite(lnl))
        
    def test_loglike_bad_data(self, bruteforce_fitter):
        """Test likelihood with bad data values."""
        # Create data with NaN and inf
        flux = np.array([0.1, np.nan, 0.15, np.inf, 0.2])
        flux_err = np.ones(5) * 0.01
        mask = np.ones(5, dtype=bool)
        
        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            return_vals=False
        )
        
        # Should handle bad values gracefully
        assert ndim < 5  # Bad values should be masked
        assert np.any(np.isfinite(lnl))  # At least some models should have valid likelihood


class TestBruteForcePosterior:
    """Test posterior computation."""
    
    def test_logpost_grid_basic(self, bruteforce_fitter, synthetic_observation):
        """Test basic posterior computation."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        # Get likelihood results first
        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            return_vals=True
        )
        
        # Compute posteriors
        results = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=10,  # Small for testing
            wt_thresh=0.01
        )
        
        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc = results
        
        assert len(sel) > 0  # Some models selected
        assert cov.shape == (len(sel), 3, 3)
        assert len(lnp) == len(sel)
        assert dist_mc.shape == (len(sel), 10)
        
    def test_logpost_with_priors(self, bruteforce_fitter, synthetic_observation):
        """Test posterior with explicit priors."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        # Get likelihood results
        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            return_vals=True
        )
        
        # Add simple prior (prefer low mass)
        lnprior = -bruteforce_fitter.models_labels['mini']
        
        # Compute posteriors
        results = bruteforce_fitter.logpost_grid(
            like_results,
            lnprior=lnprior,
            Nmc_prior=10,
            wt_thresh=0.01
        )
        
        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc = results
        
        # Selected models should be biased toward low mass
        selected_masses = bruteforce_fitter.models_labels['mini'][sel]
        all_masses = bruteforce_fitter.models_labels['mini']
        assert np.mean(selected_masses) <= np.mean(all_masses)
        
    def test_logpost_model_selection(self, bruteforce_fitter, synthetic_observation):
        """Test different model selection methods."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            return_vals=True
        )
        
        # Test weight threshold
        results_wt = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=10,
            wt_thresh=1e-3,
            cdf_thresh=None
        )
        
        # Test CDF threshold
        results_cdf = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=10,
            wt_thresh=None,
            cdf_thresh=0.95
        )
        
        # Both should select some models
        assert len(results_wt[0]) > 0
        assert len(results_cdf[0]) > 0
        
    def test_logpost_monte_carlo_sampling(self, bruteforce_fitter, synthetic_observation):
        """Test Monte Carlo sampling produces reasonable distributions."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask,
            return_vals=True
        )
        
        results = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=100,  # More samples for better statistics
            wt_thresh=0.01
        )
        
        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc = results
        
        # Check distance samples are positive
        assert np.all(dist_mc > 0)
        
        # Check extinction is bounded
        assert np.all(av_mc >= 0)
        assert np.all(av_mc <= 20)  # Default avlim
        
        # Check Rv is bounded
        assert np.all(rv_mc >= 1)
        assert np.all(rv_mc <= 8)  # Default rvlim


class TestBruteForceInternal:
    """Test internal methods."""
    
    def test_setup_method(self, bruteforce_fitter):
        """Test data preprocessing in _setup."""
        # Create test data
        data = 10**(-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)
        
        # Coordinates for Galactic prior
        coords = (120.0, 45.0)
        
        results = bruteforce_fitter._setup(
            data, data_err, data_mask,
            data_coords=coords,
            mag_max=30.0,
            merr_max=0.5
        )
        
        proc_data, proc_err, proc_mask, lnprior, gal_prior, dust_prior = results
        
        # Data should be unchanged (all good values)
        np.testing.assert_array_equal(proc_data, data)
        np.testing.assert_array_equal(proc_mask, data_mask)
        
        # Prior should be initialized
        assert len(lnprior) == bruteforce_fitter.NMODEL
        assert gal_prior is not None  # Function should be set
        
    def test_fit_internal(self, bruteforce_fitter, synthetic_observation):
        """Test internal _fit method."""
        flux, flux_err, mask, true_idx = synthetic_observation
        
        results = bruteforce_fitter._fit(
            flux, flux_err, mask,
            parallax=1.0,
            parallax_err=0.2,
            coord=(120.0, 45.0),
            Nmc_prior=10,
            wt_thresh=0.01
        )
        
        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc = results
        
        # Check basic outputs
        assert len(sel) > 0
        assert np.all(np.isfinite(lnp))
        
    def test_repr_method(self, bruteforce_fitter):
        """Test string representation."""
        repr_str = repr(bruteforce_fitter)
        
        assert "BruteForce" in repr_str
        assert "27" in repr_str  # nmodels
        assert "5" in repr_str   # nfilters


class TestBruteForceWithRealGrid:
    """Tests using real MIST grid."""
    
    def test_initialization_real_grid(self, mist_grid):
        """Test initialization with real MIST grid."""
        fitter = BruteForce(mist_grid, verbose=False)
        
        assert fitter.NMODEL > 1000  # Real grid is large
        assert fitter.NDIM > 0
        assert len(fitter.labels_mask) > 5
        
    def test_synthetic_fit_real_grid(self, mist_grid):
        """Test fitting synthetic data with real grid."""
        fitter = BruteForce(mist_grid, verbose=False)
        
        # Create synthetic observation from a solar-like model
        # Find a model close to solar parameters
        labels = fitter.models_labels
        solar_like = np.where(
            (labels['mini'] > 0.9) & (labels['mini'] < 1.1) &
            (labels['eep'] > 300) & (labels['eep'] < 400) &
            (np.abs(labels['feh']) < 0.1)
        )[0]
        
        if len(solar_like) > 0:
            true_idx = solar_like[0]
            true_model = fitter.models[true_idx, :, 0]
            
            # Create synthetic observation
            true_flux = 10**(-0.4 * true_model)
            flux = true_flux * (1 + np.random.normal(0, 0.02, len(true_flux)))
            flux_err = true_flux * 0.02
            mask = np.ones(len(flux), dtype=bool)
            
            # Run likelihood
            lnl, ndim, chi2 = fitter.loglike_grid(
                flux, flux_err, mask,
                return_vals=False
            )
            
            # Check we get reasonable results
            assert np.all(np.isfinite(lnl))
            
            # True model should have high likelihood
            best_models = np.argsort(lnl)[-100:]  # Top 100
            assert true_idx in best_models


class TestBruteForceWithRealGrid:
    """Test BruteForce with real MIST grid subset for performance and accuracy."""
    
    def test_real_grid_initialization(self, real_grid_subset):
        """Test BruteForce initializes correctly with real grid subset."""
        grid, ps_indices = real_grid_subset
        fitter = BruteForce(grid, verbose=False)
        
        assert fitter.nmodels <= 200  # Should be subset
        assert fitter.nfilters > 40   # Should have many filters  
        assert len(fitter.labels_mask) >= 8  # Should have grid + prediction parameters
        
        # Check that grid parameters are marked correctly
        assert fitter.labels_mask['mini'] == True
        assert fitter.labels_mask['eep'] == True 
        assert fitter.labels_mask['feh'] == True
        
    def test_panstarrs_photometry_fit(self, real_grid_subset):
        """Test fitting with PanSTARRS filters and realistic errors."""
        grid, ps_indices = real_grid_subset
        fitter = BruteForce(grid, verbose=False)
        
        # Create PanSTARRS observations with 0.05 mag errors
        nfilters = grid.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)
        
        # Solar-like star magnitudes in PanSTARRS
        ps_mags = [15.0, 14.8, 14.6, 14.4, 14.3]  # g,r,i,z,y
        for i, idx in enumerate(ps_indices[:5]):
            obs[idx] = ps_mags[i]
            obs_err[idx] = 0.05  # 0.05 mag error as recommended
            
        obs_mask = ~np.isnan(obs)
        
        # Test with subset of models for speed
        indices = np.arange(min(100, fitter.nmodels))
        
        # Run likelihood computation
        start_time = time.time()
        lnl, ndim, chi2 = fitter.loglike_grid(
            obs, obs_err, obs_mask,
            indices=indices,
            ltol=3e-2,
            verbose=False,
            return_vals=False
        )
        elapsed = time.time() - start_time
        
        # Should run quickly
        assert elapsed < 1.0, f"Test took {elapsed:.2f}s, expected < 1.0s"
        
        # Check results
        assert lnl.shape == (len(indices),)
        assert np.all(np.isfinite(lnl))
        assert np.all(ndim >= 0)
        assert np.all(chi2 >= 0)
        
        # Should have reasonable likelihood range
        lnl_range = np.max(lnl) - np.min(lnl)
        assert lnl_range > 10, f"Likelihood range {lnl_range:.1f} too small"
        
    def test_clipping_behavior(self, real_grid_subset):
        """Test that clipping thresholds work properly to speed up computation."""
        grid, ps_indices = real_grid_subset
        fitter = BruteForce(grid, verbose=False)
        
        # Create observations
        nfilters = grid.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)
        
        for i, idx in enumerate(ps_indices[:3]):  # Just 3 filters
            obs[idx] = 15.0 + 0.2 * i
            obs_err[idx] = 0.05
            
        obs_mask = ~np.isnan(obs)
        indices = np.arange(min(100, fitter.nmodels))
        
        # Test with default threshold
        start = time.time()
        lnl_default, _, _ = fitter.loglike_grid(
            obs, obs_err, obs_mask,
            indices=indices,
            ltol=3e-2,  # Default
            verbose=False,
            return_vals=False
        )
        time_default = time.time() - start
        
        # Test with aggressive clipping
        start = time.time()
        lnl_aggressive, _, _ = fitter.loglike_grid(
            obs, obs_err, obs_mask,
            indices=indices,
            ltol=1e-1,  # More aggressive
            verbose=False,
            return_vals=False
        )
        time_aggressive = time.time() - start
        
        # Both should give similar results for top models
        best_default = np.argsort(lnl_default)[-10:]
        best_aggressive = np.argsort(lnl_aggressive)[-10:]
        overlap = len(set(best_default) & set(best_aggressive))
        assert overlap >= 5, f"Only {overlap}/10 top models agree between thresholds"


class TestBruteForcePerformance:
    """Test performance characteristics."""
    
    def test_scaling_with_model_count(self, real_grid_subset):
        """Test performance scales reasonably with number of models."""
        grid, ps_indices = real_grid_subset
        fitter = BruteForce(grid, verbose=False)
        
        # Create simple observations
        nfilters = grid.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)
        
        for i, idx in enumerate(ps_indices[:3]):
            obs[idx] = 15.0
            obs_err[idx] = 0.05
        obs_mask = ~np.isnan(obs)
        
        # Test different model counts
        times = []
        model_counts = [10, 50, 100]
        
        for n_models in model_counts:
            if n_models > fitter.nmodels:
                continue
                
            indices = np.arange(n_models)
            
            start = time.time()
            lnl, _, _ = fitter.loglike_grid(
                obs, obs_err, obs_mask,
                indices=indices,
                ltol=3e-2,
                verbose=False,
                return_vals=False
            )
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Should scale roughly linearly (allowing for overhead)
        if len(times) >= 2:
            # Time per model should be roughly constant
            time_per_model = [t/n for t, n in zip(times, model_counts[:len(times)])]
            assert max(time_per_model) / min(time_per_model) < 5, \
                   f"Poor scaling: {time_per_model}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])