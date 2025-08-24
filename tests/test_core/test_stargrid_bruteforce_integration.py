#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive integration tests for StarGrid and BruteForce classes.

Tests the complete workflow from grid creation to stellar parameter estimation.
"""

import pytest
import numpy as np
import warnings
import tempfile
import os

# Import the classes to test
from src.brutus.core.individual import StarGrid
from src.brutus.analysis.individual import BruteForce
from src.brutus.data import load_models


# ============================================================================
# Module-level fixtures
# ============================================================================

@pytest.fixture(scope="module")
def mist_grid():
    """Load MIST v9 grid once for all integration tests."""
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
def comprehensive_mock_grid():
    """Create a more comprehensive mock grid for testing."""
    np.random.seed(42)
    
    # Larger grid for more realistic testing (4x4x3 = 48 models)
    mini_vals = [0.3, 0.8, 1.2, 2.0]
    eep_vals = [200, 300, 400, 500]
    feh_vals = [-1.0, 0.0, 0.5]
    
    nmodels = len(mini_vals) * len(eep_vals) * len(feh_vals)
    nfilters = 5
    
    # Create models
    models = np.zeros((nmodels, nfilters, 3))
    
    # Create labels with more realistic stellar properties
    dtype = [('mini', 'f4'), ('eep', 'f4'), ('feh', 'f4'),
             ('age', 'f4'), ('mass', 'f4'), ('radius', 'f4'),
             ('logg', 'f4'), ('Teff', 'f4'), ('Mr', 'f4'),
             ('agewt', 'f4')]
    labels = np.zeros(nmodels, dtype=dtype)
    
    # Fill grid with realistic stellar evolution
    idx = 0
    for i, mini in enumerate(mini_vals):
        for j, eep in enumerate(eep_vals):
            for k, feh in enumerate(feh_vals):
                # Mass loss during evolution
                mass = mini * (1.0 - 0.05 * (j / len(eep_vals)))
                
                # Stellar properties based on mass and evolution
                if eep < 250:  # Pre-main sequence
                    age = 10**(7 + 2 * (1 - mini))
                    radius = mini**0.8 * 2
                    Teff = 3000 + 2000 * mini**0.8
                elif eep < 350:  # Main sequence
                    age = 10**(9 + 0.5 * (1 - mini))
                    radius = mini**0.8
                    Teff = 5777 * mini**0.6
                elif eep < 450:  # Subgiant
                    age = 10**(9.5 + 0.3 * (1 - mini))
                    radius = mini**0.8 * (1 + 0.5 * (eep - 350) / 100)
                    Teff = 5777 * mini**0.4 * (0.9 + 0.1 * (eep - 350) / 100)
                else:  # Giant
                    age = 10**(10)
                    radius = mini**0.8 * (2 + (eep - 450) / 50)
                    Teff = 4000 * mini**0.2
                
                logg = np.log10(mass / radius**2) + 4.44
                
                # Absolute magnitude (rough main sequence relation)
                Mr = 5.0 - 2.5 * np.log10(mass) + feh * 0.5
                
                # Base magnitudes with realistic colors
                base_mag = Mr + 5 * np.log10(100)  # At 100 pc
                g_mag = base_mag
                r_mag = g_mag - 0.3 - 0.2 * feh
                i_mag = r_mag - 0.2 - 0.1 * feh  
                z_mag = i_mag - 0.1 - 0.05 * feh
                y_mag = z_mag - 0.05
                
                models[idx, :, 0] = [g_mag, r_mag, i_mag, z_mag, y_mag]
                models[idx, :, 1] = [1.5, 1.2, 1.0, 0.8, 0.7]  # Reddening
                models[idx, :, 2] = [0.15, 0.12, 0.10, 0.08, 0.07]  # dR/dRv
                
                # Age weighting (IMF-like, favor younger stars)
                agewt = age**(-0.5) if age > 0 else 1.0
                
                labels[idx] = (mini, eep, feh, age, mass, radius, 
                              logg, Teff, Mr, agewt)
                idx += 1
    
    params = {
        'filters': ['g', 'r', 'i', 'z', 'y'],
        'mini_vals': mini_vals,
        'eep_vals': eep_vals,
        'feh_vals': feh_vals
    }
    
    return StarGrid(models, labels, params)


@pytest.fixture
def multi_star_observations():
    """Create multiple synthetic stellar observations."""
    np.random.seed(42)
    
    # Define different stellar types
    star_params = [
        # Solar-like star
        {'mini': 1.0, 'eep': 350, 'feh': 0.0, 'av': 0.1, 'rv': 3.1, 'dist': 100},
        # Low-mass main sequence
        {'mini': 0.5, 'eep': 300, 'feh': -0.5, 'av': 0.2, 'rv': 3.3, 'dist': 50},
        # Massive star
        {'mini': 1.8, 'eep': 320, 'feh': 0.2, 'av': 0.5, 'rv': 2.8, 'dist': 200},
        # Evolved star
        {'mini': 1.2, 'eep': 450, 'feh': 0.1, 'av': 0.3, 'rv': 3.5, 'dist': 150}
    ]
    
    return star_params


# ============================================================================
# Integration Tests
# ============================================================================

class TestStarGridBruteForceIntegration:
    """Test integration between StarGrid and BruteForce."""
    
    def test_complete_workflow(self, comprehensive_mock_grid):
        """Test complete workflow from grid to fitting."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Create synthetic observation from known model
        true_idx = 24  # Middle model
        true_params = {
            'mini': grid.labels['mini'][true_idx],
            'eep': grid.labels['eep'][true_idx], 
            'feh': grid.labels['feh'][true_idx]
        }
        
        # Generate SED through StarGrid
        true_sed = grid.get_seds(**true_params, av=0.2, rv=3.1, dist=100.0,
                                 return_flux=True)
        
        # Add noise
        flux = true_sed['sed'] * (1 + np.random.normal(0, 0.03, 5))
        flux_err = true_sed['sed'] * 0.03
        mask = np.ones(5, dtype=bool)
        
        # Fit through BruteForce
        results = fitter._fit(
            flux, flux_err, mask,
            Nmc_prior=20,
            wt_thresh=0.05
        )
        
        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc = results
        
        # Check results are reasonable
        assert len(sel) > 0
        assert np.all(np.isfinite(lnp))
        
        # Best model should be close to truth
        best_idx = sel[np.argmax(lnp)]
        assert abs(best_idx - true_idx) <= 5  # Within 5 neighbors
        
    def test_interpolation_vs_grid_points(self, comprehensive_mock_grid):
        """Compare interpolated SEDs vs exact grid points."""
        grid = comprehensive_mock_grid
        
        # Get SED at exact grid point
        exact_params = {'mini': 1.2, 'eep': 400, 'feh': 0.0}
        sed_exact = grid.get_seds(**exact_params, use_multilinear=False)
        
        # Get SED with interpolation at same point
        sed_interp = grid.get_seds(**exact_params, use_multilinear=True)
        
        # Should be very similar
        if sed_interp['grid_idx'] is None:  # Interpolation was used
            np.testing.assert_allclose(sed_exact['sed'], sed_interp['sed'], 
                                     rtol=1e-3)
        else:
            # If interpolation fell back to grid point, should be identical
            np.testing.assert_array_equal(sed_exact['sed'], sed_interp['sed'])
            
    def test_optimization_convergence_properties(self, comprehensive_mock_grid):
        """Test optimization produces reasonable results."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Create clean synthetic data
        true_model = grid.models[20, :, 0]  # Base magnitudes
        flux = 10**(-0.4 * true_model)
        flux_err = flux * 0.01  # Very low noise
        mask = np.ones(5, dtype=bool)
        
        # Run with tight optimization
        results = fitter.loglike_grid(
            flux, flux_err, mask,
            avlim=(0., 1.0),
            rvlim=(2.5, 4.0),
            ltol=1e-3,
            return_vals=True
        )
        
        lnl, ndim, chi2, scale, av, rv, icov = results
        
        # Best model should have reasonable parameters
        best_idx = np.argmax(lnl)
        
        # Scale should be close to 1 (no distance scaling in synthetic data)
        assert 0.5 < scale[best_idx] < 2.0
        
        # Extinction should be small for this clean case
        assert av[best_idx] < 0.5
        
        # Inverse covariance should be positive definite
        assert np.all(np.linalg.eigvals(icov[best_idx]) > 0)
        
    def test_multi_object_fitting(self, comprehensive_mock_grid, multi_star_observations):
        """Test fitting multiple objects."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        results = []
        
        for i, star_params in enumerate(multi_star_observations):
            # Generate synthetic observation
            sed = grid.get_seds(**star_params, return_flux=True)
            flux = sed['sed'] * (1 + np.random.normal(0, 0.05, 5))
            flux_err = sed['sed'] * 0.05
            mask = np.ones(5, dtype=bool)
            
            # Fit
            result = fitter._fit(
                flux, flux_err, mask,
                Nmc_prior=15,
                wt_thresh=0.1
            )
            
            results.append(result)
            
        # Check all fits succeeded
        for result in results:
            sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc = result
            assert len(sel) > 0
            assert np.all(np.isfinite(lnp))
            
        # Different stars should prefer different models
        best_models = [sel[np.argmax(lnp)] for sel, _, lnp, _, _, _, _ in results]
        assert len(set(best_models)) > 1  # Not all the same
        
    def test_prior_effects_on_posteriors(self, comprehensive_mock_grid):
        """Test that priors affect posterior distributions."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Create ambiguous data (high noise)
        true_model = grid.models[15, :, 0]
        flux = 10**(-0.4 * true_model)
        flux = flux * (1 + np.random.normal(0, 0.2, 5))  # High noise
        flux_err = flux * 0.2
        mask = np.ones(5, dtype=bool)
        
        # Get likelihood
        like_results = fitter.loglike_grid(flux, flux_err, mask, return_vals=True)
        
        # Test with flat prior
        results_flat = fitter.logpost_grid(
            like_results,
            lnprior=np.zeros(fitter.NMODEL),
            Nmc_prior=20,
            wt_thresh=0.1
        )
        
        # Test with mass-biased prior (favor low mass)
        mass_prior = -2 * fitter.models_labels['mini']  # Exponential penalty
        results_biased = fitter.logpost_grid(
            like_results,
            lnprior=mass_prior,
            Nmc_prior=20,
            wt_thresh=0.1
        )
        
        # Prior should shift the distribution
        flat_sel, _, flat_lnp, _, _, _, _ = results_flat
        biased_sel, _, biased_lnp, _, _, _, _ = results_biased
        
        # Average mass should be lower with biased prior
        flat_masses = fitter.models_labels['mini'][flat_sel]
        biased_masses = fitter.models_labels['mini'][biased_sel]
        
        weighted_flat_mass = np.average(flat_masses, weights=np.exp(flat_lnp))
        weighted_biased_mass = np.average(biased_masses, weights=np.exp(biased_lnp))
        
        assert weighted_biased_mass < weighted_flat_mass


class TestEdgeCasesAndErrorHandling:
    """Test edge cases in the integrated workflow."""
    
    def test_single_band_observation(self, comprehensive_mock_grid):
        """Test fitting with only one observed band."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Create observation with only one band
        flux = np.array([0.1, np.nan, np.nan, np.nan, np.nan])
        flux_err = np.array([0.005, 1.0, 1.0, 1.0, 1.0])
        mask = np.array([True, False, False, False, False])
        
        lnl, ndim, chi2 = fitter.loglike_grid(flux, flux_err, mask)
        
        assert ndim == 1
        assert np.any(np.isfinite(lnl))
        
    def test_extreme_noise_levels(self, comprehensive_mock_grid):
        """Test with very high noise levels."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        true_model = grid.models[10, :, 0]
        flux = 10**(-0.4 * true_model)
        
        # Very high noise (100% errors)
        flux = flux * (1 + np.random.normal(0, 1.0, 5))
        flux_err = flux * 1.0
        mask = np.ones(5, dtype=bool)
        
        # Should still produce results, just with low confidence
        lnl, ndim, chi2 = fitter.loglike_grid(flux, flux_err, mask)
        
        assert np.all(np.isfinite(lnl))
        # With high noise, likelihoods should be more uniform
        assert np.std(lnl) < 10  # Not too spread out
        
    def test_boundary_conditions(self, comprehensive_mock_grid):
        """Test behavior at grid boundaries."""
        grid = comprehensive_mock_grid
        
        # Get grid limits
        mini_min, mini_max = np.min(grid.labels['mini']), np.max(grid.labels['mini'])
        eep_min, eep_max = np.min(grid.labels['eep']), np.max(grid.labels['eep'])
        feh_min, feh_max = np.min(grid.labels['feh']), np.max(grid.labels['feh'])
        
        # Test at boundaries
        boundary_cases = [
            {'mini': mini_min, 'eep': eep_min, 'feh': feh_min},
            {'mini': mini_max, 'eep': eep_max, 'feh': feh_max},
            {'mini': mini_min, 'eep': eep_max, 'feh': feh_min},
            {'mini': mini_max, 'eep': eep_min, 'feh': feh_max}
        ]
        
        for params in boundary_cases:
            result = grid.get_seds(**params)
            assert result['sed'] is not None
            assert np.all(np.isfinite(result['sed']))
            
    def test_optimization_bounds_enforcement(self, comprehensive_mock_grid):
        """Test that optimization respects parameter bounds."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Create synthetic data
        flux = 10**(-0.4 * np.array([15, 14.8, 14.6, 14.4, 14.2]))
        flux_err = flux * 0.05
        mask = np.ones(5, dtype=bool)
        
        # Use tight bounds
        results = fitter.loglike_grid(
            flux, flux_err, mask,
            avlim=(0.1, 0.3),  # Tight Av bounds
            rvlim=(2.8, 3.2),  # Tight Rv bounds
            return_vals=True
        )
        
        lnl, ndim, chi2, scale, av, rv, icov = results
        
        # All results should respect bounds
        assert np.all(av >= 0.1)
        assert np.all(av <= 0.3)
        assert np.all(rv >= 2.8)
        assert np.all(rv <= 3.2)


class TestPerformanceAndScaling:
    """Test performance characteristics (basic checks)."""
    
    def test_grid_size_scaling(self, mist_grid):
        """Test that operations scale reasonably with grid size."""
        # Use real grid which is large
        fitter = BruteForce(mist_grid, verbose=False)
        
        # Create simple observation
        flux = np.array([0.1, 0.12, 0.15, 0.18, 0.2])
        flux_err = flux * 0.05
        mask = np.ones(5, dtype=bool)
        
        # Time a likelihood calculation (just check it completes)
        lnl, ndim, chi2 = fitter.loglike_grid(flux, flux_err, mask)
        
        # Should complete for large grid
        assert len(lnl) == fitter.NMODEL
        assert np.all(np.isfinite(lnl))
        
    def test_batch_operations_efficiency(self, comprehensive_mock_grid):
        """Test batch SED computation is efficient."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Single SED computation
        single_result = grid.get_seds(mini=1.0, eep=350, feh=0.0)
        
        # Batch computation for all models
        seds, rvecs, drvecs = fitter.get_sed_grid()
        
        # Should produce results for all models
        assert seds.shape == (grid.nmodels, grid.nfilters)
        assert np.all(np.isfinite(seds))


class TestRealWorldScenarios:
    """Test scenarios resembling real observations."""
    
    def test_typical_photometric_survey(self, comprehensive_mock_grid):
        """Test scenario like typical photometric survey."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Simulate Pan-STARRS-like observation
        # Typical errors: ~0.02 mag in good conditions
        true_model = grid.models[25, :, 0]  # Random model
        flux = 10**(-0.4 * true_model)
        
        # Add realistic noise
        flux = flux * (1 + np.random.normal(0, 0.02, 5))
        flux_err = flux * 0.02
        mask = np.ones(5, dtype=bool)
        
        # Add some parallax constraint (like Gaia)
        results = fitter._fit(
            flux, flux_err, mask,
            parallax=2.0,  # 500 pc
            parallax_err=0.2,  # 10% error
            coord=(180.0, 30.0),  # Some Galactic coordinates
            Nmc_prior=50,
            wt_thresh=0.01
        )
        
        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc = results
        
        # Should get reasonable posterior samples
        assert len(sel) > 0
        assert np.all(dist_mc > 0)
        
        # Distance should be roughly consistent with parallax
        median_dist = np.median(dist_mc)
        assert 300 < median_dist < 800  # Rough range around 500 pc
        
    def test_reddened_star_scenario(self, comprehensive_mock_grid):
        """Test scenario with significant reddening."""
        grid = comprehensive_mock_grid
        fitter = BruteForce(grid, verbose=False)
        
        # Create heavily reddened observation
        sed = grid.get_seds(mini=1.2, eep=300, feh=0.0, 
                           av=1.5, rv=2.8,  # High extinction, steep curve
                           return_flux=True)
        
        flux = sed['sed'] * (1 + np.random.normal(0, 0.03, 5))
        flux_err = sed['sed'] * 0.03
        mask = np.ones(5, dtype=bool)
        
        # Fit with appropriate extinction priors
        results = fitter.loglike_grid(
            flux, flux_err, mask,
            avlim=(0.0, 3.0),  # Allow high extinction
            rvlim=(2.0, 5.0),  # Wide Rv range
            av_gauss=(1.0, 0.5),  # Prior favoring some extinction
            return_vals=True
        )
        
        lnl, ndim, chi2, scale, av, rv, icov = results
        
        # Should find models with significant extinction
        best_idx = np.argmax(lnl)
        assert av[best_idx] > 0.5  # Should find significant extinction
        assert 2.0 < rv[best_idx] < 4.0  # Should find reasonable Rv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])