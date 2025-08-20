#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test configuration and fixtures for brutus test suite.

This module provides common test fixtures, utilities, and configuration
for all brutus tests.
"""

import pytest
import numpy as np
import numpy.testing as npt
import tempfile
import os
from pathlib import Path


# Test data and fixtures
@pytest.fixture
def simple_photometry():
    """
    Simple test photometry data for unit tests.

    Returns
    -------
    phot : np.ndarray
        Mock photometry data in flux units
    phot_err : np.ndarray
        Mock photometry errors
    filters : list
        Filter names
    """
    # Simple 3-band photometry for a solar-type star
    filters = ["g", "r", "i"]
    mags = np.array([15.2, 14.8, 14.5])  # Rough solar colors
    mag_errs = np.array([0.01, 0.01, 0.01])

    # Convert to flux units (standard for brutus)
    phot = 10 ** (-0.4 * mags)
    phot_err = mag_errs * 0.4 * np.log(10) * phot

    return phot, phot_err, filters


@pytest.fixture
def simple_cluster_photometry():
    """
    Simple cluster photometry for testing cluster fitting.

    Returns
    -------
    phot : np.ndarray
        Mock cluster photometry (Nstars, Nfilters)
    phot_err : np.ndarray
        Mock photometry errors
    filters : list
        Filter names
    """
    np.random.seed(42)  # Reproducible

    filters = ["g", "r", "i"]
    n_stars = 20

    # Simple main sequence with some scatter
    base_mags = np.array([16.0, 15.5, 15.2])

    # Add magnitude variations for different stellar masses
    mag_variations = np.random.normal(0, 0.5, (n_stars, 3))
    all_mags = base_mags[None, :] + mag_variations

    # Convert to flux
    phot = 10 ** (-0.4 * all_mags)
    phot_err = np.full_like(phot, 0.01) * 0.4 * np.log(10) * phot

    return phot, phot_err, filters


@pytest.fixture
def temp_data_dir():
    """
    Temporary directory for test data files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_stellar_parameters():
    """
    Mock stellar parameters for testing.

    Returns
    -------
    params : dict
        Dictionary of stellar parameters
    """
    return {
        "mini": 1.0,  # Solar mass
        "feh": 0.0,  # Solar metallicity
        "afe": 0.0,  # Solar alpha-enhancement
        "eep": 350,  # Main sequence EEP
        "loga": 9.0,  # 1 Gyr age
        "av": 0.1,  # Small extinction
        "rv": 3.3,  # Standard R(V)
        "dist": 1000,  # 1 kpc distance
    }


@pytest.fixture(scope="session")
def skip_slow_tests():
    """
    Skip slow tests unless explicitly requested.

    Use with: @pytest.mark.skipif(skip_slow_tests, reason="Slow test")
    """
    return not os.environ.get("RUN_SLOW_TESTS", False)


# Custom assertion helpers
def assert_array_almost_equal(actual, desired, decimal=7, err_msg=""):
    """
    Wrapper around numpy.testing.assert_array_almost_equal with
    better error messages for brutus tests.
    """
    try:
        npt.assert_array_almost_equal(actual, desired, decimal=decimal)
    except AssertionError as e:
        if err_msg:
            raise AssertionError(f"{err_msg}: {str(e)}")
        else:
            raise


def assert_valid_photometry(phot, phot_err):
    """
    Assert that photometry arrays are valid for brutus.

    Parameters
    ----------
    phot : np.ndarray
        Photometry in flux units
    phot_err : np.ndarray
        Photometry errors
    """
    assert isinstance(phot, np.ndarray), "Photometry must be numpy array"
    assert isinstance(phot_err, np.ndarray), "Photometry errors must be numpy array"
    assert phot.shape == phot_err.shape, "Photometry and errors must have same shape"
    assert np.all(phot > 0), "All photometry must be positive (flux units)"
    assert np.all(phot_err > 0), "All photometry errors must be positive"
    assert np.all(np.isfinite(phot)), "All photometry must be finite"
    assert np.all(np.isfinite(phot_err)), "All photometry errors must be finite"


# Test markers
pytest.mark.slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS", False),
    reason="Slow test - set RUN_SLOW_TESTS=1 to run",
)

pytest.mark.integration = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS", False),
    reason="Integration test - set RUN_INTEGRATION_TESTS=1 to run",
)
