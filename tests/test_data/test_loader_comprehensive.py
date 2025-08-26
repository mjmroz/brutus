#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus data loader utilities.

Note: This file previously contained extensive mock-based tests that have been
removed in favor of superior real data tests in test_data_comprehensive.py.

The real data tests provide better coverage by testing actual functionality
with real MIST data files rather than artificial mock scenarios.

All mock-based test classes removed:
- TestLoadModels: Redundant with real data tests
- TestLoadModelsErrorHandling: Artificial error scenarios
- TestLoadOffsets: Redundant with real data tests
- TestLoadOffsetsErrorHandling: Artificial error scenarios
- TestDataLoaderIntegration: Mock integration tests
- TestDataLoaderPerformance: Mock performance tests

For actual functionality testing, see:
- tests/test_data/test_data_comprehensive.py (real MIST data tests)
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from brutus.data.loader import load_models, load_offsets


# All test classes removed - see test_data_comprehensive.py for real data tests
# that provide superior functional coverage with actual MIST data files.


class TestPlaceholder:
    """Placeholder class to keep file structure valid."""

    def test_imports(self):
        """Test that imports work correctly."""
        # Basic smoke test to ensure imports don't fail
        assert callable(load_models)
        assert callable(load_offsets)

        # This minimal test ensures the file doesn't break pytest discovery
        # while we rely on the superior real data tests in test_data_comprehensive.py
        pass
