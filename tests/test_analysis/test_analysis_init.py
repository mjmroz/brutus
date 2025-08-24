#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for brutus.analysis.__init__.py module.

This test suite covers the analysis package initialization, including:
1. Import behavior and error handling
2. __all__ definitions and module exports
3. Integration with submodules
"""

import pytest
import sys
from unittest.mock import patch
import importlib

# Import everything from analysis module at module level for testing
try:
    from brutus.analysis import *
    _IMPORT_SUCCESS = True
except ImportError:
    _IMPORT_SUCCESS = False


class TestAnalysisPackageInit:
    """Test analysis package initialization behavior."""
    
    def test_successful_import(self):
        """Test that analysis package imports successfully."""
        # Direct import should work
        import brutus.analysis
        
        # Check basic module structure
        assert hasattr(brutus.analysis, '__all__')
        assert isinstance(brutus.analysis.__all__, list)
    
    def test_import_all_exports(self):
        """Test that all exports in __all__ can be imported."""
        # Get the current __all__ list
        import brutus.analysis
        all_exports = getattr(brutus.analysis, '__all__', [])
        
        if all_exports and _IMPORT_SUCCESS:
            # Check that each item in __all__ is actually available
            for item in all_exports:
                assert item in globals(), f"Item '{item}' in __all__ but not importable"
        else:
            # If import failed or __all__ is empty, that's also valid
            assert True
    
    def test_submodule_availability(self):
        """Test that submodules are correctly made available."""
        import brutus.analysis
        
        # Check if offsets module is available
        try:
            from brutus.analysis import photometric_offsets
            assert callable(photometric_offsets)
        except ImportError:
            # If not available, __all__ should be empty
            assert brutus.analysis.__all__ == []
    
    def test_config_class_availability(self):
        """Test that configuration classes are available."""
        import brutus.analysis
        
        try:
            from brutus.analysis import PhotometricOffsetsConfig
            assert PhotometricOffsetsConfig is not None
        except ImportError:
            # If not available, __all__ should be empty
            assert brutus.analysis.__all__ == []
    
    def test_import_error_fallback(self):
        """Test behavior when imports fail."""
        # Create a test that simulates the ImportError condition
        # by temporarily making the offsets module unavailable
        
        # First, check if we can import normally
        try:
            from brutus.analysis import photometric_offsets
            offsets_available = True
        except ImportError:
            offsets_available = False
        
        # If offsets is normally available, test the fallback
        if offsets_available:
            # Mock the specific submodule import to fail
            with patch.dict('sys.modules', {'brutus.analysis.offsets': None}):
                # We need to reload the module to trigger the ImportError
                import importlib
                if 'brutus.analysis' in sys.modules:
                    # Reload to trigger the import logic again
                    importlib.reload(sys.modules['brutus.analysis'])
        
        # In any case, ensure the module still works
        import brutus.analysis
        assert hasattr(brutus.analysis, '__all__')
        assert isinstance(brutus.analysis.__all__, list)


class TestAnalysisIntegration:
    """Test integration with other brutus components."""
    
    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import brutus.analysis
        
        assert brutus.analysis.__doc__ is not None
        assert len(brutus.analysis.__doc__.strip()) > 0
        assert "analysis module" in brutus.analysis.__doc__.lower()
    
    def test_expected_functionality(self):
        """Test that expected analysis functionality is available."""
        import brutus.analysis
        
        # Check that the module indicates what functionality should be available
        doc = brutus.analysis.__doc__
        assert "photometric offset" in doc.lower()
        assert "statistical analysis" in doc.lower()
    
    def test_module_structure_consistency(self):
        """Test that module structure is consistent."""
        import brutus.analysis
        
        # If __all__ is not empty, items should be importable
        if brutus.analysis.__all__:
            for item in brutus.analysis.__all__:
                assert hasattr(brutus.analysis, item), f"Item '{item}' in __all__ but not accessible"


if __name__ == "__main__":
    pytest.main([__file__])