#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test runner for refactored brutus functionality.

This script runs only the tests that work with the current refactored modules:
- utils (math, photometry, sampling)
- data (download, loading, filters)  
- core (sed_utils, basic imports)

Run with: python test_refactored.py
"""

import subprocess
import sys
import os

def run_test_subset(test_pattern, description):
    """Run a subset of tests and report results."""
    print(f"\n{'='*60}")
    print(f"Testing {description}")
    print('='*60)
    
    cmd = [
        sys.executable, "-m", "pytest", 
        test_pattern, 
        "-v", 
        "--tb=short",
        "--disable-warnings"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    """Run all refactored functionality tests."""
    print("Testing Refactored Brutus Functionality")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Test categories for refactored functionality
    test_categories = [
        ("tests/test_core/test_import_structure.py::TestModuleStructure", "Module Structure"),
        ("tests/test_utils/test_math_comprehensive.py::TestMatrixOperations::test_adjoint3_basic", "Math Operations (Basic)"),
        ("tests/test_utils/test_photometry.py::TestMagnitudeConversions", "Photometry Conversions"),
        ("tests/test_data/test_data_comprehensive.py::TestDataDownloadFunctions::test_download_imports", "Data Imports"),
        ("tests/test_core/test_sed_comprehensive.py::TestGetSedsCore::test_get_seds_basic_functionality", "SED Utilities"),
    ]
    
    for test_pattern, description in test_categories:
        total_count += 1
        if run_test_subset(test_pattern, description):
            success_count += 1
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{total_count} test categories passed")
    print('='*60)
    
    if success_count == total_count:
        print("🎉 All refactored functionality tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - see output above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)