#!/bin/bash
# Helper script to run tests with proper coverage measurement
# Usage: ./run_tests_with_coverage.sh [module]
# Example: ./run_tests_with_coverage.sh core

MODULE=${1:-all}

if [ "$MODULE" = "all" ]; then
    echo "Running all tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/ --cov=src/brutus --cov-report=term-missing --cov-report=html
elif [ "$MODULE" = "core" ]; then
    echo "Running core tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/test_core/ --cov=src/brutus/core --cov-report=term-missing
elif [ "$MODULE" = "analysis" ]; then
    echo "Running analysis tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/test_analysis/ --cov=src/brutus/analysis --cov-report=term-missing
elif [ "$MODULE" = "utils" ]; then
    echo "Running utils tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/test_utils/ --cov=src/brutus/utils --cov-report=term-missing
elif [ "$MODULE" = "priors" ]; then
    echo "Running priors tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/test_priors/ --cov=src/brutus/priors --cov-report=term-missing
elif [ "$MODULE" = "data" ]; then
    echo "Running data tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/test_data/ --cov=src/brutus/data --cov-report=term-missing
elif [ "$MODULE" = "dust" ]; then
    echo "Running dust tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/test_dust/ --cov=src/brutus/dust --cov-report=term-missing
elif [ "$MODULE" = "plotting" ]; then
    echo "Running plotting tests with NUMBA_DISABLE_JIT=1..."
    NUMBA_DISABLE_JIT=1 pytest tests/test_plotting/ --cov=src/brutus/plotting --cov-report=term-missing
else
    echo "Unknown module: $MODULE"
    echo "Available modules: all, core, analysis, utils, priors, data, dust, plotting"
    exit 1
fi
