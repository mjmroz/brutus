# Testing and Coverage Measurement Notes

## Coverage Measurement Problems and Solutions

### Problem 1: Misleading Total Coverage (37% vs 84.5%)

**Symptom**: Running `pytest tests/test_core/` shows "TOTAL: 37%" even though core module is actually 84.5% covered.

**Root Cause**: The `pyproject.toml` configuration includes `--cov=brutus` in the default pytest options (line 119). This means pytest ALWAYS measures coverage for the entire brutus package, including:
- Untested modules (dust: 0%, plotting: 0%)
- Modules being tested (core: 84.5%)

**Result**: The total coverage is a weighted average across ALL modules, making it appear much lower than the actual coverage of tested modules.

**Solution**: When testing a specific module, explicitly specify the coverage target:
```bash
# WRONG - shows 37% (measures all of brutus)
pytest tests/test_core/

# RIGHT - shows 84.5% (measures only core)
pytest tests/test_core/ --cov=src/brutus/core --cov-report=term-missing
```

### Problem 2: Numba JIT Coverage (57% vs 100%)

**Symptom**: `sed_utils.py` shows 57% coverage, with lines 57-81 (the `@jit` decorated function) marked as uncovered, even though tests execute them.

**Root Cause**: The `coverage.py` library instruments Python bytecode to measure coverage. When numba's JIT compiler is enabled:
1. Numba compiles `@jit`/`@njit` decorated functions to machine code
2. This bypasses the Python interpreter entirely
3. coverage.py never sees the function execute (from its perspective)
4. The function appears uncovered

**Solution**: Disable JIT compilation during testing:
```bash
# Set NUMBA_DISABLE_JIT=1 before running tests
NUMBA_DISABLE_JIT=1 pytest tests/test_core/

# sed_utils.py coverage:
# Without NUMBA_DISABLE_JIT=1: 57% coverage
# With NUMBA_DISABLE_JIT=1: 100% coverage
```

**Note**: Disabling JIT makes tests slower but provides accurate coverage. This is acceptable for testing.

### Combined Solution

For accurate per-module coverage measurement with numba support:
```bash
NUMBA_DISABLE_JIT=1 pytest tests/test_core/ --cov=src/brutus/core --cov-report=term-missing
```

Or use the helper script:
```bash
./run_tests_with_coverage.sh core
```

## Actual Core Module Coverage (with NUMBA_DISABLE_JIT=1)

| File | Lines | Covered | Coverage |
|------|-------|---------|----------|
| `__init__.py` | 6 | 6 | 100% |
| `grid_generation.py` | 125 | 118 | 94% |
| `individual.py` | 542 | 456 | 84% |
| `neural_nets.py` | 65 | 56 | 86% |
| `populations.py` | 177 | 154 | 87% |
| `sed_utils.py` | 42 | 42 | **100%** (57% without NUMBA_DISABLE_JIT) |
| **TOTAL** | **957** | **832** | **86.9%** |

**Tests**: 169 passing (5 new tests added for edge cases)

## Coverage Targets

- **Minimum acceptable**: 80%
- **Target**: 90%
- **Core module current**: 84.5% (need 52 more lines for 90%)

## Files Affected by Numba

Check for `@jit` or `@njit` decorators:
```bash
grep -r "@jit\|@njit" src/brutus/
```

Current files using numba:
- `src/brutus/core/sed_utils.py` - Uses `@jit(nopython=True, cache=True)`

## Test Suite Performance

The full test suite takes approximately **4-5 minutes** to run with `NUMBA_DISABLE_JIT=1`.

**Individual Module Runtimes:**
- **Core**: ~100s (includes comprehensive integration tests)
- **Data**: ~60s (loads large MIST grid files)
- **Dust**: ~40s (Bayestar map queries)
- **Plotting**: ~30s
- **Analysis**: ~20s
- **Utils**: ~10s
- **Priors**: ~10s

**Performance Notes:**
- Core module includes `test_slow_integration.py` which intentionally tests full pipeline
- Data module tests load real MIST HDF5 files multiple times for test independence
- Runtime is acceptable for comprehensive testing; optimizations would require shared state

## Recommendations

1. **Always use `NUMBA_DISABLE_JIT=1` for coverage measurement**
2. **Always specify `--cov=src/brutus/<module>` when testing a specific module**
3. **Use `./run_tests_with_coverage.sh <module>` helper script**
4. **Document any new `@jit`/`@njit` usage in this file**
5. **Expect 4-5 minute runtime for full test suite** (per-module testing is much faster)

## Configuration Updates Made

1. **pyproject.toml**: Added comment explaining NUMBA_DISABLE_JIT requirement
2. **CLAUDE.md**: Added comprehensive coverage measurement guide
3. **run_tests_with_coverage.sh**: Created helper script for easy coverage measurement
