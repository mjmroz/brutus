# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2025-01-XX (In Development)

### 🚀 Major Refactoring Release

This release represents a major refactoring of brutus to improve usability, maintainability, and development workflow while preserving all existing scientific functionality.

### Added

#### 📦 Modern Python Packaging

- **Modern build system**: Migrated from `setup.py` to `pyproject.toml`
- **Development tools**: Added black, isort, flake8, mypy configurations
- **CI/CD pipeline**: Added GitHub Actions for automated testing, linting, and coverage
- **Multi-platform testing**: Automated testing on Linux, macOS, and Windows (WSL)

#### 🧪 Testing Infrastructure

- **Comprehensive test suite**: Added pytest-based testing framework
- **Test fixtures**: Created reusable test data and utilities
- **Coverage reporting**: Added code coverage tracking with codecov
- **Test categories**: Organized unit tests, integration tests, and slow tests

#### 📁 Modular Architecture

- **Organized modules**: Split functionality into logical modules:
  - `brutus.core` - Isochrones, tracks, and stellar models
  - `brutus.analysis` - Fitting algorithms and statistical analysis  
  - `brutus.dust` - 3D dust mapping and extinction
  - `brutus.utils` - Mathematical, photometric, and sampling utilities
  - `brutus.data` - Data downloading, loading, and management
- **Clean public API**: Simplified imports for common workflows
- **Model registry**: Added centralized model management system

#### 📚 Documentation Improvements

- **Enhanced README**: Comprehensive installation and usage guide
- **Installation notes**: Clear Windows/WSL requirements and alternatives
- **Quick start examples**: Added code examples for common workflows
- **API documentation**: Structured documentation for new module organization

### Changed

#### 🐍 Python Support

- **Minimum Python version**: Now requires Python 3.8+ (dropped Python 2.7 support)
- **Dependency updates**: Updated all dependencies to modern versions
- **Removed legacy code**: Cleaned up Python 2 compatibility code

#### 🔧 Internal Reorganization

- **Split large modules**: Broke up the massive `utils.py` file into logical components
- **Improved imports**: Reorganized internal import structure for better modularity
- **Code organization**: Moved functions and classes to more appropriate modules

#### 📋 Dependency Management

- **Removed direct dependencies**: Removed `six` package (Python 2/3 compatibility)
- **Updated constraints**: Added minimum version requirements for all dependencies
- **Optional dependencies**: Organized development and documentation dependencies

### Fixed

#### 🐛 Platform Compatibility  

- **Windows installation**: Added clear documentation about healpy/Windows issues
- **WSL support**: Provided guidance for Windows users to use WSL
- **Alternative installations**: Documented conda and Docker alternatives

#### 🔧 Build System

- **Modern packaging**: Fixed various packaging issues with modern Python build tools
- **Dependency resolution**: Improved dependency management and conflict resolution

### Migration Guide

#### For Existing Users

**Current import patterns continue to work** during the transition period with deprecation warnings:

```python
# OLD (still works but deprecated)
from brutus.seds import Isochrone
from brutus.fitting import BruteForce

# NEW (recommended)
from brutus import Isochrone, BruteForce
# or
from brutus.core import Isochrone
from brutus.analysis import BruteForce
```

#### For Developers

**Development workflow changes:**

```bash
# OLD
python setup.py install

# NEW  
pip install -e ".[dev]"
pytest  # instead of custom test runners
```

### Backward Compatibility

- ✅ **All scientific algorithms preserved**: No changes to core Bayesian inference
- ✅ **Existing import patterns work**: Old imports show warnings but function correctly  
- ✅ **Same file formats**: No changes to data file formats or model grids
- ✅ **API compatibility**: Core function signatures unchanged

### Known Issues

- **Windows native installation**: Requires WSL due to healpy dependency
- **Transition warnings**: Some import patterns may show deprecation warnings
- **Documentation**: Full API documentation still being updated for new structure

## [0.8.3] - Previous Release

### Last release before major refactoring

This was the final release using the old project structure and Python 2 compatibility.

#### Features in 0.8.3 and earlier

- Individual star fitting with brute-force Bayesian inference
- Star cluster modeling and analysis
- 3D dust mapping with Bayestar integration
- MIST isochrone and evolutionary track support
- Neural network SED prediction
- Comprehensive photometric system support
- Data downloading utilities with Pooch

---

## Development Status

### 🔄 Currently In Progress (Phase 1 Complete)

- ✅ Modern packaging and build system
- ✅ Testing infrastructure
- ✅ CI/CD pipeline  
- ✅ Modular directory structure

### 🎯 Coming Next (Phase 2)

- High-level convenience functions (`quick_star_fit`, `quick_cluster_fit`)
- Enhanced error handling and user-friendly messages
- Sampling integration (MCMC, nested sampling)
- Progress bars for long computations

### 🚀 Future Phases  

- Publication-ready visualization utilities
- Interactive plotting capabilities
- JAX compatibility for autodifferentiation
- Enhanced cluster modeling (spatially distributed populations)
- Comprehensive documentation website

---

## Support and Migration

- **Migration questions**: Please open an [issue](https://github.com/joshspeagle/brutus/issues)
- **Bug reports**: Use the issue tracker with the "bug" label
- **Feature requests**: Use the issue tracker with the "enhancement" label
- **Documentation**: Check the updated README and tutorial notebooks

The development team is committed to maintaining backward compatibility during this transition period.
