# brutus

*Et tu, Brute?*

![brutus logo](https://github.com/joshspeagle/brutus/blob/master/brutus_logo.png?raw=true)

[![Tests](https://github.com/joshspeagle/brutus/workflows/Tests/badge.svg)](https://github.com/joshspeagle/brutus/actions)
[![Coverage](https://codecov.io/gh/joshspeagle/brutus/branch/main/graph/badge.svg)](https://codecov.io/gh/joshspeagle/brutus)
[![PyPI](https://img.shields.io/pypi/v/astro-brutus.svg)](https://pypi.org/project/astro-brutus/)
[![Python](https://img.shields.io/pypi/pyversions/astro-brutus.svg)](https://pypi.org/project/astro-brutus/)

`brutus` is a Pure Python package for **"brute force" Bayesian inference** to derive distances, reddenings, and stellar properties from photometry. The package is designed to be highly modular and user-friendly, with comprehensive support for modeling individual stars, star clusters, and 3-D dust mapping.

Please contact Josh Speagle (<j.speagle@utoronto.ca>) with any questions.

## Installation

The most recent stable release can be installed via `pip` by running

```bash
pip install astro-brutus
```

> ## ⚠️ NOTE: Large-Scale Refactor in Progress
>
> I am currently in the midst of a large-scale refactor and the main documentation and repository will be unstable. Please refer to the last stable release in the meantime and avoid installing from GitHub.

## Key Features

🌟 **Individual Star Modeling**: Fit distances, reddenings, and stellar properties for individual stars using Bayesian inference

🌟 **Cluster Analysis**: Model stellar clusters with consistent ages, metallicities, and distances

🌟 **3D Dust Mapping**: Integrate with 3D dust maps and model extinction along lines of sight

🌟 **Modern Stellar Models**: Built-in support for MIST isochrones and evolutionary tracks

🌟 **Flexible & Fast**: Optimized algorithms with numba acceleration and modular design

🌟 **Publication Ready**: Designed for ease of use in research workflows

## Detailed Installation

### Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL (see Windows note below)

### Quick Install

For most users, install from PyPI:

```bash
pip install astro-brutus
```

### Development Install

For development or to get the latest features:

```bash
git clone https://github.com/joshspeagle/brutus.git
cd brutus
pip install -e ".[dev]"
```

### Windows Users - Important Note

⚠️ **Windows Compatibility**: Due to the `healpy` dependency (required for dust mapping), brutus does not work reliably on native Windows. **Windows users should install and run brutus in WSL (Windows Subsystem for Linux)**.

Alternative Windows installation options:

- **WSL (Recommended)**: Install Ubuntu or another Linux distribution via WSL and use the standard installation
- **Conda**: Try `conda install -c conda-forge astro-brutus` which may have pre-compiled Windows wheels
- **Docker**: Use a Linux-based Docker container

### Conda Installation

If you use conda, you can install from conda-forge:

```bash
conda install -c conda-forge astro-brutus
```

### Dependencies

Core dependencies that will be automatically installed:

- `numpy` (≥1.19) - Numerical computing
- `scipy` (≥1.6) - Scientific computing  
- `matplotlib` (≥3.3) - Plotting
- `h5py` (≥3.0) - HDF5 file support
- `healpy` (≥1.14) - HEALPix utilities for incorporating dust maps
- `numba` (≥0.53) - Just-in-time compilation for performance
- `pooch` (≥1.4) - Data downloading and management

## Quick Start

### Individual Star Fitting

```python
import numpy as np
from brutus import BruteForce, StarGrid, load_models

# Load stellar models
models, labels = load_models('path/to/models.h5')

# Create a StarGrid
star_grid = StarGrid(models, labels)

# Set up the fitter
fitter = BruteForce(star_grid)

# Your photometry data (flux units)
photometry = np.array([1.2e-3, 0.8e-3, 0.6e-3])  # g, r, i bands
errors = np.array([1e-5, 1e-5, 1e-5])

# Fit the star
results = fitter.fit(photometry, errors, parallax=2.5, parallax_err=0.1)
```

### Isochrone Generation

```python
from brutus import Isochrone

# Create isochrone generator
iso = Isochrone()

# Generate stellar parameters for an isochrone
params = iso.get_predictions(
    feh=0.0,        # Solar metallicity [Fe/H]
    afe=0.0,        # Solar alpha enhancement [alpha/Fe] 
    loga=9.0        # 1 Gyr age (log10(age/yr))
)
```

### Data Management

```python
from brutus import fetch_grids, fetch_isos, fetch_dustmaps

# Download stellar evolution grids  
fetch_grids()

# Download isochrone data
fetch_isos()

# Download 3D dust maps
fetch_dustmaps()
```

## Documentation

📚 **Full Documentation**: [https://github.com/joshspeagle/brutus](https://github.com/joshspeagle/brutus) (comprehensive docs coming soon)

🎓 **Tutorials**: The `tutorials/` directory contains Jupyter notebooks demonstrating key workflows


## Data

Brutus requires stellar evolution models and other data files to function. These can be downloaded automatically using built-in utilities:

```python
from brutus import fetch_grids, fetch_isos, fetch_dustmaps

# Download MIST stellar evolution grids
fetch_grids()

# Download MIST isochrones  
fetch_isos()

# Download 3D dust maps
fetch_dustmaps()
```

## Recent Changes (v0.9.0)

🔧 **Major Refactoring**: Complete reorganization for improved usability and maintainability

🐍 **Modern Python**: Updated to support Python 3.8+ (dropped Python 2.7 support)

📦 **Modern Packaging**: Migrated to `pyproject.toml` and modern build system

🧪 **Testing Framework**: Added comprehensive test suite with CI/CD

📁 **Modular Structure**: Reorganized into logical modules (`core`, `analysis`, `dust`, `utils`, `data`)

🔄 **Backward Compatibility**: Old import patterns continue to work during transition

See [CHANGELOG.md](CHANGELOG.md) for complete details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/joshspeagle/brutus.git
cd brutus
pip install -e ".[dev]"
```

### Running Tests

```bash
# Basic tests
pytest

# Include slow tests
RUN_SLOW_TESTS=1 pytest

# With coverage
pytest --cov=brutus
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Joshua S. Speagle
- **Email**: <j.speagle@utoronto.ca>
- **GitHub**: [https://github.com/joshspeagle/brutus](https://github.com/joshspeagle/brutus)
- **Issues**: [https://github.com/joshspeagle/brutus/issues](https://github.com/joshspeagle/brutus/issues)

## Acknowledgments

brutus builds on stellar evolution models from the [MIST project](http://waps.cfa.harvard.edu/MIST/) and dust maps from the [Bayestar project](https://argonaut.skymaps.info/). We thank the developers of these projects for making their data publicly available.
