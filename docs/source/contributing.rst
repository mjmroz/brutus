Contributing
============

We welcome contributions to brutus! This document provides guidelines for contributing to the project.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/brutus.git
      cd brutus

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Create a new branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

Running Tests
-------------

We use pytest for testing. Run tests with:

.. code-block:: bash

   # Basic tests
   pytest

   # Include slow tests
   RUN_SLOW_TESTS=1 pytest

   # With coverage
   python run_coverage.py

Code Style
----------

We use several tools to maintain code quality:

- **black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

Run these tools before submitting:

.. code-block:: bash

   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/

Documentation
-------------

- All public functions and classes should have NumPy-style docstrings
- Update documentation when adding new features
- Build docs locally to test:

  .. code-block:: bash

     cd docs
     make html

Submitting Changes
------------------

1. Commit your changes with a descriptive message
2. Push to your fork on GitHub
3. Create a pull request with:
   
   - Clear description of changes
   - Tests for new functionality
   - Updated documentation as needed

Issue Reporting
---------------

When reporting bugs:

- Include brutus version and Python environment details
- Provide a minimal example that reproduces the issue
- Include full error tracebacks

Feature Requests
----------------

For feature requests:

- Describe the use case and motivation
- Consider if it fits the scope of brutus
- Be willing to help implement or test

Release Process
---------------

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backward compatible
- **PATCH**: Bug fixes, backward compatible

Contact
-------

- GitHub Issues: https://github.com/joshspeagle/brutus/issues
- Email: j.speagle@utoronto.ca

Thank you for contributing to brutus!