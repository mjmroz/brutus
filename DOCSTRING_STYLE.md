# Docstring Style Guide for brutus

This document defines the official docstring standards for the brutus project. All code should follow these conventions to ensure consistent, high-quality documentation that integrates seamlessly with Sphinx autodoc.

## Table of Contents

1. [General Principles](#general-principles)
2. [Format Standard](#format-standard)
3. [Module-Level Docstrings](#module-level-docstrings)
4. [Function Docstrings](#function-docstrings)
5. [Class Docstrings](#class-docstrings)
6. [Method Docstrings](#method-docstrings)
7. [Mathematical Notation](#mathematical-notation)
8. [Cross-References](#cross-references)
9. [Examples](#examples)
10. [Common Patterns](#common-patterns)

---

## General Principles

### Core Requirements

1. **All public modules, functions, classes, and methods MUST have docstrings**
2. **Use NumPy-style docstrings** (configured in Sphinx with `napoleon` extension)
3. **First line is a one-line summary** ending with a period
4. **Be concise but complete** - explain *what* and *why*, not just *how*
5. **Include executable examples** when helpful for understanding
6. **Cross-reference related functionality** to help users discover the API

### Private Functions

- Internal functions (prefixed with `_`) should have brief docstrings explaining their purpose
- Use simpler format: one-line summary + Parameters/Returns if complex
- Can omit Examples section for internal functions

### Docstring Quality Checklist

- [ ] One-line summary is clear and ends with period
- [ ] All parameters documented with types
- [ ] Return values documented with types
- [ ] Exceptions documented when raised
- [ ] Examples included (for public API)
- [ ] Cross-references to related functions/classes
- [ ] Mathematical notation properly formatted
- [ ] References cited when based on literature

---

## Format Standard

brutus uses **NumPy-style docstrings**, which are parsed by Sphinx's Napoleon extension. This format is chosen for:

- Clean, readable plain-text documentation
- Excellent support for scientific Python projects
- Integration with numpydoc for API documentation
- Compatibility with most Python IDEs

### Key Formatting Rules

1. **Section headers** use underlines with dashes: `Parameters`, `Returns`, etc.
2. **Type annotations** use colon after parameter name: `param : type`
3. **Optional parameters** indicated explicitly: `param : type, optional`
4. **Default values** specified in description: "Default is X."
5. **Multi-line descriptions** indent continuation lines
6. **Code examples** use `>>>` prompts for interactive Python
7. **Math** uses reStructuredText math directive

---

## Module-Level Docstrings

Every `.py` file must start with a module docstring that explains its purpose and contents.

### Template

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
One-line summary of module purpose.

Extended description explaining what this module does, its role in the
package architecture, and any important design decisions or patterns used.

Classes
-------
ClassName : Brief description
    Extended description of the class and its purpose.

Another_Class : Brief description
    What this class does.

Functions
---------
function_name : Brief description
    What this function does.

another_function : Brief description
    What this function does.

Notes
-----
Any important module-level notes, such as:
- Performance considerations
- Thread safety
- Module dependencies
- Design patterns used

See Also
--------
related_module : How it relates to this module
another.module : How it relates to this module

Examples
--------
>>> # Basic usage example
>>> from brutus.module import ClassName
>>> obj = ClassName()

References
----------
.. [1] Author (Year), "Paper Title", Journal, Volume, Pages
.. [2] Author (Year), "Another Paper", Journal, Volume, Pages
"""
```

### Excellent Example from brutus

From `src/brutus/priors/stellar.py`:

```python
"""
Stellar priors for Bayesian parameter estimation.

This module provides log-prior functions for stellar properties including
the initial mass function (IMF) and luminosity functions.
"""
```

### When to Include Each Section

- **Classes/Functions listings**: Always include for modules with multiple public items
- **Notes**: Include for architecture decisions, performance notes, or important caveats
- **See Also**: Include when module is part of a larger workflow
- **Examples**: Include for top-level user-facing modules
- **References**: Include when module implements published algorithms

---

## Function Docstrings

All public functions must have complete docstrings following this structure.

### Template

```python
def function_name(param1, param2, param3=None):
    """
    One-line summary of what the function does.

    Extended description providing more context about the function's
    purpose, behavior, and any important details about its implementation
    or usage. Can span multiple paragraphs if needed.

    Parameters
    ----------
    param1 : type
        Description of param1. Be specific about expected values,
        shape for arrays, units for physical quantities.
    param2 : float or numpy.ndarray
        Description of param2. Can specify multiple allowed types.
        For arrays, specify shape like (N,) or (M, N).
    param3 : str, optional
        Description of optional parameter. Always end with:
        Default is `None` (or the actual default value).

    Returns
    -------
    result : type
        Description of return value. For tuples, list each element.
    error : type
        Second return value if applicable.

    Raises
    ------
    ValueError
        When param1 is negative or out of valid range.
    TypeError
        When param2 is not numeric.

    See Also
    --------
    related_function : Brief description of how it relates
    AnotherClass.method : Brief description

    Notes
    -----
    Additional information about the algorithm, performance
    characteristics, or mathematical formulation.

    Can include mathematical equations using LaTeX:

    .. math::
        f(x) = \\int_{-\\infty}^{\\infty} e^{-x^2} dx

    Examples
    --------
    >>> # Simple example
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0

    >>> # Example with arrays
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> result = function_name(arr, 2.0, param3='option')

    >>> # Example showing edge case
    >>> result = function_name(0, 1.0)  # Valid edge case

    References
    ----------
    .. [1] Kroupa, P. (2001), "The Initial Mass Function of Stars",
           MNRAS, 322, 231
    .. [2] Author et al. (Year), Journal, Volume, Page
    """
```

### Excellent Example from brutus

From `src/brutus/priors/stellar.py`:

```python
def logp_imf(mgrid, alpha_low=1.3, alpha_high=2.3, mass_break=0.5, mgrid2=None):
    """
    Log-prior for a Kroupa-like broken initial mass function.

    Implements a broken power-law IMF with separate slopes for low and high
    stellar masses, following Kroupa (2001). Supports binary systems with
    a secondary mass component.

    Parameters
    ----------
    mgrid : array_like
        Grid of initial stellar masses in solar units. Must be > 0.
    alpha_low : float, optional
        Power-law slope for low-mass stars (M ≤ mass_break).
        Default is 1.3 (Kroupa 2001).
    alpha_high : float, optional
        Power-law slope for high-mass stars (M > mass_break).
        Default is 2.3 (Kroupa 2001).
    mass_break : float, optional
        Transition mass between low and high mass regimes in solar units.
        Default is 0.5.
    mgrid2 : array_like, optional
        Grid of secondary stellar masses for binary systems in solar units.
        If provided, computes joint prior for binary system.

    Returns
    -------
    logp : array_like
        Normalized log-prior probability density for the input mass grid(s).
        Returns -inf for masses below hydrogen burning limit (0.08 solar masses).

    Notes
    -----
    The IMF follows the form:

    .. math::
        \\xi(M) \\propto M^{-\\alpha}

    where α = α_low for M ≤ M_break and α = α_high for M > M_break.

    For binary systems, assumes independent sampling from the same IMF
    for both components.

    References
    ----------
    Kroupa, P. (2001), MNRAS, 322, 231
    """
```

### Section Guidelines

#### Parameters

- **Type first, then description**: `param : type`
- **Use specific types**: `numpy.ndarray of shape (N, M)` not just `array`
- **Specify shapes**: `(Nobj, Nfilt)` for 2D arrays, `(N,)` for 1D
- **Include units**: "in solar masses", "in kpc", "in magnitudes"
- **Mark optional**: `param : type, optional` + "Default is X."
- **Multiple types**: `param : int or float or None`

#### Returns

- **Name the return value**: `distance : float` not just `float`
- **For tuples, list each**: Separate entry for each tuple element
- **Specify shapes for arrays**: Same as Parameters
- **Document special values**: "Returns None if computation fails"

#### Raises

- Include when function explicitly raises exceptions
- List exception type and when it occurs
- Don't document exceptions from called functions unless important

#### See Also

- Cross-reference related functionality
- Use format: `function_name : Brief description`
- Include both alternatives and complementary functions
- Link to related classes: `ClassName.method_name`

#### Notes

- Implementation details users should know
- Performance characteristics
- Thread safety
- Algorithmic complexity
- Mathematical formulation (with LaTeX)
- Caveats or limitations

#### Examples

- **Always include for public API functions**
- Use interactive Python (`>>>` prompt)
- Show imports if not obvious
- Start simple, build complexity
- Include output when helpful: `>>> result\n3.0`
- Show common use cases
- Demonstrate edge cases if important

#### References

- Cite papers/books for algorithms
- Use numbered references: `.. [1] Citation`
- Format: Author (Year), "Title", Journal, Volume, Pages
- Reference in text: "Following [1]_, we compute..."

---

## Class Docstrings

Classes require comprehensive documentation as they're central to the API.

### Template

```python
class ClassName:
    """
    One-line summary of class purpose.

    Extended description of what the class does, its role in the package,
    and important design decisions. Explain the class's responsibilities
    and how it fits into larger workflows.

    Parameters
    ----------
    param1 : type
        Description of constructor parameter.
    param2 : type, optional
        Description of optional constructor parameter.
        Default is `value`.

    Attributes
    ----------
    attr1 : type
        Description of public attribute.
    attr2 : type
        Description of another attribute.

    Methods
    -------
    method_name(arg1, arg2)
        Brief description of what method does.
    another_method(arg1)
        Brief description of another method.

    See Also
    --------
    RelatedClass : How it relates to this class
    complementary_function : How this function relates

    Notes
    -----
    Important notes about:
    - Thread safety
    - State management
    - Performance characteristics
    - Design patterns used
    - Caching behavior

    Examples
    --------
    >>> # Basic instantiation
    >>> obj = ClassName(param1=value)
    >>>
    >>> # Using the object
    >>> result = obj.method_name(arg1, arg2)
    >>> print(result)

    >>> # Advanced usage
    >>> obj2 = ClassName(param1=value1, param2=value2)
    >>> results = [obj2.method_name(x, y) for x, y in zip(xs, ys)]

    References
    ----------
    .. [1] Citation for the algorithm or method
    """

    def __init__(self, param1, param2=None):
        # Implementation
        pass
```

### Excellent Example from brutus

From `src/brutus/core/neural_nets.py`:

```python
class FastNN(object):
    """
    Object that wraps the underlying neural networks used to interpolate
    between grid points on the bolometric correction tables.

    This class provides the core neural network functionality for predicting
    bolometric corrections from stellar parameters. It loads pre-trained
    neural network weights and biases, and provides methods for encoding
    input parameters and evaluating the network.

    Parameters
    ----------
    filters : list of str, optional
        The names of filters that photometry should be computed for.
        If not provided, all available filters will be used. Filter names
        should match those defined in `brutus.data.filters.FILTERS`.

    nnfile : str, optional
        Path to the neural network file containing pre-trained weights
        and biases. Default is `'brutus/data/DATAFILES/nnMIST_BC.h5'` which will
        be downloaded automatically if not present.

    verbose : bool, optional
        Whether to print initialization progress messages to stderr.
        Default is `True`.

    Attributes
    ----------
    w1, w2, w3 : numpy.ndarray
        Neural network weight matrices for each layer.

    b1, b2, b3 : numpy.ndarray
        Neural network bias vectors for each layer.

    xmin, xmax : numpy.ndarray
        Minimum and maximum values for input parameter scaling.

    xspan : numpy.ndarray
        Range of input parameters (xmax - xmin).

    Notes
    -----
    The neural network architecture is a 3-layer feedforward network with
    sigmoid activation functions. Input parameters are scaled to [0,1] range
    before evaluation.

    Expected input parameters (in order):
    - log10(Teff) : Effective temperature in Kelvin
    - log g : Surface gravity in cgs units
    - [Fe/H] : Surface metallicity (log scale)
    - [α/Fe] : Alpha enhancement (log scale)
    - Av : V-band extinction in magnitudes
    - Rv : Reddening parameter R(V) = A(V)/E(B-V)
    """
```

### Guidelines

- **Document design decisions** in extended description
- **List all public attributes** with types and descriptions
- **Summarize methods** (detailed docs go in method docstrings)
- **Show typical usage** in Examples
- **Note state management** if stateful
- **Explain parameters vs attributes** when not obvious

---

## Method Docstrings

Methods follow the same format as functions, but with these considerations:

### Template

```python
def method_name(self, param1, param2=None):
    """
    One-line summary of what the method does.

    Extended description. Can reference class attributes using
    `self.attribute_name` notation in description.

    Parameters
    ----------
    param1 : type
        Description of parameter.
    param2 : type, optional
        Description of optional parameter. Default is `None`.

    Returns
    -------
    result : type
        Description of what is returned.

    Raises
    ------
    ValueError
        When input is invalid.

    See Also
    --------
    other_method : Related method in this class
    OtherClass.method : Related method in different class

    Notes
    -----
    Method-specific implementation notes.

    Examples
    --------
    >>> obj = ClassName()
    >>> result = obj.method_name(param1=value)
    >>> print(result)
    """
```

### Special Methods

#### `__init__`

- Document in the **class docstring**, not as a separate method
- List all constructor parameters in class Parameters section
- Explain initialization behavior in class description

#### `__repr__` and `__str__`

- Can use brief one-line docstrings
- Usually don't need full documentation

#### Magic methods (`__getitem__`, etc.)

- Document if part of public API
- Explain the supported operations

---

## Mathematical Notation

Use reStructuredText math directives for equations. This renders properly in Sphinx HTML output.

### Inline Math

```python
"""
The function computes :math:`f(x) = x^2` for all inputs.
"""
```

### Block Math

```python
"""
Notes
-----
The log-likelihood is computed as:

.. math::
    \\ln \\mathcal{L} = -\\frac{1}{2} \\sum_i \\frac{(y_i - f(x_i))^2}{\\sigma_i^2}

where :math:`y_i` are the observations and :math:`\\sigma_i` are the errors.
"""
```

### Guidelines

- **Use LaTeX syntax** within math directives
- **Escape backslashes**: Use `\\` not `\`
- **Define variables**: Explain what each symbol means
- **Use inline math** for simple expressions: `:math:`x^2``
- **Use block math** for important equations
- **Number equations** when referenced later: `:eq:`label``

### Common Patterns

```python
# Greek letters
:math:`\\alpha, \\beta, \\gamma`

# Subscripts and superscripts
:math:`M_{\\odot}, R^2`

# Fractions
:math:`\\frac{a}{b}`

# Summations and integrals
:math:`\\sum_{i=1}^{N} x_i`
:math:`\\int_{-\\infty}^{\\infty} f(x) dx`

# Operators
:math:`\\log, \\exp, \\sin, \\cos`

# Special formatting
:math:`\\text{solar masses}` for text in equations
```

---

## Cross-References

Sphinx can automatically link to other parts of the documentation.

### Function/Class References

Use backticks to create cross-references:

```python
"""
See Also
--------
brutus.core.EEPTracks : Stellar evolution tracks
load_models : Load pre-computed model grids
`numpy.ndarray` : NumPy array type
"""
```

### Module References

```python
"""
See Also
--------
brutus.analysis : Analysis module
brutus.core.individual : Individual star modeling
"""
```

### External References

Link to external documentation via intersphinx:

```python
"""
Uses `numpy.interp` for interpolation and `scipy.optimize.minimize`
for parameter optimization.
"""
```

### Guidelines

- Use `` `backticks` `` for automatic linking
- Reference full paths: `brutus.core.EEPTracks`
- Link to related functionality to help discoverability
- Cross-reference complementary functions
- Link to alternative implementations

---

## Examples

Examples are crucial for user understanding. Follow these best practices:

### Structure

```python
"""
Examples
--------
>>> # Comment describing what this example shows
>>> import numpy as np
>>> from brutus import function_name
>>>
>>> # Simple usage
>>> result = function_name(1.0, 2.0)
>>> print(result)
3.0

>>> # Example with arrays
>>> arr = np.array([1, 2, 3])
>>> results = function_name(arr, 2.0)
>>> len(results)
3

>>> # Edge case or advanced usage
>>> result = function_name(0, 1.0, option='special')
```

### Guidelines

1. **Use `>>>` prompt** for interactive Python
2. **Include imports** if not obvious
3. **Add comments** to explain what's being demonstrated
4. **Show output** when it helps understanding
5. **Start simple**, build complexity
6. **Include edge cases** if important
7. **Make them executable** - examples should actually work
8. **Blank lines** between logical groups

### What to Show

- **Basic usage**: The simplest way to use the function
- **Common patterns**: How users will typically call it
- **Important options**: Key parameters and their effects
- **Edge cases**: Boundary conditions if relevant
- **Integration**: How it fits into larger workflows

### Example from brutus

From `src/brutus/data/loader.py`:

```python
"""
Examples
--------
>>> from brutus.data import load_models
>>> models, labels, mask = load_models('./data/DATAFILES/grid_mist_v9.h5')
>>> print(f"Loaded {len(models)} models with {models.shape[1]} filters")

>>> # Load only main sequence models
>>> ms_models, ms_labels, _ = load_models('./data/DATAFILES/grid_mist_v9.h5',
...                                       include_postms=False)
"""
```

---

## Common Patterns

### Array Parameters

```python
"""
Parameters
----------
arr : numpy.ndarray of shape (N,)
    1D array of values in solar masses.
matrix : numpy.ndarray of shape (M, N)
    2D array where rows are observations and columns are filters.
data : numpy.ndarray of shape (Nobj, Nfilt, Ncoef)
    3D array of model coefficients.
"""
```

### Physical Quantities

```python
"""
Parameters
----------
distance : float or numpy.ndarray
    Distance to the star(s) in parsecs (pc).
mass : float
    Initial stellar mass in solar masses (M☉).
wavelength : numpy.ndarray
    Wavelength grid in Angstroms.
temperature : float
    Effective temperature in Kelvin (K).
"""
```

### Optional Parameters with Defaults

```python
"""
Parameters
----------
verbose : bool, optional
    Whether to print progress messages to stderr. Default is `True`.
n_samples : int, optional
    Number of bootstrap samples for uncertainty estimation.
    Default is 1000.
method : {'newton', 'bfgs', 'powell'}, optional
    Optimization method. Default is `'newton'`.
"""
```

### Multiple Return Values

```python
"""
Returns
-------
models : numpy.ndarray of shape (Nmodel, Nfilt, Ncoef)
    Array of model coefficients.
labels : structured numpy.ndarray of shape (Nmodel,)
    Structured array with labeled stellar parameters.
mask : numpy.ndarray of shape (Nlabel,)
    Boolean mask indicating which labels are grid parameters.
"""
```

### Configuration Classes

```python
class Config:
    """
    Configuration class for [purpose].

    This class encapsulates all configuration parameters and provides
    sensible defaults with the ability to customize behavior.

    Parameters
    ----------
    param1 : type, optional
        Description. Default is X.
    param2 : type, optional
        Description. Default is Y.

    Attributes
    ----------
    param1 : type
        Stored value of param1.
    param2 : type
        Stored value of param2.

    Methods
    -------
    validate()
        Validate configuration parameters.
    """
```

---

## Checklist for Reviewers

When reviewing docstrings, check:

- [ ] One-line summary is clear and complete
- [ ] All parameters documented with correct types
- [ ] Optional parameters marked and defaults specified
- [ ] Return values fully documented
- [ ] Exceptions documented when relevant
- [ ] Cross-references to related functionality
- [ ] Examples included for public API
- [ ] Examples are executable and correct
- [ ] Mathematical notation properly formatted
- [ ] References cited when implementing published algorithms
- [ ] Consistent with this style guide
- [ ] Sphinx builds without warnings

---

## References

### NumPy Style Guide
- https://numpydoc.readthedocs.io/en/latest/format.html

### Sphinx Documentation
- https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

### Best Practices
- [PEP 257](https://www.python.org/dev/peps/pep-0257/) - Docstring Conventions
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Tools
- `pydocstyle` - Docstring linter
- `interrogate` - Docstring coverage checker
- Sphinx `autodoc` extension
- Sphinx `napoleon` extension (NumPy style)
