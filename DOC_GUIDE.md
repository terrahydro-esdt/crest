# CREST Documentation Guide

CREST will use the **NumPy-style docstrings** (configured in Sphinx with the Napoleon extension).

## Full class documentation example (based on HTG)

Note the use of the `r` prefix at the start of the docstring. That makes it a raw string, so one can use LaTeX expressions like \mathbf and \theta and thus render equations. If you are not using LaTex expressions then leave the `r` prefix out.

```python
class HierarchalTensorGraph:
    r"""Hierarchical graph structure for tensor-based Earth system models.

    HierarchalTensorGraph (HTG): The HTG is the core object where the Earth System Model (ESM)
    is encoded and specified. The HTG provides the information needed to build a CREST
    model using the Tensor Network backend. As the name suggest, HTG is a hierarchical graph object,
    and as such, nodes within a HTG are HTGs themselves.

    Parameters
    ----------
    name : str
        The name of the HTG which must be unique across the graph.
        If ``name`` is ``None``, it defaults to one of
        ['name', '__name__', '__qualname__'].

    node : Callable, optional
        A callable function to initialize a HTG. If set, a single
        node (basenode) HTG is created.

    inputs : dict, optional
        Dictionary of input tensor names (keys) and their specifications (values).
        Best practice: always define them for clarity and validation.

    outputs : dict, optional
        Dictionary of expected outputs in the same format as ``inputs``.

    edges : list[tuple]
        List of (source, target) edge pairs defining graph connectivity.

    Examples
    --------
    Create a simple HTG for soil moisture modeling:

    >>> from crest.model import HierarchalTensorGraph, LambdaNode, TensorSpec
    >>> inputs = {
    ...     'precipitation': TensorSpec(shape=(None, 100), dtype=tf.float32),
    ...     'temperature': TensorSpec(shape=(None, 100), dtype=tf.float32)
    ... }
    >>> htg = HierarchalTensorGraph(name='soil_moisture', inputs=inputs)
    >>> et_node = LambdaNode(compute_et, name='evapotranspiration')
    >>> htg.add_node(et_node)

    Notes
    -----
    **Design Philosophy**

    The HTG architecture separates model structure from execution, enabling:
    - Modular component reuse across models
    - Experimentation with process representations
    - Integration of ML and physics-based components

    **Mathematical Representation**

    Each HTG node represents a transformation :math:`f_i` applied to its inputs:

    .. math::

        \mathbf{y}_i = f_i(\mathbf{x}_i; \theta_i)

    The full hierarchical graph composes these transformations:

    .. math::

        \mathbf{Y} = f_N(f_{N-1}(...f_1(\mathbf{X}; \theta_1)...);\theta_N)

    where :math:`\theta_i` are learnable parameters.

    **Performance Considerations**

    For large graphs (>100 nodes), consider:
    - Compiling with ``.compile()`` before execution
    - Batching tensor operations
    - Profiling execution with TensorBoard integration

    See Also
    --------
    Model : High-level model interface using HTG
    LambdaNode : Wrapper for custom Python functions
    KerasNode : Wrapper for Keras models and layers

    References
    ----------
    .. [1] Pelissier, C. et al. (2024).
       "CREST: A Framework for Coupled Reusable Earth System Tensors".
       (In preparation)
    """
    pass
```

## Full module documentation example

At the top of each module file, add documentation like this:

```python
"""Soil moisture modeling components.

This module provides classes and functions for soil moisture calculation,
including:
- Water balance computation
- Evapotranspiration estimation
- Runoff generation
- Integration with CREST HTG framework

The implementations follow standard hydrological modeling practices and
are optimized for tensor operations on large gridded datasets.

Examples
--------
Basic usage:

>>> from crest.model.hydrology import SoilMoistureNode
>>> sm_node = SoilMoistureNode(method='bucket')
>>> sm_node.compile()

See Also
--------
crest.model.HierarchalTensorGraph : Graph framework
crest.data.Dataset : Data loading for hydrological variables
"""
```

## Get Started

### Docstring template (copy-paste this)

```python
def your_function(param1, param2, optional_param=None):
    """Brief one-line description of what this function does.

    More detailed explanation if needed. This can span multiple
    paragraphs and explain the algorithm, use cases, etc.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : np.ndarray, shape (n, m)
        Description of param2 with shape
    optional_param : str, optional
        Description (default is None)

    Returns
    -------
    result : type
        Description of what is returned

    Raises
    ------
    ValueError
        If the input array is empty.
    TypeError
        If `param1` is not a string.

    Examples
    --------
    >>> result = your_function(arg1, arg2)
    >>> print(result)
    expected output

    Notes
    -----
    Optional section for algorithms, equations, or important details.

    See Also
    --------
    related_function : Description of relation
    """
    # Your code here
    pass
```

There are other optional sections one can include. For that see the offical guide:

- NumPy docstring guide: https://numpydoc.readthedocs.io/

### Class documentation template

```python
class YourClass:
    """Brief description of the class.

    Longer explanation of what this class does and when to use it.

    Parameters
    ----------
    param1 : type
        Description
    param2 : type, optional
        Description (default is None)

    Attributes
    ----------
    attr1 : type
        Description of instance attribute
    attr2 : type
        Description of instance attribute

    Examples
    --------
    >>> obj = YourClass(param1_value)
    >>> obj.method()
    """

    def __init__(self, param1, param2=None):
        self.attr1 = param1
        self.attr2 = param2

    def method(self):
        """Brief description of method.

        Returns
        -------
        result : type
            What this method returns
        """
        pass
```

### DON'T DO THIS

```python
def process(data):
    """Process data."""  # Too vague and redundant!
    pass

# Do not describe here!
def compute(x, y):
    return x + y

def compute(x, y):
    # No docstring at all!
    return x + y

# Uses a different style!
def calculate(values):
    """
    Calculate something
    Args:  # Wrong style! Use NumPy style, not Google
        values: some values
    """
    pass
```

### DO THIS

Use Numpy-style documentation template

```python
def process_soil_moisture(data):
    """Filter and normalize soil moisture measurements.

    Removes outliers beyond 3 standard deviations and normalizes
    values to [0, 1] range for model input.

    Parameters
    ----------
    data : np.ndarray, shape (n_timesteps, n_locations)
        Raw soil moisture measurements in mm

    Returns
    -------
    processed : np.ndarray, shape (n_timesteps, n_locations)
        Normalized soil moisture values in [0, 1]

    Examples
    --------
    >>> raw = np.array([[100, 200], [150, 180]])
    >>> clean = process_soil_moisture(raw)
    """
    pass
```

## Jupyter notebooks

When creating/updating example notebooks:

1. **Add header cell** (first cell, markdown):

   ```markdown
   # Title of Example

   **Purpose**: What this notebook demonstrates

   **Prerequisites**:
   - What needs to be installed
   - What concepts user should know

   **Estimated Runtime**: ~X minutes
   ```

2. **Structure your notebook**:
   - Introduction (what we'll do)
   - Setup and imports
   - Load data
   - Main content (with explanations)
   - Results/visualizations
   - Conclusion and next steps

3. **Before committing**:
   - Clear all output cells (except important visualizations)
   - Test that notebook runs from top to bottom
   - Add to automated tests if it's a critical example

## Helpful commands

Build documentation locally:

```bash
# If starting on a brand new feature branch, start from develop
git checkout -b my_feature

# If already on a feature branch
git checkout my_feature
git rebase develop

# When done with your feature (after testing, etc.)
git checkout develop
git merge --no-ff my_feature

# Testing:
# --------

# Build docs locally (see your changes)
cd docs/ && make html && open _build/html/index.html

# Test that notebooks work
pytest tests/examples/test_nbs.py --examples -v

# Test documentation examples You need to use run_doctest.py):
# For example:
python run_doctest.py crest.model.HierarchalTensorGraph -v

# Optional
# --------
# Check your docstring coverage (need to "pip install interrogate")
interrogate crest/your_module.py

# Optional
# --------
# Format your code
# Note black enforces a single consistent format (need to "pip install black")
black crest/your_module.py
# dry-run:
black --check crest/your_module.py


# for example
def foo(a,b= 42):  print(  a + b )

# after black
def foo(a, b=42):
    print(a + b)

# isort sorts and groups your import statements in a consistent, logical order
# (need to "pip install isort")
# This is actually quite harmless and produces nice import groupings
isort crest/your_module.py
# dry-run:
isort --check crest/your_module.py

# for example:
import xarray as xr
import os
import tensorflow as tf
from crest.graph import GraphBuilder
import numpy as np

# after isort

import os

import numpy as np
import tensorflow as tf
import xarray as xr

from crest.graph import GraphBuilder

```

### Look at these for "inspiration"

- `crest/model/HierarchalTensorGraph.py` - Class documentation
- `crest/utils/Metrics.py` - Function documentation
- `examples/metrics.ipynb` - Notebook structure

## Notes on development tools

If you look in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
  "pytest",
  "black",
  "isort",
  "interrogate",
]
```

`dev` is an **optional dependency group**, also known as an "extra" (PEP 621).

It is **not installed by default**. Instead, it allows someone working on the project to request the development tools when installing. For example, for normal installation:

```bash
pip install .
```

or

```bash
pip install crest
```

That installs only the project's required dependencies.

The packages

* `pytest`
* `black`
* `isort`
* `interrogate`

are **not** installed.

---

### Installing the development dependencies

In this case run

```bash
pip install ".[dev]"
```

from the source tree.

This installs:

* the `crest` package
* `pytest`
* `black`
* `isort`
* `interrogate`

Note that if the package is on PyPI (or another package index), the syntax is

```bash
pip install "crest[dev]"
```

---

### Multiple extras

If you also want the documentation tools:

```bash
pip install ".[dev,docs]"
```

which installs

* the package itself
* everything in `dev`
* everything in `docs`

---

### Why this is useful

It separates dependencies by purpose.

* **Users** installing the package don't need formatting or testing tools.
* **Developers** working on the package do.


## FAQ

**Q: Do I need a docstring for every function?**
A: Yes, for all public functions (not starting with `_`). Private functions should have at least a brief description.

**Q: How long should examples be?**
A: Keep examples minimal (3-5 lines) but make them runnable. Use notebooks for complex examples.

**Q: My function has 10 parameters. Do I document all?**
A: Yes! Users need to know what each parameter does. Consider if your function should be split up.

**Q: Can I use Google style docstrings instead of NumPy?**
A: No, CREST uses NumPy style. It's configured in Sphinx and expected throughout the codebase.

## Documentation checklist

Before submitting code, ensure:

- [ ] All public functions have docstrings with Parameters, Returns, Examples
- [ ] All classes have docstrings with Parameters, Attributes, Examples
- [ ] Module-level docstring describes purpose and main components
- [ ] Complex algorithms include Notes section with equations or references
- [ ] Related functions cross-referenced with "See Also"
- [ ] At least one working example per major function/class
- [ ] Type hints match docstring parameter descriptions
- [ ] Exceptions/errors are documented in Raises section

## References

- Sphinx documentation: https://www.sphinx-doc.org/
- NumPy docstring guide: https://numpydoc.readthedocs.io/
