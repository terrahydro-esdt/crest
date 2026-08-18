# Contributing to CREST

Thank you for your interest in contributing to the Coupled Reusable Earth System Tensor (CREST) framework!

## Quick Links

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Documentation Standards](#documentation-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Setting Up Your Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/crest.git
   cd crest
   ```

2. **Create a development environment**

   Choose the environment file that matches your platform:

   | Platform | File |
   |---|---|
   | macOS Apple Silicon (M1/M2/M3) | `cicd/environment_macos_arm64.yaml` |
   | Linux x86\_64 (e.g. AWS cluster) | `cicd/environment_cpu_x86_64.yaml` |
   | Linux aarch64 | `cicd/environment_cpu_aarch64.yaml` |

   ```bash
   conda env create -f cicd/environment_macos_arm64.yaml   # adjust for your platform
   conda activate OCETRA_cpu
   ```

3. **Install in development mode**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs everything needed for development and style/type checks
   (`pytest`, `pytest-cov`, `black`, `isort`, `ruff`, `mypy`, `interrogate`,
   `build`, `twine`) — no separate install step needed.

4. **Verify your installation**

   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Branch Naming Conventions

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation improvements
- `refactor/description` - Code refactoring
- `test/description` - Test additions or improvements

### Making Changes

1. **Create a new branch**

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Write code following our style guide (see below)
   - Add tests for new functionality
   - Update documentation

3. **Run tests and checks**

   ```bash
   # Run tests
   pytest tests/ -v

   # Check code style
   black crest/ tests/
   isort crest/ tests/
   ruff check crest/ tests/

   # Check type hints
   mypy crest/

   # Check docstring coverage
   interrogate crest/ -v
   ```

4. **Commit your changes**

   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

## Code Style Guide

### Python Style

We follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Grouped and sorted (use `isort`)

### Type Hints

All public functions should have type hints:

```python
from typing import Optional
import numpy as np

def process_data(
    data: np.ndarray,
    method: str = "mean",
    threshold: Optional[float] = None
) -> np.ndarray:
    """Process data with specified method."""
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `HierarchalTensorGraph`)
- **Functions/methods**: `snake_case` (e.g., `compute_metrics`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- **Private members**: Leading underscore (e.g., `_internal_method`)

## Documentation Standards

All contributions must include appropriate documentation. See [DOC_GUIDE.md](docs/DOC_GUIDE.md) for detailed templates.

### Minimum Requirements

- **Functions**: Docstring with Parameters, Returns, and at least one Example
- **Classes**: Docstring with Parameters, Attributes, and usage Example
- **Modules**: Module-level docstring describing contents
- **New features**: Tutorial or example notebook in `examples/`

### Building Documentation Locally

```bash
pip install -e ".[docs]"
cd docs/
make html
open _build/html/index.html
```

**Note:** notebook rendering (`nbsphinx`) also requires the `pandoc` binary,
which is *not* installable via pip. Install it separately (e.g.
`conda install pandoc` or `brew install pandoc`) if it isn't already on your
`PATH`.

## Testing Requirements

### Writing Tests

- Use `pytest` framework
- Place tests in `tests/` mirroring source structure
- Name test files `test_*.py`
- Name test functions `test_*`

Example test structure:

```python
import pytest
from crest.model import Model

class TestModel:
    """Test suite for Model class."""

    def test_model_initialization(self):
        """Test that Model initializes correctly."""
        model = Model(graph)
        assert model is not None

    def test_model_fit_with_invalid_data(self):
        """Test that Model raises error with invalid data."""
        model = Model(graph)
        with pytest.raises(ValueError):
            model.fit(invalid_data)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/model/test_Model.py -v

# Run with coverage
pytest tests/ --cov=crest --cov-report=html
```

### Test Coverage

- New code should have >80% test coverage
- Critical paths (data loading, model execution) need >90% coverage
- Check coverage with: `pytest --cov=crest --cov-report=term-missing`

## Submitting Changes

### Pull Request Process

1. **Update your branch**

   ```bash
   git fetch origin
   git rebase origin/develop
   ```

2. **Push your changes**

   ```bash
   git push origin feature/my-new-feature
   ```

3. **Create Pull Request**
   - Go to GitHub and create a PR from your branch to `develop`
   - Fill out the PR template completely
   - Link any related issues

4. **PR Requirements**
   - [ ] All tests pass
   - [ ] Code style checks pass
   - [ ] Documentation is updated
   - [ ] New features have examples
   - [ ] CHANGELOG.md is updated
   - [ ] At least one approving review

### Pull Request Template

Opening a PR on GitHub automatically populates the description from
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) — fill
that out rather than writing a description from scratch.

## Contribution Types

### Code Contributions

- New features and enhancements
- Bug fixes
- Performance improvements
- Refactoring

### Documentation Contributions

- Fixing typos or unclear documentation
- Adding examples and tutorials
- Improving API reference
- Translating documentation (future)

### Community Contributions

- Answering questions in issues/discussions
- Reviewing pull requests
- Triaging issues
- Improving processes

## Getting Help

- **Documentation**: https://terrahydro-esdt.github.io/crest/
- **Issues**: https://github.com/terrahydro-esdt/crest/issues
- **Discussions**: https://github.com/terrahydro-esdt/crest/discussions

## Recognition

Contributors are recognized in:
- CONTRIBUTING.md file
- Release notes
- Annual acknowledgments

Thank you for contributing to CREST!
