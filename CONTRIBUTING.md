# Contributing to Solar PV Forecasting Benchmark

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Solar PV Forecasting Benchmark repository.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Commit Guidelines](#commit-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Reporting Bugs](#reporting-bugs)
8. [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Assume good intentions
- Accept feedback gracefully

## How Can I Contribute?

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Found a bug? Let us know!
2. **Feature Requests**: Have an idea for improvement? Share it!
3. **Code Contributions**: 
   - Fix bugs
   - Add new models
   - Improve documentation
   - Optimize performance
   - Add tests
4. **Documentation**: 
   - Improve README
   - Add tutorials
   - Fix typos
   - Translate documentation

### Good First Issues

Look for issues tagged with `good-first-issue` - these are suitable for newcomers.

## Development Setup

1. **Fork the repository**

2. **Clone your fork**:
```bash
git clone https://github.com/yourusername/Solar_PV_Forecasting_Benchmark.git
cd Solar_PV_Forecasting_Benchmark
```

3. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

5. **Create a branch**:
```bash
git checkout -b feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized with `isort`

### Code Formatting

We use [Black](https://github.com/psf/black) for code formatting:

```bash
black .
```

### Linting

We use `flake8` for linting:

```bash
flake8 --max-line-length=88 --extend-ignore=E203
```

### Type Hints

Use type hints where appropriate:

```python
def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(X_train, y_train, model_type="xgboost"):
    """Train a forecasting model.
    
    Args:
        X_train: Training features as numpy array or pandas DataFrame
        y_train: Training targets as numpy array or pandas Series
        model_type: Type of model to train (default: "xgboost")
        
    Returns:
        Trained model object
        
    Raises:
        ValueError: If model_type is not supported
        
    Example:
        >>> X_train, y_train = load_data()
        >>> model = train_model(X_train, y_train, "random_forest")
    """
    pass
```

## Adding New Models

To add a new forecasting model:

1. Create a new file in `models/` directory:
```python
# models/your_model.py

class YourModel:
    """Your model description."""
    
    def __init__(self, **kwargs):
        """Initialize model with hyperparameters."""
        pass
    
    def fit(self, X_train, y_train):
        """Train the model."""
        pass
    
    def predict(self, X_test):
        """Generate predictions."""
        pass
```

2. Add tests in `tests/test_your_model.py`

3. Update documentation

4. Add to benchmark comparison script

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(models): add LightGBM model implementation

- Implemented LightGBM regression model
- Added hyperparameter optimization
- Updated benchmarking script

Closes #123
```

```
fix(data): correct clearness index calculation

The previous formula didn't handle edge cases where G0h = 0.
Now properly filters out nighttime hours before calculation.

Fixes #456
```

## Pull Request Process

1. **Update your branch**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run tests**:
```bash
pytest tests/
```

3. **Run linters**:
```bash
black .
flake8 .
```

4. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

5. **Create Pull Request**:
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what changed and why
   - Include screenshots for UI changes
   - Ensure all CI checks pass

6. **Code Review**:
   - Address reviewer comments
   - Make requested changes
   - Re-request review when ready

7. **Merge**:
   - Maintainer will merge when approved
   - Delete your branch after merge

## Reporting Bugs

### Before Submitting

1. Check existing issues
2. Try latest version
3. Check documentation

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run script '...'
2. With parameters '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.8.10]
- Package versions: [run `pip freeze`]

**Additional context**
Any other relevant information.
```

## Suggesting Enhancements

### Enhancement Proposal Template

```markdown
**Is your feature related to a problem?**
A clear description of the problem.

**Proposed solution**
Describe your proposed solution.

**Alternatives considered**
Other solutions you've considered.

**Additional context**
Any other relevant information.
```

## Testing

### Writing Tests

All new code should include tests:

```python
# tests/test_models.py

import pytest
from models.xgboost_model import XGBoostModel

def test_xgboost_training():
    """Test XGBoost model training."""
    model = XGBoostModel()
    X_train, y_train = generate_sample_data()
    model.fit(X_train, y_train)
    assert model.is_fitted()
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=models --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::test_xgboost_training
```

## Documentation

### Adding Documentation

- Keep README.md up to date
- Document all functions and classes
- Add examples to docstrings
- Update REPRODUCIBILITY.md for methodology changes

### Building Documentation (if using Sphinx)

```bash
cd docs/
make html
```

## Questions?

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: your.email@institution.edu

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in paper acknowledgments (for significant contributions)
- Cited in derivative works

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to advancing solar energy forecasting research! 🌞
