# Contributing to Underconfidence Adversarial Training

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/underconfidence-adversarial-training.git
   cd underconfidence-adversarial-training
   ```

3. Install the package and dev tools:
   ```bash
   pip install -e ".[dev]"   # or: make install
   ```

4. Install the pre-commit hooks:
   ```bash
   make precommit
   ```
   These run black, isort, codespell and basic file hygiene on every commit,
   using the same versions CI lints with.

5. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

This project follows standard Python conventions:

- **Formatting**: Use `black` (configured in `pyproject.toml` for a 100 character line length); just run `make format`
- **Import sorting**: Use `isort` (black profile, also configured in `pyproject.toml`)
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Type hints**: Add type hints where appropriate
- **Spelling**: `codespell` runs over the tree; the project is currently clean
  with no ignore list

Run formatting tools before committing:
```bash
make format  # Format code
make lint    # Check formatting
make spell   # Spell check
```

## Testing

All new features should include tests:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_attacks.py -v
```

Test guidelines:
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for >80% code coverage

## Pull Request Process

1. **Update documentation**: Add/update docstrings, README sections, and comments as needed

2. **Add tests**: Ensure your changes are covered by tests

3. **Run checks**:
   ```bash
   make format  # Format code
   make check   # Everything CI gates on: lint + spell + test
   ```

   A green `make check` should mean green CI. If it does not, that is a bug in
   the Makefile or the workflow, worth reporting.

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**: Go to the original repository and open a PR from your branch

## Continuous integration

Every pull request runs `.github/workflows/ci.yml`:

| Job | What it guards |
|-----|----------------|
| `lint` | black, isort, codespell |
| `test` | the suite on Linux (3.11, 3.12) and macOS (3.12); Linux 3.13 runs as a non-blocking early warning |
| `smoke` | `tests/smoke_test_comprehensive.py` and `experiments/verify_attacks.py` — the real entry points, which no unit test imports |
| `clean-install` | builds the wheel, installs it into a fresh venv and runs `verify_install.py`, so a broken package layout cannot hide behind an editable install from the repo root |
| `min-versions` | resolves the *lowest* declared dependency versions, so the floors in `pyproject.toml` stay honest — the `torch>=2.6.0` floor matters most, because 2.6 is where `torch.load` flipped to `weights_only=True` |
| `ci-ok` | the single required status check; fails closed if any job did not succeed or get skipped |

Set branch protection to require only `ci-ok`.

Note that `tests/test_compatibility.py` asserts the declared floors (Python
3.11+, torch 2.6+, torchvision 0.21+). Those assertions are deliberate: they
fail on an environment the project does not claim to support, so they will fail
locally on an older interpreter even when nothing is wrong with your change.

## Contribution Ideas

Areas where contributions are especially welcome:

### New Features
- Additional datasets (CIFAR-100, ImageNet, MSTAR)
- Additional architectures (Wide ResNets, Vision Transformers)
- Visualization tools (t-SNE, decision boundaries)
- Multi-GPU training support

### Documentation
- Tutorial notebooks
- Architecture diagrams
- Performance benchmarks
- Use case examples

### Bug Fixes
- Report bugs via GitHub Issues
- Include minimal reproducible example
- Specify environment (OS, Python version, PyTorch version)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on improving the project
- Help newcomers get started

## Questions?

- Open a GitHub Issue for questions
- Tag issues with appropriate labels (bug, feature, question, etc.)
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
