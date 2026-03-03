# Contributing to Underconfidence Adversarial Training

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/underconfidence-adversarial-training.git
   cd underconfidence-adversarial-training
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black isort  # Development tools
   ```

4. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

This project follows standard Python conventions:

- **Formatting**: Use `black` with default settings (88 character line length)
- **Import sorting**: Use `isort` with default settings
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Type hints**: Add type hints where appropriate

Run formatting tools before committing:
```bash
make format  # Format code
make lint    # Check formatting
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
   make test    # Run tests
   make lint    # Check formatting
   ```

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

## Contribution Ideas

Areas where contributions are especially welcome:

### New Features
- Additional datasets (CIFAR-100, ImageNet, MSTAR)
- Additional architectures (Wide ResNets, Vision Transformers)
- Confidence calibration metrics (ECE, MCE, Brier score)
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
