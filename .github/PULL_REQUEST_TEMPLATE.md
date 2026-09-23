<!-- Thanks for contributing! Please fill out the sections below. -->

## Summary

<!-- What does this PR do and why? -->

## Related issue

<!-- e.g. Closes #123 -->

## Checklist

- [ ] I read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] `make check` passes locally (lint + spell + tests)
- [ ] I added or updated tests for my change
- [ ] I updated docs / docstrings where relevant
- [ ] If this changes attack or training behaviour, I confirmed the
      epsilon-ball invariant still holds on data from `get_cifar10_loaders`
      (`pytest tests/test_attacks.py -k pipeline_data`)
- [ ] If this changes reported numbers, I said so explicitly — the README
      carries a notice about results that predate the epsilon-projection fix

## Notes for reviewers

<!-- Anything that needs extra attention. -->
