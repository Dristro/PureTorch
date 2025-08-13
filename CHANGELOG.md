# Changelog

All notable changes to this project will be documented here.\
This project adheres to [Semantic Versioning](https://semver.org/).\
(Documented from v1.0.0+dev and onwards. Will add previous version documentation soon)
<br></br>
**Date formatting**: YYYY/MM/DD

---

## [v1.0.0+dev] - 2024-11-24
### Added
- Introduced `PureTorch.Tensor`.
- Added `PureTorch.nn` module with PyTorch-like syntax.
- Introduced `PureTorch.nn.Sequential`, allows you to create a model in a sequential manner.

*WIll deprecate `PureTorch.nn.Perceptron` by the next commit.*

### Deprecated
- Removed `PureTorch.layers.x`.
- Temporarily removed activations, losses, and optimizers (will return soon).

---

## [v1.1.0+dev] - 2025-08-??

### Added
- `autograd`: autograd framework for puretorch
  - Complete autograd support. More efficient graph initialization and backward-operations.
  - Allows users to define custom funtions with `.forward` and `.backward` logic.

### Depricated
- `PureTorch`: now named as `puretorch`, removed capitalization.
- `PureTorch.nn.Perceptron`: no longer in use, replace with `nn.Linear` according to use case.

### General Changes in src
Logical changes in src, not effecting user-workflow. So previous functionality is not broken, while allowing new features.

- `puretorch.tensor`: abstraction of `autograd.tensor` with neural-network operations in mind.
- Seamless autograd intigration, allowing faster model training due to new graph-construction and gradient-ops from `autograd`.