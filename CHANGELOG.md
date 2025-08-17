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

*WIll deprecate `PureTorch.nn.Perceptron` by next version release.*

### Deprecated
- Removed `PureTorch.layers.x`.
- Temporarily removed activations, losses, and optimizers (will return soon).

---

## [1.1.0] - 2025-08-18

### Added
- `autograd`: autograd framework for puretorch:
  - Complete autograd support. More efficient graph initialization and backward-operations.
  - Allows users to define custom funtions with `.forward` and `.backward` logic.
- `nn.functional`: all functional stuff, like `.relu()`, `.cross_entropy()`, etc.
- `nn.Module`, base class for all modules.
- activation functions `nn.relu`, `nn.softmax`, `nn.tanh` (all call implementation at functional).
- `puretorch.utils`, allows user to call functions like `puretorch.zeros_like()`, etc.
  - user can now init new tensor-instances using `puretorch.tensor(shape)`
- better abstraction from autograd's Varaible to Tensor, allows for faster integration of device etc later down-the-line.

### Depricated
- `PureTorch`: now named as `puretorch`, removed capitalization.
- `PureTorch.nn.Perceptron`: no longer in use, replace with `nn.Linear` according to use case.

### General Changes in src
Logical changes in src, not effecting user-workflow. So previous functionality is not broken, while allowing new features.

- `puretorch.Tensor`: abstraction of `autograd.Variable` with neural-network operations in mind.
- Seamless autograd intigration, allowing faster model training due to new graph-construction and gradient-ops from `autograd`.
 