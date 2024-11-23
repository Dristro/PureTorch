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
- Introduced `PureTorch.nn.Perceptron` as a building block for layers.
- Introduced `PureTorch.nn.Sequential`, allows you to create a model in a sequential manner.

### Deprecated
- Removed `PureTorch.layers.x`.
- Temporarily removed activations, losses, and optimizers (will return soon).
