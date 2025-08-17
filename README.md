# PureTorch

**PureTorch** is a NumPy-based implementation of PyTorch. It aims to replicate the functionality and user experience of PyTorch while offering a transparent and lightweight codebase for educational and experimental purposes.

---

# Table of Contents
| Section              | Description                              |
|----------------------|------------------------------------------|
| [Version](#version-info) | Current version and features           |
| [Purpose](#purpose)      | The what and why of this repository    |
| [Structure](#structure)  | File structure of various modules      |
| [Upcoming Features](#upcoming-features) | Features under development           |
| [Setup](#setup)          | Instructions for local setup           |
| [Report Bugs](#report-bugs) | How to report issues or contribute    |


---

# Version Info

**Current Version**: **1.1.0+dev**
### Migration Guide
If you're upgrading from `v0.x.x` to `v1.x.x-dev`, please refer to the [Migration Guide](MIGRATIONGUIDE.md) for detailed instructions on updating your codebase.

The guide covers:
- Breaking changes in the API.
- Updated module structures.
- Examples of how to transition your code.

Click [here](MIGRATIONGUIDE.md) to view the Migration Guide.
<br></br>

### New Features
1. **`puretorch.Tensor`**:
   - autograd.tensor wrapper, tracks gradients for efficient back-propagation.
   - back-propagation is easier and more convenient.
2. **`puretorch.nn`**:
   - neural network modules like: `linear`, `sequential`, more.
3. **puretorch.optim**:
   - optimizers like: `SGD`, other (will add more soon).
4. **autograd**:
   - autograd is now supported, allowing users to build custom functions and give full control over gradnients.
<br></br>

### Deprecations
1. The **puretorch.layers.x** system is replaced with a simpler, PyTorch-like API.
2. Temporarily removed activations, losses, and optimizers. These will return with updated functionality, supporting Tensors.
<br></br>

> *Runs on CPU only. Might add GPU support later.*

---

# Purpose
Raw implementation of PyTorch-like library using NumPy.
The structure and essence of torch remains the same, but its  implemented using NumPy.

---

# Structure
<u><b>New file structure</b></u>:
- puretorch
    - nn
        - linear
        - Perceptron (will be deprecated in v1.1.0)
        - sequential
    - optim
      - optimizer
      - sgd
    - tensor (autograd.tensor, modified for better nn compatibility)
- autograd
  - context
  - engine (not in use now. Abstractions of tensor-ops will be added here, along with higher level tensor logic)
  - function
  - ops
  - tensor

Will be adding other layers, activations, losses and optimizers.

---

# Upcoming features
These are the features that im working on, and will soon be a part of PureTorch.
1. More optimizers (like sgd w/ momentum, adam, etc)
2. Loss functions
3. Model summary (like torchinfo.summary())

---

# Setup
> **Note**:
this for setting-up the "dev" branch locally.\
> **For stable installation**: go to the "main" branch's setup guide

### Prerequisites
- Python 3.8 or higher
- Git installed on your system


### Installation Steps
1. **Install the development branch:**
```bash
pip install PureTorch
```
2. **Verify installation in Python**
```Python
import puretorch
print(puretorch.__version__)
```
3. **Verify installation in the terminal**
```bash
python3 -c "import puretorch; print(puretorch.__version__)"
```

If the version indicated is: **1.1.0**, then the package was installed correctly.\
If not, try reinstalling the package (or) check if you installed the stable (vs) development package.

---

# Report Bugs
If you encounter issues or have suggestions, please open an issue via the [GitHub Issues tab](https://github.com/Dristro/PureTorch/issues). 

For those interested in contributing:
- Fork this repository.
- Make your changes.
- Open a pull request with a detailed description of your changes.

*Test cases will be added soon to help verify contributions and new features.*

---