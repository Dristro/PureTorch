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

**Current Version**: **1.0.0+dev**
### Migration Guide
If you're upgrading from `v0.x.x` to `v1.0.0-beta`, please refer to the [Migration Guide](MIGRATIONGUIDE.md) for detailed instructions on updating your codebase.

The guide covers:
- Breaking changes in the API.
- Updated module structures.
- Examples of how to transition your code.

Click [here](MIGRATIONGUIDE.md) to view the Migration Guide.
<br></br>

### New Features
1. **`PureTorch.Tensor`**:
   - A NumPy array wrapper to track gradients for efficient back-propagation.
   - Back-propagation is easier and more convenient.
2. **`PureTorch.nn`**:
   - New modules replicate PyTorch's syntax and usability.
3. **Perceptron as the Building Block**:
   - Located at `PureTorch.nn.Perceptron`, this component will help build layers like `Linear`.
<br></br>

### Deprecations
1. The **PureTorch.layers.x** system is replaced with a simpler, PyTorch-like API.
2. Temporarily removed activations, losses, and optimizers. These will return with updated functionality, supporting Tensors and Perceptrons.
<br></br>

*The current implementation of a Tensor is inspired by **Andrej Karpathy's 'Micrograd'***
<br></br>

*Runs on CPU only. Might add GPU support later.*

---

# Purpose
Raw implementation of PyTorch using NumPy.
The structure and essence of torch remains the same, but its  implemented using NumPy.

---

# Structure
<u><b>New file structure</b></u>:
- PureTorch
    - nn
        - Linear
        - Perceptron
    - Tensor

<u><b>Old file structure</b></u>:
- PureTorch
    - activations
        - ReLU
        - Softmax
        - Tanh
    - layers
        - Sequential
        - Linear
        - Flatten
        - Conv2D
    - loss
        - CrossEntropyLoss
        - SparseCrossEntropyLoss

Will be adding other layers, activations, losses, optimizers.

---

# Upcoming features
These are the features that im working on, and will soon be a part of PureTorch.
- Restoring the same functionality as PureTorch **v0.1.x**

---

# Setup
***Note**:
this for setting-up the "dev" branch locally.\
**For the stable installation**: go to the "main" branch's setup guide\

### Prerequisites
- Python 3.8 or higher
- Git installed on your system


### Installation Steps
1. **Install the development branch:**
```bash
pip install "git+https://github.com/Dristro/PureTorch@dev"
```
2. **Verify installation in Python**
```Python
import PureTorch
print(PureTorch.__version__)
```
3. **Verify installation in the terminal**
```bash
python3 -c "import PureTorch; print(PureTorch.__version__)"
```


If the version indicated is: **1.0.0+dev**, then the package was installed correctly.\
If not, try reinstalling the package (or) check if you installed the stable (vs) development package.

---

# Report Bugs
If you encounter issues or have suggestions, please open an issue via the [GitHub Issues tab](https://github.com/Dristro/PureTorch/issues). 

For those interested in contributing:
- Fork this repository.
- Make your changes.
- Open a pull request with a detailed description of your changes.

*Test cases will be added soon to help verify contributions.*

---