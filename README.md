***
# Table of contents
|Title|Desc|
|-|-|
|[Version](#version-info)|Current version and features|
|[Purpose](#purpose)|What and why of this repo|
|[Structure](#structure)|File structure of various modules|
|[Upcoming features](#upcoming-features)|Features that im working on|
|[Setup](#setup)|How to setup locally|
****

# Version info
**Current version** - 0.1.3\
**New features** - new Flatten layer added inside PureTorch.layers

*Runs on CPU only. Might add GPU support later.*
****

# Purpose
Raw implementation of PyTorch using NumPy.
The structure and essence of torch remains the same, but its fully implemented using NumPy.
****

# Structure
<u><b>Current file structure</b></u>:
- PureTorch
    - activations
        - ReLU
        - Softmax
        - Tanh
    - layers
        - Sequential
        - Linear
        - Flatten
    - loss
        - CrossEntropyLoss
        - SparseCrossEntropyLoss

Will be adding other layers, activations, losses, optimizers.\
As of now, the default optimizer is SGD.\
To check the "development code" check the "dev" branch
****

# Upcoming features
These are the features that im working on, and will soon be a part of PureTorch.
- Convolutional layer(s)
    - 2D first, then 1D and multi-dim
****

# Setup
Install git before running this command in your env, then run:
```
pip install PureTorch
```
Run: (to verify the installation in python)
```
import PureTorch
print(PureTorch.__version__)
```
or: (to verify the installation on the terminal)
```
python3 -c "import PureTorch; print(PureTorch.__version__)"
```
****