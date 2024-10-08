|Title|Desc|
|-|-|
|[Version](#version-info)|Current version and features|
|[Purpose](#purpose)|What and why|
|[Structure](#structure)|File structure|
|[Issues](#issues)|Known issues|
|[Setup](#setup)|How to setup locally|
****

# Version info
**Current version** - 0.1.1\
**New features** - added the training loop inside Sequential, you can train by using Sequential.train()

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
    - loss
        - CrossEntropyLoss
        - SparseCrossEntropyLoss

Will be adding other layers, activations, losses, optimizers.\
As of now, the default optimizer is SGD.\
To check the "development code" check the "dev" branch (will add it soon)
****

# Issues
<u>Known issues:</u>
- Training only works if (data_size % batch_size) == 0, else the code breaks
    - Bug fixed (in version 0.1.1)
****

# Setup
Install git before running this command in your env
```
pip install "git+https://github.com/Dristro/PureTorch"
```
Then run: (to verify the installation)
```
import PureTorch
PureTorch.__version__
```
****