from setuptools import setup, find_packages

setup(
    name = "PureTorch",
    version = "v1.0.0+dev",
    author = "Dhruv",
    description = "Raw implementation of PyTorch using NumPy. Structured similar to PyTorch, used to create deep learning models without using PyTorch and TensorFlow.",
    url = "https://github.com/Dristro/PureTorch.git",
    packages = find_packages(),
    install_requires = [
        "numpy>=1.26.0",
    ],
    python_requires = ">=3.8",
)