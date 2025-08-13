from setuptools import setup, find_packages

setup(
    name = "PureTorch",
    version = "v1.1.0+dev",
    author = "Dhruv",
    description = "Custom implementation of a Neural Network library using numpy. Now supports autograd.",
    url = "https://github.com/Dristro/PureTorch.git",
    packages = find_packages(),
    install_requires = [
        "numpy>=1.26.0",
    ],
    python_requires = ">=3.8",
)