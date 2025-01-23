from pathlib import Path
from setuptools import setup, find_packages

setup(
    name = "PureTorch",
    version = "0.1.2",
    description = "Raw implementation of PyTorch using NumPy. Used to create deep-learning models.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url = "https://github.com/Dristro/PureTorch.git",
    author = "Dhruv N",
    author_email="dhruvn853@gmail.com",
    license="MIT",
    project_urls={
        "Source": "https://github.com/Dristro/PureTorch",
        "Dev branch": "https://github.com/Dristro/PureTorch/tree/dev",
        "Report issues": "https://github.com/Dristro/PureTorch/issues"
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9,<3.12",
    install_requires = [
        "numpy>=1.26.0,<2.0",
        "setuptools",
    ],
    packages = find_packages(),
    include_package_data=True,
)