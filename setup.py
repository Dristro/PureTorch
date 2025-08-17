from setuptools import setup, find_packages
from pathlib import Path

curr_dir = Path(__file__).parent
long_description = (curr_dir/"README.md").read_text(encoding="utf-8")

setup(
    name = "puretorch",
    version = "1.1.1",
    author = "Dhruv",
    description = "Custom implementation of a Neural Network library using numpy. Now supporting autograd.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/Dristro/PureTorch",
    install_requires = [
        "numpy>=2.0.0,<3.0",
        "graphviz>=0.20.0,<1.0"
    ],
    packages = find_packages(),
    python_requires = ">=3.11",  # for new typing related stuff
    include_package_data=True,
)
