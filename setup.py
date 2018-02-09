import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

setup(
    name="pygrassml",
    version="0.1",
    author="Steven Pawley",
    author_email="steven.pawley@gmail.com",
    description=("Package to facilitate the application of scikit-learn classification and regression models to GRASS GIS spatial data"),
    license="GNU",
    keywords="GRASS GIS",
    url="https://github.com/stevenpawley/PygrassML",
    packages=["pygrassml"]
)