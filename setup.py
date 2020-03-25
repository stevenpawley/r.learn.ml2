from setuptools import setup, find_packages

setup(
    # package metadata
    name="r.learn.ml",
    version="0.0.1",
    author="Steven Pawley",
    author_email="steven.pawley@gmail.com",
    description=("Machine learning in GRASS GIS"),
    license="GNU",
    keywords="GIS",
    url="https://github.com/stevenpawley/r.learn.ml2",

    # files/directories to be installed with package
    packages=find_packages(),

    # package dependencies
    install_requires=[
        'numpy>=1.10.0',
        'scipy>1.0.0',
        'pandas>=0.20',
        'matplotlib>=2.2.4',
        'sklearn'],
    python_requires='>=3.5',

    # testing
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
