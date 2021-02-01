from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

requires = [
    "torch",
    "numpy",
    "pandas",
    "psutil",
    "threadpoolctl",
    "scikit-learn>=0.23",
    "importlib_metadata>=0.7; python_version < '3.8'",
]

setup(
    name="harmony-pytorch",
    use_scm_version=True,
    description="Pytorch implementation of Harmony algorithm on single-cell sequencing data integration",
    long_description=long_description,
    url="https://github.com/lilab-bcb/harmony-pytorch",
    author="Yiming Yang, Bo Li",
    author_email="yyang43@mgh.harvard.edu, bli28@mgh.harvard.edu",
    classifiers=[ # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="single-cell genomics data integration",
    packages=find_packages(),
    install_requires=requires,
    setup_requires=["setuptools_scm"],
    python_requires="~=3.5",
)
