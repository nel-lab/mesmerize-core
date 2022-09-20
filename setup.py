from setuptools import setup, find_packages
from pathlib import Path


install_requires = [
    "pandas>=1.5.0",
    "pytest",
    "requests",
    "tqdm",
    "numpy",
    "matplotlib",
    "click",
    "psutil",
    "pims",
    "jupyterlab",
]


with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()

with open(Path(__file__).parent.joinpath("mesmerize_core", "VERSION"), "r") as f:
    ver = f.read().split("\n")[0]


classifiers = \
    [
        "Programming Language :: Python :: 3",
        "LICENSE :: OSI APPROVED :: APACHE SOFTWARE LICENSE",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research"
    ]


setup(
    name="mesmerize-core",
    description="High level pandas-based API for batch analysis of Calcium Imaging data using CaImAn",
    long_description=readme,
    classifiers=classifiers,
    version=ver,
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/nel-lab/mesmerize-core",
    license="Apache-Software-License",
    author="Kushal Kolar, Caitlin Lewis, Arjun Putcha",
    author_email="",
)
