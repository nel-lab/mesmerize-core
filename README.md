# mesmerize-core

![Python Package](https://github.com/nel-lab/mesmerize-core/actions/workflows/python-app.yml/badge.svg) ![Conda install](https://github.com/nel-lab/mesmerize-core/actions/workflows/python-package-conda.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/mesmerize-core/badge/?version=latest)](https://mesmerize-core.readthedocs.io/en/latest/?badge=latest)

Mesmerize core backend

**News: there will be a workshop in ~late September, more info:** https://twitter.com/kushalkolar/status/1554927353251262464 

**Note: We're currently waiting for the release of pandas v1.5 before the initial release of mesmerize-core**

A batch management system for calcium imaging analysis using the CaImAn library. 
It contains `pandas.DataFrame` and `pandas.Series` extensions that interface with CaImAn for running the various algorithms and organizing input & output data.

This **replaces** the [Mesmerize legacy desktop application](https://github.com/kushalkolar/MESmerize).\
`mesmerize-core` is MUCH faster, more efficient, and offers many more features! For example there are simple extensions which you can just call to get the motion correction shifts, CNMF reconstructed movie, CNMF residuals, contours etc.

See the demo notebook at `notebooks/mcorr_cnmf.ipynb` for more details. Note that the demo requires [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) for visualization.

# Documentation

We recommend starting out with the demo notebook ```notebooks/mcorr_cnmf.ipynb```

API Documentation is available at: https://mesmerize-core.readthedocs.io/

# Visualization

For visualization we recommend [`mesmerize-viz`](https://github.com/kushalkolar/mesmerize-viz) which contains a standard set of visualizations (a WIP), or [`fastplotlib`](https://github.com/kushalkolar/fastplotlib). You can also use the `mesmerize-napari` plugin for smaller datasets.

# Overview

![batch_management](https://user-images.githubusercontent.com/9403332/179145962-82317da6-0340-44e4-83ba-7dace0300f55.png)

# Installation

## For development

### conda

```bash
# create an env, can be same env you use for mesmerize-napari or another viz tool like fastplotlib
conda create --name mesmerize-core python=3.10

# install mamba
conda install -c conda-forge mamba
conda clean -a

# activate env
conda activate mesmerize-core

# clone this repo
git clone https://github.com/nel-lab/mesmerize-core.git
cd mesmerize-core

# update env with environment file
mamba env update -n mesmerize-core --file environment.yml

# temporary until pandas v1.5.0 is released
pip install git+https://github.com/pandas-dev/pandas.git

# install caimanmanager
caimanmanager.py install

# install mesmerize-core
pip install -e .

# install pytest and run tests
mamba install pytest
MESMERIZE_KEEP_TEST_DATA=1 DOWNLOAD_GROUND_TRUTHS=1 pytest -s .
```

### python venvs
```bash
# create a new env, can be the same env that you use for mesmerize-napari or another viz tool like fastplotlib
python3.10 -m venv ~/python-venvs/mesmerize-core
source ~/python-venvs/mesmerize-core/bin/activate

# get latest pip setuptools and wheel
pip install --upgrade setuptools wheel pip

# cd into or make a dir that has your repos
cd ~/repos

# install caiman
git clone https://github.com/flatironinstitute/CaImAn.git
cd CaImAn
pip install -r requirements.txt
pip install .
caimanmanager.py install

# clone this repo
cd ..
git clone https://github.com/nel-lab/mesmerize-core.git
cd mesmerize-core

# get dependencies
pip install -r requirements.txt

# temporary until pandas v1.5.0 is released
pip install git+https://github.com/pandas-dev/pandas.git

# install mesmerize-core
pip install -e .

# install pytest and run tests
pip install pytest
MESMERIZE_KEEP_TEST_DATA=1 DOWNLOAD_GROUND_TRUTHS=1 pytest -s .
```

# Examples demonstrating the API

List of API examples: https://mesmerize-core.readthedocs.io/en/latest/examples.html

**See `notebooks/mcorr_cnmf.ipynb` for more detailed examples.** Note that running the demo requires [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) for visualizations.


