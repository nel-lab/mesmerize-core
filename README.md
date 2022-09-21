# mesmerize-core

[![Linux pip](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml)
[![Linux Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml)
[![MacOS Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml)
[![Windows Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml)
[![Documentation Status](https://readthedocs.org/projects/mesmerize-core/badge/?version=latest)](https://mesmerize-core.readthedocs.io/en/latest/?badge=latest)

### Mesmerize core backend

**News: there will be a workshop in ~late September, more info:** https://twitter.com/kushalkolar/status/1554927353251262464 

A batch management system for calcium imaging analysis using the CaImAn library. 
It contains `pandas.DataFrame` and `pandas.Series` extensions that interface with CaImAn for running the various algorithms and organizing input & output data.

This **replaces** the [Mesmerize legacy desktop application](https://github.com/kushalkolar/MESmerize).\
`mesmerize-core` is MUCH faster, more efficient, and offers many more features! For example there are simple extensions which you can just call to get the motion correction shifts, CNMF reconstructed movie, CNMF residuals, contours etc.

See the demo notebook at `notebooks/mcorr_cnmf.ipynb` for more details. Note that the demo requires [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) for visualization.

# Documentation

We recommend starting out with the demo notebook ```notebooks/mcorr_cnmf.ipynb```

Documentation is available at: https://mesmerize-core.readthedocs.io/ \
User guide: https://mesmerize-core.readthedocs.io/en/latest/user_guide.html

# Overview

![batch_management](https://user-images.githubusercontent.com/9403332/179145962-82317da6-0340-44e4-83ba-7dace0300f55.png)

# Visualization

For visualization we recommend [`mesmerize-viz`](https://github.com/kushalkolar/mesmerize-viz) which contains a standard set of visualizations (a WIP), or [`fastplotlib`](https://github.com/kushalkolar/fastplotlib). Here are some examples of visualizations using `fastplotlib`, these visualizations are all performed within jupyter notebooks therefore they will also work on cloud computing intrastructure!

### View raw and motion corrected movie side by side:

https://user-images.githubusercontent.com/9403332/191207398-39a027d7-079e-475b-baec-381f2d271652.mp4

### Contours from CNMF, good components in cyan and bad components in magenta:

https://user-images.githubusercontent.com/9403332/191207461-9c5c4cad-867b-413a-b30b-ea61f010eed6.mp4

### Input movie, constructed movie `(A * C)`, residuals `(Y - A * C - b * f)`, and reconstructed background `(b * f)`:

https://user-images.githubusercontent.com/9403332/191207782-566e24bc-7f0d-40a3-9442-37c86d0ebe48.mp4

### Interactive Component evaluation after CNMF:

https://user-images.githubusercontent.com/9403332/191207883-2393664d-b5e1-49a5-84d1-8ed7eadcf7a0.mp4

### This is all possible within jupyter notebooks using `fastplotlib`!

# Examples

**See `notebooks/mcorr_cnmf.ipynb` for detailed examples.** Note that running the demo requires [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) for visualizations.

# Installation

## For development

### conda

```bash
# create an env
conda create --item_name mesmerize-core python=3.10

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
# create a new env
# tested on python3.9 and 3.10
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

# clone this repo and install mesmerize-core
cd ..
git clone https://github.com/nel-lab/mesmerize-core.git
cd mesmerize-core
pip install -e .

# run tests to make sure everything works
MESMERIZE_KEEP_TEST_DATA=1 DOWNLOAD_GROUND_TRUTHS=1 pytest -s .
```
