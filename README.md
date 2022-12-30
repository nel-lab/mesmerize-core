# mesmerize-core

[![Linux pip](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml)
[![Linux Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml)
[![MacOS Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml)
[![Windows Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml)
[![Documentation Status](https://readthedocs.org/projects/mesmerize-core/badge/?version=latest)](https://mesmerize-core.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/mesmerize-core.svg)](https://anaconda.org/conda-forge/mesmerize-core)

[![Gitter](https://badges.gitter.im/mesmerize_discussion/mesmerize-viz.svg)](https://gitter.im/mesmerize_discussion/mesmerize-viz?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

### Mesmerize core backend

**News: there will be an in-person workshop in January 2023, more info:** https://indico.flatironinstitute.org/event/3293/

A batch management system for calcium imaging analysis using the CaImAn library. 
It contains `pandas.DataFrame` and `pandas.Series` extensions that interface with CaImAn for running the various algorithms and organizing input & output data.

This **replaces** the [Mesmerize legacy desktop application](https://github.com/kushalkolar/MESmerize).\
`mesmerize-core` is MUCH faster, more efficient, and offers many more features! For example there are simple extensions which you can just call to get the motion correction shifts, CNMF reconstructed movie, CNMF residuals, contours etc.

See the demo notebook at `notebooks/mcorr_cnmf.ipynb` for more details. Note that the demo requires [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) for visualization.

# Documentation

We recommend starting out with the demo notebook ```notebooks/mcorr_cnmf.ipynb```

Documentation is available at: https://mesmerize-core.readthedocs.io/ \
User guide: https://mesmerize-core.readthedocs.io/en/latest/user_guide.html

Please use the GitHub issue tracker for any issues. For smaller questions or discussion use gitter.

gitter: https://gitter.im/mesmerize_discussion/mesmerize-viz

Video tutorial/virtual workshop from September 2022: https://www.youtube.com/watch?v=0AGiAaslJdk

# Overview

![batch_management](https://user-images.githubusercontent.com/9403332/179145962-82317da6-0340-44e4-83ba-7dace0300f55.png)

# Visualization

For visualization we recommend [`mesmerize-viz`](https://github.com/kushalkolar/mesmerize-viz) which contains a standard set of visualizations (a WIP), or [`fastplotlib`](https://github.com/kushalkolar/fastplotlib). Here are some examples of visualizations using `fastplotlib`, these visualizations are all performed within jupyter notebooks therefore they will also work on cloud computing intrastructure!

![interact](https://user-images.githubusercontent.com/9403332/210027293-ea836623-d035-4505-a186-b731126f382a.gif)

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

## For users

The instructions below will install `mesmerize-core`.

For visualization install `fastplotlib` like this into the same environment as `mesmerize-core`:

```bash
pip install git+https://github.com/kushalkolar/fastplotlib.git
```

You may need to install Vulkan drivers depending on your system, see the `fastplotlib` repo for more information: https://github.com/kushalkolar/fastplotlib#install-vulkan-drivers

### Conda

`mesmerize-core` is availabe as a conda package which also gives you CaImAn! These instructions will give you a working `mesmerize-core` along with `caiman` in the same environment.

1. Install `mamba` into your base environment. Skip this step if you have `mamba`. This step may take 10 minutes and display several messages like "Solving environment: failed with..." but it should eventually install `mamba`.
```bash
conda install -c conda-forge mamba

# this command helps prevent things from being slow
conda clean -a
```

2. Create a new environment

```bash
# on linux and mac you can use python=3.10
conda create --name mesmerize-core python=3.10
# on windows you MUST use python=3.9
conda create --name mesmerize-core python=3.9
```
3. Activate environment. You can only use `mesmerize-core` in the environment that it's installed into.

```bash
conda activate mesmerize-core
conda clean -a
```

4. Install `mesmerize-core`

```bash
mamba install -c conda-forge mesmerize-core
```

5. Install `caimanmanager`

```bash
caimanmanager.py install
```

6. Run `ipython` and verify that `mesmerize_core` is installed:

```bash
# run in ipython
import mesmerize_core
mesmerize_core.__version__
```

7. Install `fastplotlib` for visualization into the same environment (run this in the anaconda prompt, not ipython)

```bash
pip install git+https://github.com/kushalkolar/fastplotlib.git
```

### python virtual environments

```bash
# create a new env in some directory
# tested on python3.9 and 3.10
python3.10 -m venv python-venvs/mesmerize-core
source python-venvs/mesmerize-core/bin/activate

# get latest pip setuptools and wheel
pip install --upgrade setuptools wheel pip

# cd into or make a dir that has your repos
mkdir repos
cd repos

# install caiman
git clone https://github.com/flatironinstitute/CaImAn.git
cd CaImAn
pip install -r requirements.txt
pip install .
caimanmanager.py install

# install mesmerize-core
pip install mesmerize-core

# you should now be able to import mesmerize_core
# start ipython
ipython

# run in ipython
import mesmerize_core
mesmerize_core.__version__
```

## For development

### conda

```bash
# install mamba in your base environment
conda install -c conda-forge mamba
conda clean -a

# on linux and mac you can use python=3.10
conda create --name mesmerize-core python=3.10
# on windows you MUST use python=3.9
conda create --name mesmerize-core python=3.9

# activate environment
conda activate mesmerize-core
conda clean -a

# clone this repo
git clone https://github.com/nel-lab/mesmerize-core.git
cd mesmerize-core

# update env with environment file
# this installs caiman as well
mamba env update -n mesmerize-core --file environment.yml

# install caimanmanager
caimanmanager.py install

# install mesmerize-core
pip install .

# install pytest and run tests to make sure everything works properly
mamba install pytest
MESMERIZE_KEEP_TEST_DATA=1 DOWNLOAD_GROUND_TRUTHS=1 pytest -s .
```

### python venvs

```bash
# create a new env in some directory
# tested on python3.9 and 3.10
python3.10 -m venv python-venvs/mesmerize-core
source python-venvs/mesmerize-core/bin/activate

# get latest pip setuptools and wheel
pip install --upgrade setuptools wheel pip

# cd into or make a dir that has your repos
mkdir repos
cd repos

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
