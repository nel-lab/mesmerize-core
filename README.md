# mesmerize-core

[![Linux pip](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml)
[![Linux Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml)
[![MacOS Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml)
[![Windows Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml)
[![Documentation Status](https://readthedocs.org/projects/mesmerize-core/badge/?version=latest)](https://mesmerize-core.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/mesmerize-core.svg)](https://anaconda.org/conda-forge/mesmerize-core)

[![Gitter](https://badges.gitter.im/mesmerize_discussion/mesmerize-viz.svg)](https://gitter.im/mesmerize_discussion/mesmerize-viz?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

### Mesmerize core backend
[**Installation**](https://github.com/nel-lab/mesmerize-core#installation) | [**Examples**](https://github.com/nel-lab/mesmerize-core#examples)

A batch management system for calcium imaging analysis using the CaImAn library. 
It contains `pandas.DataFrame` and `pandas.Series` extensions that interface with CaImAn for running the various algorithms and organizing input & output data.

This **replaces** the [Mesmerize legacy desktop application](https://github.com/kushalkolar/MESmerize).\
`mesmerize-core` is MUCH faster, more efficient, and offers many more features! For example there are simple extensions which you can just call to get the motion correction shifts, CNMF reconstructed movie, CNMF residuals, contours etc.

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

For visualization we strongly recommend [`fastplotlib`](https://github.com/kushalkolar/fastplotlib), a very new but very fast plotting library. Here are some examples of `fastplotlib` visualizations using `mesmerize-core` outputs. You can create these interactive plots within jupyter notebooks, therefore they will also work on cloud computing intrastructure!

### View raw and motion corrected movie side by side:

![mcorr](https://user-images.githubusercontent.com/9403332/210932452-5ed344dd-9a82-41ee-adc5-a9476e1f03a5.gif)

### Contours from CNMF, good components shown here in cyan and bad components in magenta:

![cnmf](https://user-images.githubusercontent.com/9403332/210932670-d797d301-839c-48d9-b11f-3330e076e0e4.gif)

### Input movie, constructed movie `(A * C)`, residuals `(Y - A * C - b * f)`, and reconstructed background `(b * f)`:

![cnmf-rcm](https://user-images.githubusercontent.com/9403332/210932903-b994359b-62d4-49fd-aa6b-cd4855ba873e.gif)

### Interactive Component evaluation after CNMF:

https://user-images.githubusercontent.com/9403332/191207883-2393664d-b5e1-49a5-84d1-8ed7eadcf7a0.mp4

As mentioned, fastplotlib is meant to be a fast plotting library which can handle **millions** of points. You can create highly complex and interactive plots to combine outputs from the CaImAn algorithms with other experimentally relevant analysis, such as behavioral data.

![epic](https://user-images.githubusercontent.com/9403332/210304473-f36f2aaf-319e-435b-bcc8-0e8d3e1ef282.gif)

# Examples

**See the `notebooks` directory for detailed examples.**

Note that [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) is required for the visualizations.

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

# if conda is behaving slow, this command can sometimes help
conda clean -a
```

2. To create a new environment and install `mesmerize-core` into it do this:

```bash
# on linux and mac you can use python=3.10
mamba create -n mescore -c conda-forge mesmerize-core
# `caiman` is a dependency of `mesmerize-core` so it will automatically grab `caiman` too
```

If you already have an environment with `caiman`:

```bash
mamba install -n name-of-env-with-caiman mesmerize-core
```

3. Activate environment. You can only use `mesmerize-core` in the environment that it's installed into.

```bash
mamba activate mesmerize-core
```

4. Install `caimanmanager`

```bash
caimanmanager.py install
```

The `caimanmanager.py` step may cause issues, especially on Windows. Assuming your anaconda is in your user directory a workaround is to call it using the full path:

```bash
python C:\Users\your-username\anaconda3\envs\your-env-name\bin\caimanmanager.py install
```

If you continue to have issues with this step, please post an issue on the caiman github or gitterpip install git+https://github.com/kushalkolar/fastplotlib.git: https://github.com/flatironinstitute/CaImAn/issues 

5. Run `ipython` and verify that `mesmerize_core` is installed:

```bash
# run in ipython
import mesmerize_core
mesmerize_core.__version__
```

6. Install `fastplotlib` for visualization into the same environment (run this in the anaconda prompt, not ipython)

```bash
pip install git+https://github.com/kushalkolar/fastplotlib.git
```

If you don't have git installed you will need to install that first in the environment:

```bash
conda install git
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
