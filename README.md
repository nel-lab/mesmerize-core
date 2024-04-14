# mesmerize-core

[![Linux pip](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-pip.yml)
[![Linux Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/linux-conda.yml)
[![MacOS Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/macos-conda.yml)
[![Windows Conda](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml/badge.svg)](https://github.com/nel-lab/mesmerize-core/actions/workflows/windows-conda.yml)
[![Documentation Status](https://readthedocs.org/projects/mesmerize-core/badge/?version=latest)](https://mesmerize-core.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/mesmerize-core.svg)](https://anaconda.org/conda-forge/mesmerize-core)

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

## Getting Help

Please use the GitHub issue tracker for any issues, and discussions for discussions. We no longer use gitter. If you are in the Slack, we usually only respond on slack around workshops. GitHub is for long term support.

Video tutorial/virtual workshop from September 2022 (quite outdated, if you have question post on github): https://www.youtube.com/watch?v=0AGiAaslJdk

# Overview

![batch_management](https://user-images.githubusercontent.com/9403332/179145962-82317da6-0340-44e4-83ba-7dace0300f55.png)

# Visualization

Install [mesmerize-viz](https://github.com/kushalkolar/mesmerize-viz) for visualization. Mesmerize-viz gives you ready-to-use viuslizations for motion correction and CNMF - including component exploration, interactive component evaluation using metrics, and manual addition or removal of components. 

```
pip install mesmerize-viz
```

:exclamation: **Harware requirements** The large CNMF visualizations with contours etc. usually require either a dedicated GPU or integrated GPU with access to at least 1GB of VRAM. 

You may need to install drivers depending on your system, see the `fastplotlib` repo for more information: https://github.com/kushalkolar/fastplotlib#graphics-drivers

If you use `fastplotlib` directly you can create highly complex and interactive plots to combine outputs from the CaImAn algorithms with other experimentally relevant analysis, such as behavioral data.

![epic](https://user-images.githubusercontent.com/9403332/210304473-f36f2aaf-319e-435b-bcc8-0e8d3e1ef282.gif)

# Examples

**See the `notebooks` directory for detailed examples.**

# Installation

## For users

The instructions below will install `mesmerize-core`.

### Conda

`mesmerize-core` is available as a conda package which also gives you CaImAn. These instructions will give you a working `mesmerize-core` along with `caiman` in the same environment.

**Important note: Sometimes conda or mamba will get stuck at a step, such as creating an environment or installing a package. Pressing `Enter` on your keyboard can sometimes help it continue when it pauses.**

1. Install `mamba` into your base environment. Skip this step if you have `mamba`. This step may take 10 minutes and display several messages like "Solving environment: failed with..." but it should eventually install `mamba`.

```bash
conda install -c conda-forge mamba

# if conda is behaving slow, this command can sometimes help
conda clean -a
```

2. To create a new environment and install `mesmerize-core` into it do this:

```bash
mamba create -n mescore -c conda-forge mesmerize-core
```

`caiman` is a dependency of `mesmerize-core` so it will automatically grab `caiman` too

If you already have an environment with `caiman`:

```bash
mamba install -n name-of-env-with-caiman mesmerize-core
```

3. Activate environment. You can only use `mesmerize-core` in the environment that it's installed into.

```bash
mamba activate mescore
```

4. Install `caimanmanager`

If you are using an older version of `caiman` < 1.9.16, then please see [Step 4 in the old README](https://github.com/nel-lab/mesmerize-core/blob/09a81f856a6728cc3aff62f967d2dce308214c63/README.md#conda).

```bash
caimanmanager install
```

If you have issues with this step, please post an issue on the caiman github or gitter: https://github.com/flatironinstitute/CaImAn/issues 

5. Run `ipython` and verify that `mesmerize_core` is installed:

```bash
# run in ipython
import mesmerize_core
mesmerize_core.__version__
```

6. Install `mesmerize-viz` for visualization into the same environment (run this in the anaconda prompt, not ipython). You may also need to install graphics drivers depending on your system, see the `fastplotlib` repo for more information: https://github.com/kushalkolar/fastplotlib#graphics-drivers

```bash
pip install mesmerize-viz
```

**Strongly recommended: install `simplejpeg` for much faster notebook visualization, you will need C compilers and [libjpeg-turbo](https://libjpeg-turbo.org/) for this to work:**

```bash
pip install simplejpeg
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
caimanmanager install

# install mesmerize-core
pip install mesmerize-core

# install mesmerize-viz
pip install mesmerize-viz

# install simplejpeg
pip install simplejpeg

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

conda create --name mesmerize-core

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
caimanmanager install

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
caimanmanager install

# clone this repo and install mesmerize-core
cd ..
git clone https://github.com/nel-lab/mesmerize-core.git
cd mesmerize-core
pip install -e .

# run tests to make sure everything works
MESMERIZE_KEEP_TEST_DATA=1 DOWNLOAD_GROUND_TRUTHS=1 pytest -s .
```
