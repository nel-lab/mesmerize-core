# mesmerize-core

![Python Package](https://github.com/nel-lab/mesmerize-core/actions/workflows/python-app.yml/badge.svg) ![Conda install](https://github.com/nel-lab/mesmerize-core/actions/workflows/python-package-conda.yml/badge.svg) 


Mesmerize core backend

**News: there will be a workshop in ~late September, more info:** https://twitter.com/kushalkolar/status/1554927353251262464 

**Note: We're currently waiting for the release of pandas v1.5 before the initial release of mesmerize-core**

A batch management system for calcium imaging analysis using the CaImAn library. 
It contains `pandas.DataFrame` and `pandas.Series` extensions that interface with CaImAn for running the various algorithms and organizing input & output data.

This **replaces** the [Mesmerize legacy desktop application](https://github.com/kushalkolar/MESmerize).\
`mesmerize-core` is MUCH faster, more efficient, and offers many more features! For example there are simple extensions which you can just call to get the motion correction shifts, CNMF reconstructed movie, CNMF residuals, contours etc.

See the demo notebook at `notebooks/mcorr_cnmf.ipynb` for more details. Note that the demo requires [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) for visualization.

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

**See `notebooks/mcorr_cnmf.ipynb` for more detailed examples.** Note that running the demo requires [`fastplotlib`](https://github.com/kushalkolar/fastplotlib) for visualizations.

## Motion Correction

```python
from mesmerize_core import *
from matplotlib import pyplot as plt

# set the parent directory as the top-level directory for your experiment data
set_parent_raw_data_path('/home/kushal/my_exps_dir')

batch_path = '/home/kushal/my_exps_dir/my_batches/exp_1_batch.pickle'

# create a new batch
df = create_batch(batch_path)

# path to raw movie tiff file
movie_path = '/home/kushal/my_exps_dir/exp_1/my_movie.tif'

# params, exactly the same as what you'd directly use with CaImAn
mcorr_params1 =\
{
  'main': # this key is necessary for specifying that these are the "main" params for the algorithm
    {
        'max_shifts': [24, 24],
        'strides': [48, 48],
        'overlaps': [24, 24],
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
        'pw_rigid': True,
        'gSig_filt': None
    },
}

# add an item to the batch
df.caiman.add_item(
  algo='mcorr',
  name='my_movie',
  input_movie_path=movie_path,
  params=mcorr_params1
)

# We create another set of params, useful for gridsearches to find optimal parameters
mcorr_params2 =\
{
  'main':
    {
        'max_shifts': [24, 24],
        'strides': [24, 24],
        'overlaps': [12, 12],
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
        'pw_rigid': True,
        'gSig_filt': None
    },
}

# add other param variant to the batch
df.caiman.add_item(
  algo='mcorr',
  name='my_movie',
  input_movie_path=movie_path,
  params=mcorr_params2
)

# run the first "batch item"
process = df.iloc[0].caiman.run()
process.wait()

# run the second item
# you can also use a loop to run all these items
# just call process.wait() to run them one after another
process = df.iloc[1].caiman.run()

# get the mot corrected video
# fastplotlib can be used for fast random-access visualization in notebooks

mcorr_movie = df.iloc[-1].mcorr.get_output()

# get the x-y shifts
# you can plot these as a line plot
shits = df.iloc[-1].mcorr.get_shifts()
```

## CNMF

```python
# We can continue from mcorr above and perform CNMF using the mcorr output

# some params for CNMF
params_cnmf =
{
    'main':  # indicates that these are the "main" params for the CNMF algo
        {
            'p': 1,
            'gnb': 2,
            # raises error: no parameter 'merge_thresh' found
            'merge_thr': 0.85,
            'rf': 15,
            'stride_cnmf': 6,
            'K': 4,
            'gSig': [4, 4],
            'ssub': 1,
            'tsub': 1,
            'method_init': 'greedy_roi',
            'min_SNR': 2.0,
            'rval_thr': 0.7,
            'use_cnn': True,
            'min_cnn_thr': 0.8,
            'cnn_lowest': 0.1,
            'decay_time': 0.4,
        },
    'refit': True,  # If `True`, run a second iteration of CNMF
}

df.caiman.add_item(
    algo='cnmf',
    name='my_movie',
    input_movie_path=df.iloc[0].mcorr.get_output_path(),  # use mcorr output from a previous item
    params=params_cnmf
)

# run this item
process = df.iloc[-1].caiman.run()
process.wait()

# we can look at the spatial components for example
# see th demo notebook for an example that uses fastplotlib to visualize contours with the movie
coors = df.iloc[-1].cnmf.get_contours()

# let's plot that on top of the correlation image
corr_img = df.iloc[-1].caiman.get_corr_image().T  # must be transposed to line up

plt.imshow(corr_img, cmap='gray')

# plot the contours
for coor in coors:
    plt.scatter(coor[:, 0], coor[:, 1], s=4)
plt.show()

# see the demo notebook to see how to visualize residuals, reconstructed movie, etc.
```
