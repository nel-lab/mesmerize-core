# mesmerize-core

Mesmerize core backend

A high level abstraction that sits on top of the CaImAn library. 
It contains `pandas.DataFrame` and `pandas.Series` extensions that interface with CaImAn for running the various algorithms and organzing input & output data.

Required by `mesmerize-napari`. Can also be used standalone, such as in notebooks, as a high level interface for CaImAn.

# Installation

```bash
# create a new env, can be the same env that you use for `mesmerize-napari`
# you can also use conda instead of venv
python3 -m venv ~/python-venvs/mesmerize-napari

# activate env
source ~/python-venvs/mesmerize-napari/bin/activate

pip install --upgrade setuptools wheel pip

# clone this repo
cd ~/repos
git clone https://github.com/nel-lab/mesmerize-core.git
cd mesmerize-core

# install mesmerize-core
pip install -e .
```

# Examples

## Motion Correction using NoRMCorr

```python
from 

# set the parent directory as the top-level directory for your experiment data
set_parent_data_path('/home/kushal/my_exps_dir')

batch_path = '/home/kushal/my_exps_dir/my_batches/exp_1_batch.pickle'

# create a new batch
df = create_batch(batch_path)

# path to raw movie tiff file
movie_path = '/home/kushal/my_exps_dir/exp_1/my_movie.tif'

# params, exactly the same as what you'd directly use with CaImAn
mcorr_params1 = 
{
  'mcorr_kwargs': # this key is necessary for specifying that these are mcorr kwargs
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

df.caiman.add_item(
  algo='mcorr',
  name='my_movie',
  input_movie_path=movie_path,
  params=mcorr_params1
)

# We create another set of params, useful for gridsearches for example
mcorr_params2 =\ 
{
  'mcorr_kwargs': # this key is necessary for specifying that these are mcorr kwargs
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

df.caiman.add_item(
  algo='mcorr',
  name='my_movie',
  input_movie_path=movie_path,
  params=mcorr_params2
)

# run the first "batch item"
process = df.iloc[0].caiman.run(
  batch_path=batch_path,
  backend=COMPUTE_BACKEND_SUBPROCESS,  # this is for non-GUI use, COMPUTE_BACKEND_QPROCESS is for use within a Qt GUI
  callbacks_finished=[lambda: print("yay finished")], # callback function for when this item finishes
)

# run the second item
# you can also use a loop to run all these items
# just call process.wait() to run them one after another
process = df.iloc[1].caiman.run(
  batch_path=batch_path,
  backend=COMPUTE_BACKEND_SUBPROCESS,
  callbacks_finished=[lambda: print("yay finished")],
)

# now we can also perform CNMF and use the output of mcorr for that

# some params for CNMF
params_cnmf =\ 
{
    'cnmf_kwargs':
        {
            'p': 2,
            'nb': 1,
            # raises error: no parameter 'merge_thresh' found
            # 'merge_thresh': 0.7,
            'rf': None,
            'stride': 30,
            'K': 10,
            'gSig': [5,5],
            'ssub': 1,
            'tsub': 1,
            'method_init': 'greedy_roi',
        },
    'eval_kwargs':
        {
            'min_SNR': 2.50,
            'rval_thr': 0.8,
            'use_cnn': True,
            'min_cnn_thr': 0.8,
            'cnn_lowest': 0.1,
            'decay_time': 1,
        },
    'refit': True,
}

df.caiman.add_item(
  algo='cnmf',
  name='my_movie',
  input_movie_path=df.iloc[0].mcorr.get_output_path()  # use mcorr output from a previous item
  params=params_cnmf
)

# run this item
process = df.iloc[-1].caiman.run(
  batch_path=batch_path,
  backend=COMPUTE_BACKEND_SUBPROCESS,
  callbacks_finished=[lambda: print("yay finished")],
)

# we can look at the spatial components for example
coors = df.iloc[-1].cnmf.get_spatial_contour_coors()

# let's plot that on top of the correlation image
corr_img = df.iloc[-1].caiman.get_correlation_image().T  # must be transposed to line up

plt.imshow(corr_img, cmap='gray')

# plot the contours
for coor in coors:
  plt.scatter(coor[:, 0], coor[:, 1], s=4)
plt.show()
```
