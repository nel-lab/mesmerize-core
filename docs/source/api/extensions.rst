Caiman Extensions
*****************

Pandas extensions for CaImAn functionality with DataFrames.

Common
======

Extensions that are used for both MCorr (motion correction) and CNMF(E)

.. autoclass:: mesmerize_core.CaimanDataFrameExtensions
    :members: 

.. autoclass:: mesmerize_core.CaimanSeriesExtensions
    :members:
    
MCorr
=====

Extensions that are exlusive to motion correction

.. autoclass:: mesmerize_core.MCorrExtensions
    :members:


CNMF
====

.. autoclass:: mesmerize_core.CNMFExtensions
    :members:
    
Example
=======

.. code-block:: python

    from mesmerize_core import COMPUTE_BACKENDS, COMPUTE_BACKEND_QPROCESS, COMPUTE_BACKEND_SLURM, \
    COMPUTE_BACKEND_SUBPROCESS, set_parent_raw_data_path, get_parent_raw_data_path, get_full_raw_data_path, \
    load_batch, create_batch
    from mesmerize_core.caiman_extensions import *

    # set the parent directory as the top-level directory for your experiment data
    set_parent_raw_data_path('/home/kushal/my_exps_dir')

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
