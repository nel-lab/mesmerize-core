Examples
********

I recommend using the demo notebook: https://github.com/nel-lab/mesmerize-core/tree/master/notebooks

Create/load a batch
==================

.. code-block:: python

    # import the load_batch() and create_batch() functions
    # as well as the caiman pandas extensions
    from mesmerize_core import *
    from matplotlib import pyplot as plt

    # set the parent directory as the top-level directory for your experiment data
    # this is mandatory
    set_parent_raw_data_path('/home/kushal/my_exps_dir')

    batch_path = '/home/kushal/my_exps_dir/my_batches/exp_1_batch.pickle'

    # create a new batch
    df = create_batch(batch_path)
    # look at the batch dataframe, it is empty with some columns
    df.head()

    # you can also load an existing batch
    df = load_batch('/home/kushal/my_other_exps/my_old_batches/existing_batch.pickle')
    # if it has items the dataframe should be populated
    df.head()


Motion Correction Items
=======================

Continue from above with a new batch or load an existing batch

.. code-block:: python

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

Run MCorr Items
===============

Continue from above

.. code-block:: python

    # run the first "batch item"
    process = df.iloc[0].caiman.run()
    process.wait()

    # you can also use a loop to run all these items in a batch
    # just call process.wait() to run them one after another
    from tqdm import tqdm  # so we have a progress bar
    for ix, r in tqdm(df.iterrows(), total=df.index.size):
        process = r.caiman.run()
        process.wait()

    # get the mot corrected video
    # fastplotlib can be used for fast random-access visualization in notebooks
    # see the demo notebooks for viz examples
    mcorr_movie = df.iloc[-1].mcorr.get_output()
    # plot the first frame just using matplotlib
    plt.imshow(mcorr_movie[0])

    # get the x-y shifts
    # you can plot these as a line plot
    shits = df.iloc[-1].mcorr.get_shifts()

CNMF Items
==========

We can continue from mcorr above and perform CNMF using the mcorr output

.. code-block:: python

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
    # you can also use a loop as shown in the mcorr example to run multiple CNMF items
    process = df.iloc[-1].caiman.run()
    process.wait()

    # we can look at the spatial components for example
    # see the demo notebook for an example that uses fastplotlib to visualize contours with the movie
    coors = df.iloc[-1].cnmf.get_contours()

    # let's plot that on top of the correlation image
    corr_img = df.iloc[-1].caiman.get_corr_image().T  # must be transposed to line up

    plt.imshow(corr_img, cmap='gray')

    # plot the contours
    for coor in coors:
        plt.scatter(coor[:, 0], coor[:, 1], s=4)
    plt.show()

    # see the demo notebook to see how to visualize residuals, reconstructed movie, etc.
