User Guide
**********
The demo notebook is the best place to start: https://github.com/nel-lab/mesmerize-core/tree/master/notebooks

This guide provides some more details on the API and concepts for using the ``mesmerize-core`` framework.

Mesmerize-core is a framework that makes it easy to use `CaImAn <https://github.com/flatironinstitute/CaImAn>`_ algorithms and organize the outputs. It is a collection of "pandas extensions", which are functions that operate on pandas DataFrames. This enables to to essentially create a "psuedo-database" of your calcium imaging data and `CaImAn <https://github.com/flatironinstitute/CaImAn>`_ generated output files. The "pandas extensions" are useful functions that make it easier to work with the results of the various `CaImAn <https://github.com/flatironinstitute/CaImAn>`_ algorithms.

Since this framework uses ``pandas`` extensions, you should be relatively comfortable with basic pandas operations. If you're familiar with ``numpy`` then ``pandas`` will be easy, here's a quick start guide from the pandas docs: https://pandas.pydata.org/docs/user_guide/10min.html

Accessors and Extensions
========================

There are 3 *accessors* that the ``mesmerize-core`` API provides, ``caiman``, ``mcorr`` and ``cnmf``. These allow you to perform operations on a *"pandas DataFrame"* or DataFrame rows, which are called *"pandas Series"*. You must use the *accessor* on a DataFrame or Series (row) to access the appropriate extension functions.

**Examples:**

Some common ``caiman`` extensions are:

* ``caiman.add_item()`` - adds a new batch item to the dataframe
* ``caiman.run()`` - runs the batch item
* ``caiman.get_corr_image()`` - gets the correlation image for the batch item

Some motion correction extensions:

* ``mcorr.get_output()`` - get the motion corrected memmaped numpy array
* ``mcorr.get_shifts()`` - get the x, y shifts per frame

Some CNMF extensions:

* ``cnmf.get_contours()`` - get the spatial contours and centers of mass
* ``cnmf.get_rcm()`` - get the reconstructed movie, i.e. ``(A * C)``
* ``cnmf.get_residuals()`` - get the residuals, i.e. ``Y - (A * C) - (b * f)``
* ``cnmf.run_eval()`` - runs component quality evaluation

.. note:: Using the wrong accessor and extension on a batch item (row/pandas ``Series``) will raise an exception. For example,  you cannot use ``cnmf.get_contours()`` on a motion correction batch item.

**Usage Examples:**

Correlation Image, since this can be obtained regardless of motion correction or CNMF we can use the common ``caiman`` accessor on a dataframe row.

.. code-block:: python

    from mesmerize_core import *

    # the 0th index, i.e. first row, in the dataframe
    corr_img = df.iloc[0].caiman.get_corr_image()

    # plot with matplotlib
    from matplotlib import pyplot as plt

    plt.imshow(corr_img)


**More examples**

We can get motion corrected outputs as a memmaped numpy array using the ``mcorr`` accessor and ``get_output()`` function. We can also get CNMF outputs from another batch item, such as temporal components, using the ``cnmf`` accessor and ``get_temporal()`` function.

.. code-block:: python

    # get the output memmap after motion correction
    df.iloc[0].mcorr.get_output()

    memmap([[[ 1.09921265e+01,  5.52584839e+00,  2.44244690e+01, ...,
           2.74850464e+00,  5.92257690e+00,  3.67776489e+00],
         [ 8.48319397e+01,  4.00158539e+01,  6.09210205e+00, ...,
           3.89350281e+01,  5.72113037e+01,  2.35960083e+01],
         [ 1.09254852e+02,  8.75248413e+01,  1.91671143e+01, ...,
           2.50050354e+01,  7.38364258e+01,  1.21587524e+01],
         ...,

    df.iloc[0].mcorr.get_output().shape # returns [n_frames, x_pix, y_pix]

    (3000, 170, 170)

    # get temporal and spacial components
    df.iloc[1].cnmf.get_temporal()

    # this will return the [n_neurons, n_frames] array
    array([[-22.34959017, -22.34959017, -22.34959017, ..., -22.34959017,
        -22.34959017, -22.34959017],
       [-24.06055624, -24.06055624,   0.73800929, ..., -24.03839339,
        -24.04034401, -24.04212251],
       [-20.06077687, -20.06077687, -20.06077687, ..., -20.06077687,
        -20.06077687, -20.06077687],
       ...,

Common Extensions
-----------------

:ref:`API reference for common extensions <api_extenions_common>`

These extensions with the accessor ``caiman`` contain functions that are common to both motion correction and CNMF. The most frequent ``common`` extension you will probably use is ``add_item()`` which adds a new batch item (row) to the ``DataFrame``.

Basic structure of using ``add_item()``:

.. code-block::

    df.caiman.add_item(
        algo=<name of algorithm, mcorr, cnmf, or cnmfe>,
        item_name=<a name for you to keep track of this item>,
        input_movie_path="/path/to/input_movie.tif",
        params=<params dict for algo>,
    )

Example:

.. code-block:: python

    from mesmerize_core import *
    # create a new batch
    df = create_batch("/path/to/batch.pickle")

    # params, exactly the same as what you'd directly use with CaImAn
    mcorr_params =\
    {
    'main': # this key is required to specify that these are the "main" params for the algorithm
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
        algo="mcorr",
        item_name="movie_from_fav_brain",
        input_movie_path="/path/to/fav_movie.tif",
        params=mcorr_params
    )

You can add multiple "batch items" using the same **input movie** and set the same **item_name** but use different **params**. This enables you to perform a gridsearch to find the optimal **params** for your **input movie**.

You can run a batch item using the ``run()`` extension on an individual ``DataFrame`` row, technically called a pandas ``Series``. At the moment the only supported backend is ``subprocess``, the "batch item" is run using the corresponding algorithm in an external subprocess so you can continue using your notebook, i.e. calling ``run()`` is non-blocking. ``run()`` returns a ``subprocess.Popen`` instance.

Example:

.. code-block:: python

    # assuming a batch dataframe is already loaded
    # runs the item at the 0th index
    df.iloc[0].caiman.run()

You can run an entire DataFrame from the 0th index (i.e. first row) to the last index (-1), or run certain ranges just by using for loops. I would recommend a pandas tutorial if this sounds complicated (pandas concepts and syntax are similar to numpy).

.. warning:: You MUST call ``wait()`` on the ``subprocess.POpen`` instance after the ``run()`` call, otherwise you will spawn hundres of processes for multiple batch items simultaneously!

.. code-block:: python

    from tqdm import tqdm # for a progress bar

    # run an entire dataframe
    for ix, r in tqdm(df.iterrows(), total=df.index.size):
        process = r.caiman.run()
        process.wait()  # this line is VERY IMPORTANT!!

    # or run only certain rows
    for ix, r in tqdm(df.iterrows(), total=df.index.size):
        if ix < 30:  # skip the first 29 items
            continue
        if ix > 100:  # skip items after index 99
            continue

        process = r.caiman.run()
        process.wait()

Motion Correction Extensions
----------------------------

:ref:`API reference for motion correction extensions <api_extenions_mcorr>`

These extensions with the accessor ``mcorr`` contain functions that are exclusive to motion correction.


CNMF Extensions
---------------

:ref:`API reference for CNMF extensions <api_extenions_cnmf>`

These extensions with the accessor ``cnmf`` contain functions that are exclusive to CNMF.
