User Guide
**********
The demo notebook is the best place to start: https://github.com/nel-lab/mesmerize-core/tree/master/notebooks

This guide provides some more details on the API and concepts for using the ``mesmerize-core`` framework.

Mesmerize-core is a framework that interfaces with`CaImAn <https://github.com/flatironinstitute/CaImAn>`_ algorithms, helps with data organization, and provides useful functions for evaluation and visualization. It is a collection of "pandas extensions", which are functions that operate on `pandas DataFrames <https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe>`_. This enables you to create a "psuedo-database" of your calcium imaging data and `CaImAn <https://github.com/flatironinstitute/CaImAn>`_ generated output files.

Since this framework uses ``pandas`` extensions, you should be relatively comfortable with basic pandas operations. If you're familiar with ``numpy`` then ``pandas`` will be easy, here's a quick start guide from the pandas docs: https://pandas.pydata.org/docs/user_guide/10min.html

Accessors and Extensions
========================

There are 3 *accessors* that the ``mesmerize-core`` API provides, ``caiman``, ``mcorr`` and ``cnmf``. These allow you to perform operations on a ``pandas.DataFrame`` or invidual DataFrame rows, which are called ``pandas.Series``. In ``mesmerize-core`` the individual rows, ``pandas.Series``, contain data that pertains to a single **batch item**.

A **batch item** is the combination of:

* input data (input movie)
* parameters 
* algorithm
* output data (depends on algorithm)
* a user defined name for your convenience, multiple batch items can have the same name
* UUID (universally unique identifier), a 128 integer that is uniquely identifies this batch item and is used to organize the output data. **You must never modify the UUID, they are computer generated**.

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

You must use the appropriate *accessor* on a DataFrame or Series (row) to access the appropriate extension functions. Accessors that operate at the level of the DataFrame can only be referenced using the DataFrame instance.

For example the ``caiman.add_item()`` extension operates on a DataFrame, so you can use it like this:

.. code-block:: python

    # imports and load/create a dataframe
    from mesmerize_core import *
    
    # load an existing DataFrame
    df = load_batch("/path/to/batch.pickle")
    
    # in this case `df` is a DataFrame instance
    # we can use the `caiman` accessor to utilize 
    # common caiman extensions that operate at 
    # the level of the DataFrame
    
    # for example, ``add_item()`` works at the level of a DataFrame
    df.caiman.add_item(<args>)


In contrast some common extensions, such as ``cnmf.get_contours()`` operate on ``pandas.Series``, i.e. individual DataFrame *rows*. All the motion correction and CNMF extensions are also ``Series`` extensions. You will need to using indexing to get the ``pandas.Series`` (row) that you want.

.. code-block:: python

    # imports and load/create a dataframe
    from mesmerize_core import *
    
    # load an existing DataFrame
    df = load_batch("/path/to/batch.pickle")
    
    # df.iloc[n] will return the pandas.Series, i.e. row at the `nth` index
    
    # let's assume the item at index `0` is an mcorr item
    # we can get the memmaped output movie
    mcorr_movie_memmaped_array = df.iloc[0].mcorr.get_output()
    
    # let's assume the item at index `1` is a cnmf(e) item
    # we can get the contours
    contours, coms = df.iloc[1].cnmf.get_contours()


Use of the ``mcorr`` and ``cnmf`` accessors isn't limited to indexing through ``iloc[n]``, you can use any combination of pandas indexing that results in a ``pandas.Series``.
    
    
.. note:: Using the wrong accessor and extension on a batch item (row/pandas ``Series``) will raise an exception. For example,  you cannot use ``cnmf.get_contours()`` on a motion correction batch item.

Some ``common`` extensions are valid for getting outputs from motion correction and CNMF. For example the correlation Image can be obtained regardless of motion correction or CNMF using the common ``caiman`` accessor on a dataframe row.

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
=================

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

.. warning:: You MUST call ``wait()`` on the ``subprocess.Popen`` instance after the ``run()`` call, otherwise you will spawn hundres of processes for multiple batch items simultaneously!

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

Data management
---------------

See the :ref:`API reference <api_extenions_common>` for more details on these extensions.

There are some extensions under the common ``caiman`` accessor that help with data management, they operate on the ``DataFrame`` (not Series/rows).

**caimam.uloc()**

This will return the row, i.e. ``pandas.Series`` for the given UUID.

Example:

.. code-block:: python

    row = df.caiman.uloc("fd2b3734-96b1-4656-945e-6860df9b711e")
    
**caiman.remove_item()**

Removes the batch item i.e. row within the DataFrame (a.k.a ``pandas.Series``), from the DataFrame. Also delete corresponding output files from disk if ``remove_data=True`` (it is ``True`` by default). ``safe_removal`` (default ``True``) is useful to make sure you do not delete an mcorr item if this mcorr output is used later in the dataframe for cnmf.

The batch item to remove is indicated by an ``int`` index or ``UUID`` (either as a ``str`` or ``UUID`` object).

**get_children()**

Get the list of UUIDs of all batch items that use the output of the batch item passed to ``get_children()``. For example, you can get the UUIDs of all downstream CNMF batch items that use the output from a given mcorr batch item.

Note: This feature is experimental and its behavior may change in future releases.

**get_parent()**

Get the UUID of the parent batch item. For example, you can pass the UUID of a CNMF batch item to ``get_parent()`` to get the UUID of the mcorr batch item whose output was used as the input for the CNMF batch item.

Note: This feature is experimental and its behavior may change in future releases.
        
Motion Correction Extensions
============================

:ref:`API reference for motion correction extensions <api_extensions_mcorr>`

These extensions with the accessor ``mcorr`` contain functions that are exclusive to motion correction.

**mcorr.get_output()**

This returns the memmaped numpy array of the motion corrected movie. It allows fast random access scrolling which is useful for fast random-access scrolling during visualization. See the :ref:`Visuzalition <visualization>` page for details on visualization, we recommend ``mesmerize-viz`` and ``fastplotlib``.

**mcorr.get_output_path()**

This returns the ``Path`` to the memmaped numpy array. The most common use for this extension is for using the motion corrected movie as the input movie for CNMF(E). You can use the returned path from ``mcorr.get_output_path()`` to set the ``input_movie_path`` argument for ``caiman.add_item()``

CNMF Extensions
===============

These extensions with the accessor ``cnmf`` contain functions that are exclusive to CNMF, such as getting the contours and centers of mass for spatial components, getting the temporal components and dF/F0, running component evaluation, getting the reconstructed movie, residuals, etc. See the :ref:`API reference for CNMF extensions <api_extensions_cnmf>` which extensively documents these extensions along with several examples.
