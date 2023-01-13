.. mesmerize-core documentation master file, created by
   sphinx-quickstart on Wed Jun  1 17:14:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mesmerize-core's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   User Guide <user_guide>
   Visualization <visualization>
   API <api/index>

Summary
=======

``mesmerize-core`` interfaces `CaImAn <https://github.com/flatironinstitute/CaImAn>`_ algorithms, helps with parameter optimization, and data organization. It is a collection of "pandas extensions", which are functions that operate on pandas DataFrames. It essentially creates a user-friendly "psuedo-database" of your calcium imaging data and `CaImAn <https://github.com/flatironinstitute/CaImAn>`_ generated output files. This makes it easier to work with the results of the various `CaImAn <https://github.com/flatironinstitute/CaImAn>`_ algorithms to assess the outputs and create visualizations.

Installation
============

For installation please see the instructions on the README on GitHub:

https://github.com/nel-lab/mesmerize-core/blob/master/README.md#installation

Contributing
============

We're open to contributions! If you think there's a useful extension that can be added, or any other functionality, post an issue on the repo with your idea and then take a look at the `Contribution Guide <https://github.com/nel-lab/mesmerize-core/blob/master/CONTRIBUTING.md>`_.

You can also look at our "Milestones" for future versions to give you an idea of what we plan to implement in the future: https://github.com/nel-lab/mesmerize-core/milestones

Old Mesmerize Desktop application
---------------------------------

`mesmerize-core` in combination with `fastplotlib` basically replaces the Mesmerize desktop application which is now treated as legacy software. We strongly recommend `mesmerize-core` over the old desktop application, but if you really want to use the old desktop application you can access it here: http://mesmerize.readthedocs.io/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
