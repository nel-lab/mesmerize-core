.. _api_extensions_cnmf:

CNMF
****

**Accessor:** ``cnmf``

.. autoclass:: mesmerize_core.CNMFExtensions
    :members:

Lazy Arrays
===========

These are returned by the respective cnmf extensions (see above), ``get_rcm()``, ``get_rcb()``, and ``get_residuals()``.
They make it possible to view large arrays which would otherwise be larger than RAM.

.. autoclass:: mesmerize_core.arrays.LazyArrayRCM
    :members:

.. autoclass:: mesmerize_core.arrays.LazyArrayRCB
    :members:

.. autoclass:: mesmerize_core.arrays.LazyArrayResiduals
    :members:
