from pathlib import Path
from typing import *
import numpy as np
import pandas as pd
from caiman import load_memmap
from caiman.source_extraction.cnmf import CNMF
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.visualization import get_contours as caiman_get_contours
from functools import wraps
import os
from copy import deepcopy

from ._utils import validate
from .cache import Cache
from ..arrays import *
from ..arrays._base import LazyArray


cnmf_cache = Cache()


# this decorator MUST be called BEFORE caching decorators!
def _component_indices_parser(func):
    @wraps(func)
    def _parser(instance, *args, **kwargs) -> Any:
        if "component_indices" in kwargs.keys():
            component_indices: Union[np.ndarray, str, None] = kwargs["component_indices"]
        elif len(args) > 0:
            component_indices = args[0]  # always first positional arg in the extensions
        else:
            component_indices = None  # default

        cnmf_obj = instance.get_output()

        # TODO: finally time to learn Python's new switch case
        accepted = (np.ndarray, str, type(None))
        if not isinstance(component_indices, accepted):
            raise TypeError(f"`component_indices` must be one of type: {accepted}")

        if isinstance(component_indices, np.ndarray):
            pass

        elif component_indices is None:
            component_indices = np.arange(cnmf_obj.estimates.A.shape[1])

        if isinstance(component_indices, str):
            accepted = ["all", "good", "bad"]
            if component_indices not in accepted:
                raise ValueError(f"Accepted `str` values for `component_indices` are: {accepted}")

            if component_indices == "all":
                component_indices = np.arange(cnmf_obj.estimates.A.shape[1])

            elif component_indices == "good":
                component_indices = cnmf_obj.estimates.idx_components

            elif component_indices == "bad":
                component_indices = cnmf_obj.estimates.idx_components_bad
        if "component_indices" in kwargs.keys():
            kwargs["component_indices"] = component_indices
        else:
            args = (component_indices, *args[1:])

        return func(instance, *args, **kwargs)
    return _parser


def _check_permissions(func):
    @wraps(func)
    def __check(instance, *args, **kwargs):
        cnmf_obj_path = instance.get_output_path()

        if not os.access(cnmf_obj_path, os.W_OK):
            raise PermissionError(
                "You do not have write access to the hdf5 output file for this batch item"
            )

        return func(instance, *args, **kwargs)
    return __check


@pd.api.extensions.register_series_accessor("cnmf")
class CNMFExtensions:
    """
    Extensions for managing CNMF output data
    """

    def __init__(self, s: pd.Series):
        self._series = s

    @validate("cnmf")
    def get_cnmf_memmap(self, mode: str = "r") -> np.ndarray:
        """
        Get the CNMF C-order memmap. This should NOT be used for viewing the
        movie frames use ``caiman.get_input_movie()`` for that purpose.

        Parameters
        ----------

        mode: str
            passed to numpy.memmap

            one of: `{'r+', 'r', 'w+', 'c'}`

            The file is opened in this mode:

            +------+-------------------------------------------------------------+
            | 'r'  | Open existing file for reading only.                        |
            +------+-------------------------------------------------------------+
            | 'r+' | Open existing file for reading and writing.                 |
            +------+-------------------------------------------------------------+
            | 'w+' | Create or overwrite existing file for reading and writing.  |
            +------+-------------------------------------------------------------+
            | 'c'  | Copy-on-write: assignments affect data in memory, but       |
            |      | changes are not saved to disk.  The file on disk is         |
            |      | read-only.                                                  |
            +------+-------------------------------------------------------------+

        Returns
        -------
        np.ndarray
            numpy memmap array used for CNMF
        """

        path = self._series.paths.resolve(self._series["outputs"]["cnmf-memmap-path"])
        # Get order f images
        Yr, dims, T = load_memmap(str(path), mode=mode)
        images = np.reshape(Yr.T, [T] + list(dims), order="F")
        return images

    @validate("cnmf")
    def get_output_path(self) -> Path:
        """
        Get the path to the cnmf hdf5 output file.

        **Note:** You generally want to work with the other extensions instead of directly using the hdf5 file.

        Returns
        -------
        Path
            full path to the caiman-format CNMF hdf5 output file

        """

        return self._series.paths.resolve(self._series["outputs"]["cnmf-hdf5-path"])

    @validate("cnmf")
    @cnmf_cache.use_cache
    def get_output(self, return_copy=True) -> CNMF:
        """
        Parameters
        ----------
        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory, not recommend this could result in strange unexpected behavior.
            | In general you want a copy of the cached value.

        Returns
        -------
        CNMF
            Returns the Caiman CNMF object

        Examples
        --------

        Load the CNMF model with estimates from the hdf5 file.

        .. code-block:: python

            from mesmerize_core import load_batch

            df = load_batch("/path/to/batch_dataframe_file.pickle")

            # assume the 0th index is a cnmf item
            cnmf_obj = df.iloc[0].cnmf.get_output()

            # see some estimates
            print(cnmf_obj.estimates.C)
            print(cnmf_obj.estimates.f)

        """

        # Need to create a cache object that takes the item's UUID and returns based on that
        # collective global cache
        return load_CNMF(self.get_output_path())

    @validate("cnmf")
    @_component_indices_parser
    @cnmf_cache.use_cache
    def get_masks(
        self, component_indices: Union[np.ndarray, str] = None, threshold: float = 0.01, return_copy=True
    ) -> np.ndarray:
        """
        | Get binary masks of the spatial components at the given ``component_indices``.
        | Created from ``CNMF.estimates.A``

        Parameters
        ----------
        component_indices: str or np.ndarray, optional
            | indices of the components to include
            | if not provided, ``None``, or ``"all"`` uses all components
            | if ``"good"`` uses good components, i.e. ``Estimates.idx_components``
            | if ``"bad"`` uses bad components, i.e. ``Estimates.idx_components_bad``
            | if ``np.ndarray``, uses the indices in the provided array

        threshold: float
            threshold

        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory, not recommend this could result in strange unexpected behavior.
            | In general you want a copy of the cached value.

        Returns
        -------
        np.ndarray
            shape is [dim_0, dim_1, n_components]

        """

        cnmf_obj = self.get_output()

        dims = cnmf_obj.dims
        if dims is None:
            dims = cnmf_obj.estimates.dims

        masks = np.zeros(shape=(dims[0], dims[1], len(component_indices)), dtype=bool)

        for n, ix in enumerate(component_indices):
            s = cnmf_obj.estimates.A[:, ix].toarray().reshape(cnmf_obj.dims)
            s[s >= threshold] = 1
            s[s < threshold] = 0

            masks[:, :, n] = s.astype(bool)

        return masks

    @staticmethod
    def _get_spatial_contours(
        cnmf_obj: CNMF, component_indices, swap_dim
    ):

        dims = cnmf_obj.dims
        if dims is None:
            # I think that one of these is `None` if loaded from an hdf5 file
            dims = cnmf_obj.estimates.dims

        # need to transpose these
        if swap_dim:
            dims = dims[1], dims[0]
        else:
            dims = dims[0], dims[1]

        contours = caiman_get_contours(
            cnmf_obj.estimates.A[:, component_indices], dims, swap_dim=swap_dim
        )

        return contours

    @validate("cnmf")
    @_component_indices_parser
    @cnmf_cache.use_cache
    def get_contours(
            self,
            component_indices: Union[np.ndarray, str] = None,
            swap_dim: bool = True,
            return_copy=True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get the contour and center of mass for each spatial footprint
        Note, the centers of mass are different from those computed by CaImAn.
        They are based on the contours and can be used to compute click targets for visualizations.

        Parameters
        ----------
        component_indices: str or np.ndarray, optional
            | indices of the components to include
            | if not provided, ``None``, or ``"all"`` uses all components
            | if ``"good"`` uses good components, i.e. ``Estimates.idx_components``
            | if ``"bad"`` uses bad components, i.e. ``Estimates.idx_components_bad``
            | if ``np.ndarray``, uses the indices in the provided array

        swap_dim: bool
            swap the x and y coordinates, use if the contours don't align with the cells in your image

        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory, not recommend this could result in strange unexpected behavior.
            | In general you want a copy of the cached value.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            | (List[coordinates array], List[centers of masses array])
            | each array of coordinates is 2D, [xs, ys]
            | each center of mass is [x, y]

        """

        cnmf_obj = self.get_output()
        contours = self._get_spatial_contours(cnmf_obj, component_indices, swap_dim)

        coordinates = list()
        coms = list()

        for contour in contours:
            coors = contour["coordinates"]
            coordinates.append(coors)

            com = np.nanmean(coors, axis=0)
            coms.append(com)

        return coordinates, coms

    @validate("cnmf")
    @_component_indices_parser
    @cnmf_cache.use_cache
    def get_temporal(
        self,
        component_indices: Union[np.ndarray, str] = None,
        add_background: bool = False,
        add_residuals: bool = False,
        return_copy=True
    ) -> np.ndarray:
        """
        Get the temporal components for this CNMF item, basically ``CNMF.estimates.C``

        Parameters
        ----------
        component_indices: str or np.ndarray, optional
            | indices of the components to include
            | if not provided, ``None``, or ``"all"`` uses all components
            | if ``"good"`` uses good components, i.e. ``Estimates.idx_components``
            | if ``"bad"`` uses bad components, i.e. ``Estimates.idx_components_bad``
            | if ``np.ndarray``, uses the indices in the provided array

        add_background: bool, default False
            if ``True``, add the temporal background, adds ``cnmf.estimates.f``

        add_residuals: bool, default False
            if ``True``, add residuals, i.e. ``cnmf.estimates.YrA``

        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory, not recommend this could result in strange unexpected behavior.
            | In general you want a copy of the cached value.

        Returns
        -------
        np.ndarray
            shape is [n_components, n_frames]

        Examples
        --------

        Plot the temporal components as a heatmap

        .. code-block:: python

            from mesmerize_core import load_batch
            from fastplotlib import Plot

            df = load_batch("/path/to/batch_dataframe_file.pickle")

            # assumes 0th index is a cnmf batch item
            temporal = df.iloc[0].cnmf.get_temporal()

            plot = Plot()

            plot.add_line_collection(temporal)

            plot.show()
        """

        cnmf_obj = self.get_output()

        C = cnmf_obj.estimates.C[component_indices]
        f = cnmf_obj.estimates.f

        temporal = C

        if add_background:
            temporal += f
        elif add_residuals:
            temporal += cnmf_obj.estimates.YrA[component_indices]

        return temporal

    @validate("cnmf")
    @_component_indices_parser
    @cnmf_cache.use_cache
    def get_rcm(
            self,
            component_indices: Union[np.ndarray, str] = None,
            temporal_components: np.ndarray = None,
            return_copy=False
    ) -> LazyArrayRCM:
        """
        Return the reconstructed movie with no background, i.e. ``A ⊗ C``, as a ``LazyArray``.
        This is an array that performs lazy computation of the reconstructed movie only upon indexing.

        Parameters
        ----------
        component_indices: optional, Union[np.ndarray, str]
            | indices of the components to include
            | if ``np.ndarray``, uses these indices in the provided array
            | if ``"good"`` uses good components, i.e. cnmf.estimates.idx_components
            | if ``"bad"`` uses bad components, i.e. cnmf.estimates.idx_components_bad
            | if not provided, ``None``, or ``"all"`` uses all components

        temporal_components: optional, np.ndarray
            temporal components to use as ``C`` for computing reconstructed movie.

            | uses ``cnmf.estimates.C`` if not provided
            | useful if you want to create the reconstructed movie using dF/Fo, z-scored data, etc.

        return_copy: bool, default ``False``
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory
            | ``False`` is used by default when returning ``LazyArrays`` for technical reasons

        Returns
        -------
        LazyArrayRCM
            shape is [n_frames, x_dims, y_dims]

        Examples
        --------

        This example uses fastplotlib to display the reconstructed movie from a CNMF item that has already been run.

        | **fastplotlib code must be run in a notebook**

        | See the demo notebooks for more detailed examples.

        .. code-block:: python

            from mesmerize_core import *
            from fastplotlib.widgets import ImageWidget

            # load existing batch
            df = load_batch("/path/to/batch.pickle")

            # get the reconstructed movie as LazyArray
            # assumes the last index, `-1`, is a cnmf item
            # uses only the "good" components
            rcm = df.iloc[-1].cnmf.get_rcm(component_indices="good")

            # view with ImageWidget
            iw = ImageWidget(data=rcm)
            iw.show()
        """

        cnmf_obj = self.get_output()

        if temporal_components is None:
            temporal_components = cnmf_obj.estimates.C

        else:  # number of spatial components must equal number of temporal components
            if cnmf_obj.estimates.A.shape[1] != temporal_components.shape[0]:
                raise ValueError(
                    f"Number of temporal components provided: `{temporal_components.shape[0]}` "
                    f"does not equal number of spatial components provided: `{cnmf_obj.estimates.A.shape[1]}`"
                )

        if cnmf_obj.estimates.dims is not None:
            dims = cnmf_obj.estimates.dims
        elif cnmf_obj.dims is not None:
            dims = cnmf_obj.dims
        else:
            raise AttributeError(f"`dims` not found in the CNMF data, it is usually found in one of the following:\n"
                                 f"`cnmf_obj.estimates.dims` or `cnmf_obj.dims`")

        spatial = cnmf_obj.estimates.A[:, component_indices]
        temporal = temporal_components[component_indices, :]

        return LazyArrayRCM(spatial=spatial, temporal=temporal, frame_dims=dims)

    @validate("cnmf")
    @cnmf_cache.use_cache
    def get_rcb(self,) -> LazyArrayRCB:
        """
        Return the reconstructed background, ``(b ⊗ f)``

        Returns
        -------
        LazyArrayRCB
            shape is [n_frames, x_dims, y_dims]

        Examples
        --------

        This example uses fastplotlib to display the reconstructed movie from a CNMF item that has already been run.

        | **fastplotlib code must be run in a notebook**

        | See the demo notebooks for more detailed examples.

        .. code-block:: python

            from mesmerize_core import *
            from fastplotlib.widgets import ImageWidget

            # load existing batch
            df = load_batch("/path/to/batch.pickle")

            # get the reconstructed background as a LazyArray
            # assumes the last index, `-1`, is a cnmf item
            rcb = df.iloc[-1].cnmf.get_rcb()

            # view with ImageWidget
            iw = ImageWidget(data=rcb)
            iw.show()
        """

        cnmf_obj = self.get_output()

        if cnmf_obj.estimates.dims is not None:
            dims = cnmf_obj.estimates.dims
        elif cnmf_obj.dims is not None:
            dims = cnmf_obj.dims
        else:
            raise AttributeError(f"`dims` not found in the CNMF data, it is usually found in one of the following:\n"
                                 f"`cnmf_obj.estimates.dims` or `cnmf_obj.dims`")

        spatial = cnmf_obj.estimates.b
        temporal = cnmf_obj.estimates.f

        return LazyArrayRCB(spatial=spatial, temporal=temporal, frame_dims=dims)

    @validate("cnmf")
    @cnmf_cache.use_cache
    def get_residuals(self) -> LazyArrayResiduals:
        """
        Return residuals, ``Y - (A ⊗ C) - (b ⊗ f)``

        Returns
        -------
        LazyArrayResiduals
            shape is [n_frames, x_dims, y_dims]

        Examples
        --------

        This example uses fastplotlib to display the reconstructed movie from a CNMF item that has already been run.

        | **fastplotlib code must be run in a notebook**

        | See the demo notebooks for more detailed examples.

        .. code-block:: python

            from mesmerize_core import *
            from fastplotlib.widgets import ImageWidget

            # load existing batch
            df = load_batch("/path/to/batch.pickle")

            # get the reconstructed background as a LazyArray
            # assumes the last index, `-1`, is a cnmf item
            residuals = df.iloc[-1].cnmf.get_residuals()

            # view with ImageWidget
            iw = ImageWidget(data=residuals)
            iw.show()
        """

        residuals = LazyArrayResiduals(
            self._series.caiman.get_input_movie(),
            self.get_rcm(),
            self.get_rcb(),
        )

        return residuals

    @validate("cnmf")
    @_check_permissions
    @cnmf_cache.invalidate()
    def run_detrend_dfof(
            self,
            quantileMin: float = 8,
            frames_window: int = 500,
            flag_auto: bool = True,
            use_fast: bool = False,
            use_residuals: bool = True,
            detrend_only: bool = False
    ) -> None:
        """
        | Uses caiman's detrend_df_f.
        | call ``CNMF.get_detrend_dfof()`` to get the values.
        | Sets ``CNMF.estimates.F_dff``

        Warnings
        --------
        Overwrites the existing cnmf hdf5 output file for this batch item

        Parameters
        ----------
        quantileMin: float
            quantile used to estimate the baseline (values in [0,100])
            used only if 'flag_auto' is False, i.e. ignored by default

        frames_window: int
            number of frames for computing running quantile

        flag_auto: bool
            flag for determining quantile automatically

        use_fast: bool
            flag for using approximate fast percentile filtering

        detrend_only: bool
            flag for only subtracting baseline and not normalizing by it.
            Used in 1p data processing where baseline fluorescence cannot be
            determined.

        Returns
        -------
        None

        Notes
        ------
        invalidates the cache for this batch item.

        """

        cnmf_obj: CNMF = self.get_output()
        cnmf_obj.estimates.detrend_df_f(
            quantileMin=quantileMin,
            frames_window=frames_window,
            flag_auto=flag_auto,
            use_fast=use_fast,
            use_residuals=use_residuals,
            detrend_only=detrend_only
        )

        # remove current hdf5 file
        cnmf_obj_path = self.get_output_path()
        cnmf_obj_path.unlink()

        # save new hdf5 file with new F_dff vals
        cnmf_obj.save(str(cnmf_obj_path))

    @validate("cnmf")
    @_component_indices_parser
    @cnmf_cache.use_cache
    def get_detrend_dfof(
            self,
            component_indices: Union[np.ndarray, str] = None,
            return_copy: bool = True
    ):
        """
        Get the detrended dF/F0 curves after calling ``run_detrend_dfof``.
        Basically ``CNMF.estimates.F_dff``.

        Parameters
        ----------
        component_indices: str or np.ndarray, optional
            | indices of the components to include
            | if not provided, ``None``, or ``"all"`` uses all components
            | if ``"good"`` uses good components, i.e. ``Estimates.idx_components``
            | if ``"bad"`` uses bad components, i.e. ``Estimates.idx_components_bad``
            | if ``np.ndarray``, uses the indices in the provided array

        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory, not recommend this could result in strange unexpected behavior.
            | In general you want a copy of the cached value.

        Returns
        -------
        np.ndarray
            shape is [n_components, n_frames]

        """

        cnmf_obj = self.get_output()
        if cnmf_obj.estimates.F_dff is None:
            raise AttributeError("You must run ``cnmf.run_detrend_dfof()`` first")

        return cnmf_obj.estimates.F_dff[component_indices]

    @validate("cnmf")
    @_check_permissions
    @cnmf_cache.invalidate()
    def run_eval(self, params: dict) -> None:
        """
        Run component evaluation. This basically changes the indices for good and bad components.

        Warnings
        --------
        Overwrites the existing cnmf hdf5 output file for this batch item

        Parameters
        ----------
        params: dict
            dict of parameters for component evaluation

            ==============  =================
            parameter       details
            ==============  =================
            SNR_lowest      ``float``, minimum accepted SNR value
            cnn_lowest      ``float``, minimum accepted value for CNN classifier
            gSig_range      ``List[int, int]`` or ``None``, range for gSig scale for CNN classifier
            min_SNR         ``float``, transient SNR threshold
            min_cnn_thr     ``float``, threshold for CNN classifier
            rval_lowest     ``float``, minimum accepted space correlation
            rval_thr        ``float``, space correlation threshold
            use_cnn         ``bool``, use CNN based classifier
            use_ecc         ``bool``, flag for eccentricity based filtering
            max_ecc         ``float``, max eccentricity
            ==============  =================

        Returns
        -------
        None

        Notes
        ------
        invalidates the cache for this batch item.

        """

        cnmf_obj = self.get_output()

        valid = list(cnmf_obj.params.quality.keys())
        for k in params.keys():
            if k not in valid:
                raise KeyError(
                    f"passed params dict key `{k}` is not a valid parameter for quality evaluation\n"
                    f"valid param keys are: {valid}"
                )

        cnmf_obj.params.quality.update(params)
        cnmf_obj.estimates.filter_components(
            imgs=self._series.caiman.get_input_movie(),
            params=cnmf_obj.params
        )

        cnmf_obj_path = self.get_output_path()
        cnmf_obj_path.unlink()

        cnmf_obj.save(str(cnmf_obj_path))
        self._series["params"]["eval"] = deepcopy(params)

    @validate("cnmf")
    def get_good_components(self) -> np.ndarray:
        """
        get the good component indices, ``Estimates.idx_components``

        Returns
        -------
        np.ndarray
            array of ints, indices of good components

        """

        cnmf_obj = self.get_output()
        return cnmf_obj.estimates.idx_components

    @validate("cnmf")
    def get_bad_components(self) -> np.ndarray:
        """
        get the bad component indices, ``Estimates.idx_components_bad``

        Returns
        -------
        np.ndarray
            array of ints, indices of bad components

        """

        cnmf_obj = self.get_output()
        return cnmf_obj.estimates.idx_components_bad
