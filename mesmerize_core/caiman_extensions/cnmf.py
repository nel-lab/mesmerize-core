from pathlib import Path
from typing import *
import numpy as np
import pandas as pd
from caiman import load_memmap
from caiman.source_extraction.cnmf import CNMF
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.visualization import get_contours as caiman_get_contours

from .common import validate
from .cache import Cache

cache = Cache()


@pd.api.extensions.register_series_accessor("cnmf")
class CNMFExtensions:
    """
    Extensions for managing CNMF output data
    """

    def __init__(self, s: pd.Series):
        self._series = s

    @validate("cnmf")
    def get_cnmf_memmap(self) -> np.ndarray:
        """
        Get the CNMF C-order memmap

        Returns
        -------
        np.ndarray
            numpy memmap array used for CNMF
        """
        path = self._series.paths.resolve(self._series["outputs"]["cnmf-memmap-path"])
        # Get order f images
        Yr, dims, T = load_memmap(str(path))
        images = np.reshape(Yr.T, [T] + list(dims), order="F")
        return images

    def get_input_memmap(self) -> np.ndarray:
        """
        Return the F-order memmap if the input to this
        CNMF batch item was a mcorr output memmap

        Returns
        -------
        np.ndarray
            numpy memmap array of the input
        """
        movie_path = str(self._series.caiman.get_input_movie_path())
        if movie_path.endswith("mmap"):
            Yr, dims, T = load_memmap(movie_path)
            images = np.reshape(Yr.T, [T] + list(dims), order="F")
            return images
        else:
            raise TypeError(
                f"Input movie for CNMF was not a memmap, path to input movie is:\n"
                f"{movie_path}"
            )

    @validate("cnmf")
    def get_output_path(self) -> Path:
        """
        Returns
        -------
        Path
            full path to the caiman-format CNMF hdf5 output file
        """
        return self._series.paths.resolve(self._series["outputs"]["cnmf-hdf5-path"])

    @validate("cnmf")
    @cache.use_cache
    def get_output(self, return_copy=True) -> CNMF:
        """
        Parameters
        ----------
        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory.
            | In general you want a copy of the cached value.

        Returns
        -------
        CNMF
            Returns the Caiman CNMF object
        """
        # Need to create a cache object that takes the item's UUID and returns based on that
        # collective global cache
        return load_CNMF(self.get_output_path())

    @validate("cnmf")
    @cache.use_cache
    def get_masks(
        self, component_indices: Optional[np.ndarray] = None, threshold: float = 0.01, return_copy=True
    ) -> np.ndarray:
        """
        | Get binary masks of the spatial components at the given ``component_indices``.
        | Created from cnmf.estimates.A.

        Parameters
        ----------
        component_indices: np.ndarray
            numpy array containing integer indices for which you want spatial masks.
            if ``None`` uses "good" components, i.e. ``cnmf.estimates.idx_components``

        threshold: float
            threshold

        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory.
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

        if component_indices is None:
            component_indices = cnmf_obj.estimates.idx_components

        masks = np.zeros(shape=(dims[0], dims[1], len(component_indices)), dtype=bool)

        for n, ix in enumerate(component_indices):
            s = cnmf_obj.estimates.A[:, ix].toarray().reshape(cnmf_obj.dims)
            s[s >= threshold] = 1
            s[s < threshold] = 0

            masks[:, :, n] = s.astype(bool)

        return masks

    @staticmethod
    def _get_spatial_contours(
        cnmf_obj: CNMF, component_indices: Optional[np.ndarray] = None
    ):
        if component_indices is None:
            component_indices = cnmf_obj.estimates.idx_components

        dims = cnmf_obj.dims
        if dims is None:
            # I think that one of these is `None` if loaded from an hdf5 file
            dims = cnmf_obj.estimates.dims

        # need to transpose these
        dims = dims[1], dims[0]

        contours = caiman_get_contours(
            cnmf_obj.estimates.A[:, component_indices], dims, swap_dim=True
        )

        return contours

    @validate("cnmf")
    @cache.use_cache
    def get_contours(
        self, component_indices: Optional[np.ndarray] = None, return_copy=True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get the contour and center of mass for each spatial footprint

        Parameters
        ----------
        component_indices: np.ndarray
            indices for which to return spatial contours.
            if `None` uses cnmf.estimates.idx_components

        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory.
            | In general you want a copy of the cached value.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            | (List[coordinates array], List[centers of masses array])
            | each array of coordinates is 2D, [xs, ys]
            | each center of mass is [x, y]
        """
        cnmf_obj = self.get_output()
        contours = self._get_spatial_contours(cnmf_obj, component_indices)

        coordinates = list()
        coms = list()

        for contour in contours:
            coors = contour["coordinates"]
            coors = coors[~np.isnan(coors).any(axis=1)]
            coordinates.append(coors)

            com = coors.mean(axis=0)
            coms.append(com)

        return coordinates, coms

    @validate("cnmf")
    @cache.use_cache
    def get_temporal(
        self, component_indices: Optional[np.ndarray] = None, add_background: bool = False, return_copy=True
    ) -> np.ndarray:
        """
        Get the temporal components for this CNMF item, basically ``cnmf.estimates.C``

        Parameters
        ----------
        component_indices: np.ndarray
            | indices for which to return temporal components, ``cnmf.estimates.C``.
            | if ``None`` uses ``cnmf.estimates.idx_components``

        add_background: bool
            if ``True``, add the temporal background, ``cnmf.estimates.C + cnmf.estimates.f``

        return_copy: bool
            | if ``True`` returns a copy of the cached value in memory.
            | if ``False`` returns the same object as the cached value in memory.
            | In general you want a copy of the cached value.

        Returns
        -------
        np.ndarray
            shape is [n_components, n_frames]
        """
        cnmf_obj = self.get_output()

        if component_indices is None:
            component_indices = cnmf_obj.estimates.idx_components

        C = cnmf_obj.estimates.C[component_indices]
        f = cnmf_obj.estimates.f

        if add_background:
            return C + f
        else:
            return C

    @validate("cnmf")
    def get_rcm(
            self,
            frame_indices: Optional[Union[Tuple[int, int], int]] = None,
            component_indices: np.ndarray = None,
    ) -> np.ndarray:
        """
        Return the reconstructed movie with no background, (A * C)

        Parameters
        ----------
        frame_indices: Tuple[int, int], int
            (start_frame, stop_frame), return frames in this range including the ``start_frame``, upto and not
            including the ``stop_frame``
            if single int, return reconstructed movie for single frame indicated

        component_indices: Optional[np.ndarray]
            | component indices to reconstruct movie with
            | if ``None`` uses ``cnmf.estimates.idx_components``

        Returns
        -------
        np.ndarray
            shape is [n_frames, x_pixels, y_pixels]
        """
        cnmf_obj = self.get_output()

        if component_indices is None:
            component_indices = cnmf_obj.estimates.idx_components

        if frame_indices is None:
            frame_indices = (0, cnmf_obj.estimates.C.shape[1])

        if isinstance(frame_indices, int):
            frame_indices = (frame_indices, frame_indices + 1)

        dn = cnmf_obj.estimates.A[:, component_indices].dot(
            cnmf_obj.estimates.C[component_indices, frame_indices[0]: frame_indices[1]]
        )

        return dn.reshape(cnmf_obj.dims + (-1,), order="F").transpose([2, 0, 1])

    @validate("cnmf")
    def get_rcb(
            self,
            frame_indices: Optional[Union[Tuple[int, int], int]] = None,
    ) -> np.ndarray:
        """
        Return the reconstructed background, (b * f)

        Parameters
        ----------
        frame_indices: Tuple[int, int], int
            (start_frame, stop_frame), return frames in this range including the ``start_frame``, upto and not
            including the ``stop_frame``
            if single int, return reconstructed background for single frame indicated

        Returns
        -------
        np.ndarray
            shape is [n_frames, x_pixels, y_pixels]
        """
        cnmf_obj = self.get_output()

        if frame_indices is None:
            frame_indices = (0, cnmf_obj.estimates.C.shape[1])

        if isinstance(frame_indices, int):
            frame_indices = (frame_indices, frame_indices + 1)

        dn = cnmf_obj.estimates.b.dot(
            cnmf_obj.estimates.f[:, frame_indices[0]: frame_indices[1]]
        )
        return dn.reshape(cnmf_obj.dims + (-1,), order="F").transpose([2, 0, 1])

    @validate("cnmf")
    def get_residuals(
            self,
            frame_indices: Optional[Union[Tuple[int, int], int]] = None,
    ) -> np.ndarray:
        """
        Return residuals, raw movie - (A * C) - (b * f)

        Parameters
        ----------
        Tuple[int, int], int
            (start_frame, stop_frame), return residuals for frames in this range including the ``start_frame``, upto and not
            including the ``stop_frame``
            if single int, return residual for single frame indicated

        Returns
        -------
        np.ndarray
            shape is [n_frames, x_pixels, y_pixels]
        """

        cnmf_obj = self.get_output()

        if frame_indices is None:
            frame_indices = (0, cnmf_obj.estimates.C.shape[1])

        if isinstance(frame_indices, int):
            frame_indices = (frame_indices, frame_indices + 1)

        raw_movie = self.get_input_memmap()

        reconstructed_movie = self.get_rcm(frame_indices)

        background = self.get_rcb(frame_indices)

        residuals = raw_movie[np.arange(*frame_indices)] - reconstructed_movie - background

        return residuals.reshape(cnmf_obj.dims + (-1,), order="F").transpose([2, 0, 1])
