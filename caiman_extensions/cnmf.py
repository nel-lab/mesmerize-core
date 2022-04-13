from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from caiman import load_memmap
from caiman.source_extraction.cnmf import CNMF
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.visualization import get_contours as caiman_get_contours

from ..batch_utils import get_full_data_path
from .common import validate


@pd.api.extensions.register_series_accessor("cnmf")
class CNMFExtensions:
    """
    Extensions for managing CNMF output data
    """
    def __init__(self, s: pd.Series):
        self._series = s

    def get_cnmf_memmap(self) -> np.ndarray:
        """
        Get the CNMF memmap

        Returns
        -------
        np.ndarray
            numpy memmap array used for CNMF
        """
        path = get_full_data_path(self._series['outputs']['cnmf-memmap-path'])
        # Get order f images
        Yr, dims, T = load_memmap(str(path))
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        return images

    def get_input_memmap(self) -> np.ndarray:
        """
        Return the F-order memmap if the input to the
        CNMF batch item was a mcorr output memmap

        Returns
        -------
        np.ndarray
            numpy memmap array of the input
        """
        movie_path = str(self._series.caiman.get_input_movie_path())
        if movie_path.endswith('mmap'):
            Yr, dims, T = load_memmap(movie_path)
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            return images
        else:
            raise TypeError(f"Input movie for CNMF was not a memmap, path to input movie is:\n"
                            f"{movie_path}")

    # TODO: Cache this globally so that a common upper cache limit is valid for ALL batch items
    @validate('cnmf')
    def get_output_path(self) -> Path:
        """
        Returns
        -------
        Path
            Path to the Caiman CNMF hdf5 output file
        """
        return get_full_data_path(self._series['outputs']['cnmf-hdf5-path'])

    #@lru_cache(MESMERIZE_LRU_CACHE)
    @validate('cnmf')
    def get_output(self) -> CNMF:
        """
        Returns
        -------
        CNMF
            Returns the Caiman CNMF object
        """
        # Need to create a cache object that takes the item's UUID and returns based on that
        # collective global cache
        return load_CNMF(self.get_output_path())

    # TODO: Make the ``ixs`` parameter for spatial stuff optional
    @validate('cnmf')
    def get_spatial_masks(self, ixs_components: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Get binary masks of the spatial components at the given `ixs`

        Basically created from cnmf.estimates.A

        Parameters
        ----------
        ixs_components: np.ndarray
            numpy array containing integer indices for which you want spatial masks

        threshold: float
            threshold

        Returns
        -------
        np.ndarray
            shape is [dim_0, dim_1, n_components]

        """
        cnmf_obj = self.get_output()

        dims = cnmf_obj.dims
        if dims is None:
            dims = cnmf_obj.estimates.dims

        masks = np.zeros(shape=(dims[0], dims[1], len(ixs_components)), dtype=bool)

        for n, ix in enumerate(ixs_components):
            s = cnmf_obj.estimates.A[:, ix].toarray().reshape(cnmf_obj.dims)
            s[s >= threshold] = 1
            s[s < threshold] = 0

            masks[:, :, n] = s.astype(bool)

        return masks

    # TODO: Cache this globally so that a common upper cache limit is valid for ALL batch items
    @staticmethod
    @lru_cache(5)
    def _get_spatial_contour_coors(cnmf_obj: CNMF):
        dims = cnmf_obj.dims
        if dims is None:  # I think that one of these is `None` if loaded from an hdf5 file
            dims = cnmf_obj.estimates.dims

        # need to transpose these
        dims = dims[1], dims[0]

        contours = caiman_get_contours(
            cnmf_obj.estimates.A,
            dims,
            swap_dim=True
        )

        return contours

    @validate('cnmf')
    def get_spatial_contours(self, ixs_components: np.ndarray) -> List[dict]:
        """
        Get the contours for the spatial footprints

        Parameters
        ----------
        ixs_components: np.ndarray
            indices for which to return spatial contours

        Returns
        -------

        """
        cnmf_obj = self.get_output()
        contours = self._get_spatial_contour_coors(cnmf_obj)

        contours_selection = list()
        for i in range(len(contours)):
            if i in ixs_components:
                contours_selection.append(contours[i])

        return contours_selection

    @validate('cnmf')
    def get_spatial_contour_coors(self, ixs_components: np.ndarray) -> List[np.ndarray]:
        contours = self.get_spatial_contours(ixs_components)

        coordinates = []
        for contour in contours:
            coors = contour['coordinates']
            coordinates.append(coors[~np.isnan(coors).any(axis=1)])

        return coordinates

    @validate('cnmf')
    def get_temporal_components(self, ixs_components: np.ndarray = None, add_background: bool = True) -> np.ndarray:
        """
        Get the temporal components for this CNMF item

        Parameters
        ----------
        ixs_components: np.ndarray
            indices for which to return temporal components, ``cnmf.estimates.C``

        add_background: bool
            if ``True``, add the temporal background, basically ``cnmf.estimates.C + cnmf.estimates.f``

        Returns
        -------

        """
        cnmf_obj = self.get_output()

        if ixs_components is None:
            ixs_components = np.arange(0, cnmf_obj.estimates.C.shape[0])

        C = cnmf_obj.estimates.C[ixs_components]
        f = cnmf_obj.estimates.f

        if add_background:
            return C + f
        else:
            return C

    # TODO: Cache this globally so that a common upper cache limit is valid for ALL batch items
    @validate('cnmf')
    def get_reconstructed_movie(self, ixs_frames: Tuple[int, int] = None, add_background: bool = True) -> np.ndarray:
        """
        Return the reconstructed movie, (A * C) + (b * f)

        Parameters
        ----------
        ixs_frames: Tuple[int, int]
            (start_frame, stop_frame), return frames in this range including the ``start_frame``, upto and not
            including the ``stop_frame``

        add_background: bool
            if ``True``, add the spatial & temporal background, b * f

        Returns
        -------
        np.ndarray
            shape is [n_frames, x_pixels, y_pixels]
        """
        cnmf_obj = self.get_output()

        if ixs_frames is None:
            ixs_frames = (0, cnmf_obj.estimates.C.shape[1])

        dn = (cnmf_obj.estimates.A.dot(cnmf_obj.estimates.C[:, ixs_frames[0]:ixs_frames[1]]))

        if add_background:
            dn += (cnmf_obj.estimates.b.dot(cnmf_obj.estimates.f[:, ixs_frames[0]:ixs_frames[1]]))
        return dn.reshape(cnmf_obj.dims + (-1,), order='F').transpose([2, 0, 1])