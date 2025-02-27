from itertools import product as iter_product
from typing import *
from time import time
from warnings import warn

import numpy as np
from scipy.sparse import csc_matrix

from ._base import LazyArray


class LazyArrayRCM(LazyArray):
    """LazyArray for reconstructed movie, i.e. A ⊗ C"""

    def __init__(
        self,
        spatial: np.ndarray,
        temporal: np.ndarray,
        frame_dims: Tuple[int, int],
    ):
        """
        Parameters
        ----------
        spatial: np.ndarray
            spatial components

        temporal: np.ndarray
            temporal components

        frame_dims: Tuple[int, int]
            frame dimensions

        """

        if spatial.shape[1] != temporal.shape[0]:
            raise ValueError(
                f"Number of temporal components provided: `{temporal.shape[0]}` "
                f"does not equal number of spatial components provided: `{spatial.shape[1]}`"
            )

        self._spatial = spatial
        self._temporal = temporal

        self._shape: Tuple[int, int, int] = (temporal.shape[1], *frame_dims)

        # determine dtype
        if self.spatial.dtype == self.temporal.dtype:
            self._dtype = self.temporal.dtype.name
        else:
            self._dtype = self[0].dtype.name

        # precompute min and max vals for each component for spatial and temporal
        temporal_max = np.nanmax(self.temporal, axis=1)
        temporal_min = np.nanmin(self.temporal, axis=1)

        if isinstance(self.spatial, csc_matrix):
            spatial_max = self.spatial.max(axis=0).toarray()
            spatial_min = self.spatial.min(axis=0).toarray()
        else:
            spatial_max = self.spatial.max(axis=0)
            spatial_min = self.spatial.min(axis=0)

        prods = list()
        for t, s in iter_product(
            [temporal_min, temporal_max], [spatial_min, spatial_max]
        ):
            _p = np.multiply(t, s)
            prods.append(np.nanmin(_p))
            prods.append(np.nanmax(_p))

        self._max = np.max(prods)
        self._min = np.min(prods)

        temporal_mean = np.nanmean(self.temporal, axis=1)
        temporal_std = np.nanstd(self.temporal, axis=1)

        self._mean_image = self.spatial.dot(temporal_mean).reshape(
            frame_dims, order="F"
        )
        self._max_image = self.spatial.dot(temporal_max).reshape(frame_dims, order="F")
        self._min_image = self.spatial.dot(temporal_min).reshape(frame_dims, order="F")
        self._std_image = self.spatial.dot(temporal_std).reshape(frame_dims, order="F")

    @property
    def spatial(self) -> np.ndarray:
        return self._spatial

    @property
    def temporal(self) -> np.ndarray:
        return self._temporal

    @property
    def n_components(self) -> int:
        return self._spatial.shape[1]

    @property
    def n_frames(self) -> int:
        return self._temporal.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def mean_image(self) -> np.ndarray:
        """mean projection image"""
        return self._mean_image

    @property
    def max_image(self) -> np.ndarray:
        """max projection image"""
        return self._max_image

    @property
    def min_image(self) -> np.ndarray:
        """min projection image"""
        return self._min_image

    @property
    def std_image(self) -> np.ndarray:
        """standard deviation projection image"""
        return self._std_image

    def _compute_at_indices(self, indices: Union[int, Tuple[int, int]]) -> np.ndarray:
        rcm = (
            self.spatial.dot(self.temporal[:, indices])
            .reshape(self.shape[1:] + (-1,), order="F")
            .transpose([2, 0, 1])
        )

        if rcm.shape[0] == 1:
            return rcm[0]  # 2d single frame
        else:
            return rcm

    def __repr__(self):
        r = super().__repr__()
        return f"{r}" f"n_components: {self.n_components}"

    def __eq__(self, other):
        if not isinstance(other, LazyArrayRCM):
            raise TypeError(
                f"cannot compute equality for against types that are not {self.__class__.__name__}"
            )

        if (self.spatial == other.spatial) and (self.temporal == other.temporal):
            return True
        else:
            return False


# implementation for reconstructed background is identical
# this is just an interface to separate them
class LazyArrayRCB(LazyArrayRCM):
    """Lazy array for reconstructed background, i.e. b ⊗ f"""


class LazyArrayResiduals(LazyArray):
    """Lazy array for residuals, i.e. Y - (A ⊗ C) - (b ⊗ f)"""

    def __init__(
        self,
        raw_movie: np.ndarray,
        rcm: LazyArrayRCM,
        rcb: LazyArrayRCB,
        timeout: int = 10,
    ):
        """
        Create a LazyArray of the residuals, ``Y - (A ⊗ C) - (b ⊗ f)``

        Parameters
        ----------
        raw_movie: np.memmap
            numpy memmap of the raw movie

        rcm: LazyArrayRCM
            reconstructed movie lazy array

        rcb: LazyArrayRCB
            reconstructed background lazy array

        timeout: int, default ``10``
            timeout for min-max calculation, not implemented yet

        """
        self._raw_movie = raw_movie
        self._rcm = rcm
        self._rcb = rcb

        # I was lazy to figure out how to copy tuples
        self._shape = (
            self._raw_movie.shape[0],
            self._raw_movie.shape[1],
            self._raw_movie.shape[2],
        )

        if self._raw_movie.dtype == self._rcm.dtype == self._rcb.dtype:
            self._dtype = self._raw_movie.dtype
        else:
            self._dtype = self[0].dtype.name

        # TODO: implement min max for residuals
        # min_max_raw = self._quick_min_max(raw_movie, timeout)
        # if min_max_raw is None:
        #     self._min = None
        #     self._max = None

        # else:
        #     _min, _max = min_max_raw
        #
        #     _min = _min - self._rcm.max - self._rcb.max
        #     _max = _max -

    def _quick_min_max(self, data, timeout):
        # adapted from pyqtgraph.ImageView
        # Estimate the min/max values of *data* by subsampling.
        # Returns [(min, max), ...] with one item per channel

        t = time()
        while data.size > 1e6:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            data = data[tuple(sl)]
            if (time() - t) > timeout:
                return None

        return float(np.nanmin(data)), float(np.nanmax(data))

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def n_frames(self) -> int:
        return self._shape[0]

    # TODO: implement min max for residuals
    @property
    def min(self) -> float:
        warn(
            "min and max not yet implemented for LazyArrayResiduals. "
            "Using first frame of raw movie"
        )
        return float(self._raw_movie[0].min())

    @property
    def max(self) -> float:
        warn(
            "min and max not yet implemented for LazyArrayResiduals. "
            "Using first frame of raw movie"
        )
        return float(self._raw_movie[0].max())

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        residuals = self._raw_movie[indices] - self._rcm[indices] - self._rcb[indices]
        return residuals
