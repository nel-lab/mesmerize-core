from typing import *

import numpy as np

from ._base import LazyArray


class LazyArrayRCM(LazyArray):
    """LazyArray for reconstructed movie, i.e. A ⊗ C"""
    def __init__(
            self,
            spatial: np.ndarray,
            temporal: np.ndarray,
            frame_dims: Tuple[int, int],
    ):

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
            self._dtype = self.temporal.dtype
        else:
            self._dtype = self[0].dtype.name

        # precompute min and max vals
        temporal_max = np.nanmax(self.temporal, axis=1)
        spatial_max = self.spatial.max(axis=0).toarray()

        temporal_min = np.nanmin(self.temporal, axis=1)
        spatial_min = self.spatial.min(axis=0).toarray()

        self._max = np.nanmax(np.multiply(temporal_max, spatial_max))

        self._min = min(
            np.nanmin(np.multiply(temporal_max, spatial_min)),
            np.nanmin(np.multiply(temporal_min, spatial_max))
        )

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

    def _compute_at_indices(self, indices: Union[int, Tuple[int, int]]) -> np.ndarray:
        rcm = self.spatial.dot(
            self.temporal[:, indices]
        ).reshape(
            self.shape[1:] + (-1,), order="F"
        ).transpose([2, 0, 1])

        if rcm.shape[0] == 1:
            return rcm[0]  # 2d single frame
        else:
            return rcm

    def __repr__(self):
        r = super().__repr__()
        return f"{r}" \
               f"n_components: {self.n_components}"

    def __eq__(self, other):
        if not isinstance(other, LazyArrayRCM):
            raise TypeError(f"cannot compute equality for against types that are not {self.__class__.__name__}")

        if (self.spatial == other.spatial) and (self.temporal == other.temporal):
            return True
        else:
            return False


class LazyArrayRCB(LazyArray):
    pass


class LazyArrayResiduals(LazyArray):
    pass
