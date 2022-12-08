from abc import ABC, abstractmethod
import numpy as np
from typing import *


slice_or_int = Union[int, slice]


class LazyArray(ABC):
    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def n_components(self) -> int:
        pass

    @property
    @abstractmethod
    def n_frames(self) -> int:
        pass

    def __getitem__(
            self,
            item: Union[int, Tuple[slice_or_int]]
    ):
        if isinstance(item, int):
            indexer = item

        elif isinstance(item, slice):
            indexer = item

        elif isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with {len(item)} dimensions, "
                    f"only {len(self.shape)} exist in the array"
                )

            indexer = item[0]

        else:
            raise IndexError(
                f"You index EagerArrays only using slice, integer, or tuple of slice and int, "
                f"you have passed a: {type(item)}"
            )

        if isinstance(indexer, slice):
            start = indexer.start
            stop = indexer.stop
            if start is not None:
                if start > self.n_frames:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame start index of {start} "
                                     f"lies beyond `n_frames` {self.n_frames}")
            if stop is not None:
                if stop > self.n_frames:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame stop index of {stop} "
                                     f"lies beyond `n_frames` {self.n_frames}")

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            if isinstance(item, tuple):
                if len(item) == 2:
                    return frames[:, item[1]]
                elif len(item) == 3:
                    return frames[:, item[1], item[2]]

            else:
                return frames

        elif isinstance(indexer, int):
            return self._compute_at_indices(indexer)

    @abstractmethod
    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        pass


class RCMArray(LazyArray):
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
        """
        Shape of the reconstructed movie

        Returns
        -------
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        return self._shape

    def _compute_at_indices(self, indices: Union[int, Tuple[int, int]]) -> np.ndarray:
        rcm = self.spatial.dot(
            self.temporal[:, indices]
        ).reshape(
            self.shape[1:] + (-1,), order="F"
        ).transpose([2, 0, 1])

        if rcm.shape[0] == 1:
            return rcm[0]  # 2d single rame
        else:
            return rcm


class RBArray(LazyArray):
    pass


class ResidualsArray(LazyArray):
    pass
