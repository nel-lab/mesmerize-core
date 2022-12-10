from abc import ABC, abstractmethod
import numpy as np
from typing import *
from warnings import warn
from pathlib import Path


slice_or_int = Union[int, slice]


class LazyArray(ABC):
    """
    Base class for arrays that exhibit lazy computation upon indexing
    """
    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        str
            data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        pass

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """
        int
            number of frames
        """
        pass

    @property
    @abstractmethod
    def min(self) -> float:
        """
        int
            min value of the array if it were fully computed
        """
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        """
        float
            max value of the array if it were fully computed
        """
        pass

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """
        int
            number of bytes for the array if it were fully computed
        """
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,))

    @property
    def nbytes_gb(self) -> float:
        """
        float
            number of gigabytes for the array if it were fully computed
        """
        return self.nbytes / 1e9

    @abstractmethod
    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here. Computes the array at the desired indices.

        Parameters
        ----------
        indices: Union[int, slice]
            the user's desired slice, i.e. slice object or int passed from `__getitem__()`

        Returns
        -------
        np.ndarray
            array at the indexed slice
        """
        pass

    def as_numpy(self):
        """
        NOT RECOMMENDED, THIS COULD BE EXTREMELY LARGE. Converts to a standard numpy array in RAM.

        Returns
        -------
        np.ndarray
        """
        warn(
            f"\nYou are trying to create a numpy.ndarray from a LazyArray, "
            f"this is not recommended and could take a while.\n\n"
            f"Estimated size of final numpy array: "
            f"{self.nbytes_gb:.2f} GB"
        )
        a = np.zeros(shape=self.shape, dtype=self.dtype)

        for i in range(self.n_frames):
            a[i] = self[i]

        return a

    def save_hdf5(self, filename: Union[str, Path]):
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
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )

            indexer = item[0]

        else:
            raise IndexError(
                f"You can index LazyArrays only using slice, int, or tuple of slice and int, "
                f"you have passed a: <{type(item)}>"
            )

        if isinstance(indexer, slice):
            start = indexer.start
            stop = indexer.stop
            if start is not None:
                if start > self.n_frames:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame start index of <{start}> "
                                     f"lies beyond `n_frames` <{self.n_frames}>")
            if stop is not None:
                if stop > self.n_frames:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame stop index of <{stop}> "
                                     f"lies beyond `n_frames` <{self.n_frames}>")

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

    def __repr__(self):
        return f"{self.__class__.__name__} @{hex(id(self))}\n" \
               f"{self.__class__.__doc__}\n" \
               f"Frames are computed only upon indexing\n" \
               f"shape [frames, x, y]: {self.shape}\n" \


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
