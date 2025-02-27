from warnings import warn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import *

import numpy as np

slice_or_int_or_range = Union[int, slice, range]


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
        float
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
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,), dtype=np.int64)

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

    def __getitem__(self, item: Union[int, Tuple[slice_or_int_or_range]]):
        if isinstance(item, int):
            indexer = item

        # numpy int scaler
        elif isinstance(item, np.integer):
            indexer = item.item()

        # treat slice and range the same
        elif isinstance(item, (slice, range)):
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

        # treat slice and range the same
        if isinstance(indexer, (slice, range)):
            start = indexer.start
            stop = indexer.stop
            step = indexer.step

            if start is not None:
                if start > self.n_frames:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame start index of <{start}> "
                        f"lies beyond `n_frames` <{self.n_frames}>"
                    )
            if stop is not None:
                if stop > self.n_frames:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame stop index of <{stop}> "
                        f"lies beyond `n_frames` <{self.n_frames}>"
                    )

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            indexer = slice(start, stop, step)  # in case it was a range object

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            # index the remaining dims after lazy computing the frame(s)
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
        return (
            f"{self.__class__.__name__} @{hex(id(self))}\n"
            f"{self.__class__.__doc__}\n"
            f"Frames are computed only upon indexing\n"
            f"shape [frames, x, y]: {self.shape}\n"
        )
