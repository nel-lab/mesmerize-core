from typing import *
from pathlib import Path
from warnings import warn

import numpy as np
import tifffile

from ._base import LazyArray


class LazyTiff(LazyArray):
    def __init__(self, path: Union[Path, str], shape: Tuple[int] = None):
        """
        Lazy reader for tiff files. WIP, works for some tiff files.
        Try ``tifffile.memmap`` first before trying ``LazyTiff``

        Parameters
        ----------
        path: str or Path
            path to tiff file

        shape: Tuple[int]
            manually set shape
        """

        self._tif = tifffile.TiffFile(path)
        tiffseries = self._tif.series[0].levels[0]

        if shape is None:
            # TODO: someone who's better with tiff can help on this
            if len(self._tif.pages) == 1:
                n_frames = len(self._tif.series)
            else:
                n_frames = len(self._tif.pages)

            self._shape = (n_frames, *tiffseries.shape)
        else:
            self._shape = shape

        self._dtype = tiffseries.dtype.name

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def min(self) -> float:
        warn("min not implemented for LazyTiff, returning min of 0th index")
        return self[0].min()

    @property
    def max(self) -> float:
        warn("max not implemented for LazyTiff, returning min of 0th index")
        return self[0].max()

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        return self._tif.asarray(key=indices)
