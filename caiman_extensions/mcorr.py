from pathlib import Path

import numpy as np
import pandas as pd
from caiman import load_memmap

from ..batch_utils import get_full_data_path
from .common import validate


@pd.api.extensions.register_series_accessor("mcorr")
class MCorrExtensions:
    """
    Extensions for managing motion correction outputs
    """
    def __init__(self, s: pd.Series):
        self._series = s

    @validate('mcorr')
    def get_output_path(self) -> Path:
        """
        Get the path to the motion corrected output memmap file

        Returns
        -------
        Path
            path to the motion correction output memmap file
        """
        return get_full_data_path(self._series['outputs']['mcorr-output-path'])

    @validate('mcorr')
    def get_output(self) -> np.ndarray:
        """
        Get the motion corrected output as a memmaped numpy array, allows fast random-access scrolling.

        Returns
        -------
        np.ndarray
            memmap numpy array of the motion corrected movie
        """
        path = self.get_output_path()
        Yr, dims, T = load_memmap(str(path))
        mc_movie = np.reshape(Yr.T, [T] + list(dims), order='F')
        return mc_movie