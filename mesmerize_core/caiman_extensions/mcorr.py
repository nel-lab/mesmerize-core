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

    @validate('mcorr')
    def get_shifts(self) -> Path:
        path = get_full_data_path(self._series['outputs']['shifts'])
        return np.load(str(path))

    @validate('mcorr')
    def get_shifts_array(self, pw_rigid=True) -> np.ndarray:
        if pw_rigid:
            x_shifts, y_shifts = self.get_shifts()
        else:
            shifts = self.get_shifts()
            n_pts = shifts.shape[0]
            n_lines = shifts.shape[1]
            xs = [np.linspace(0, n_pts, n_pts)]
            ys = []

            for i in range(n_lines):
                ys.append(shifts[:, i])
        return xs, ys


