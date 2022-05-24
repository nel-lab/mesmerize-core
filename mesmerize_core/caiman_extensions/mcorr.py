from pathlib import Path

import numpy as np
import pandas as pd
from caiman import load_memmap

from ..batch_utils import get_full_data_path
from .common import validate
from typing import *


@pd.api.extensions.register_series_accessor("mcorr")
class MCorrExtensions:
    """
    Extensions for managing motion correction outputs
    """

    def __init__(self, s: pd.Series):
        self._series = s

    @validate("mcorr")
    def get_output_path(self) -> Path:
        """
        Get the path to the motion corrected output memmap file

        Returns
        -------
        Path
            path to the motion correction output memmap file
        """
        return get_full_data_path(self._series["outputs"]["mcorr-output-path"])

    @validate("mcorr")
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
        mc_movie = np.reshape(Yr.T, [T] + list(dims), order="F")
        return mc_movie

    @validate("mcorr")
    def get_shifts(
        self, output_type: str, pw_rigid: bool = True
    ) -> Union[np.ndarray, Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Get x & y shifts

        Parameters
        ----------
        output_type: str
            one of 'matplotlib' or 'napari-1d'.
            'matplotlib' returns ``np.ndarray`` of shape ``[xs, ys]``

        pw_rigid: bool
            if True, return pw_ridid shifts

        Returns
        -------

        """
        path = get_full_data_path(self._series["outputs"]["shifts"])
        shifts = np.load(str(path))

        if output_type == "matplotlib":
            return shifts

        if pw_rigid:
            return shifts
        else:
            n_pts = shifts.shape[0]
            n_lines = shifts.shape[1]
            xs = [np.linspace(0, n_pts, n_pts)]
            ys = []

            for i in range(n_lines):
                ys.append(shifts[:, i])
            return xs, ys
