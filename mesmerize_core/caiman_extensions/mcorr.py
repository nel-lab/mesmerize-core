from pathlib import Path

import numpy as np
import pandas as pd
from caiman import load_memmap

from ..batch_utils import get_full_data_path
from .common import validate
from matplotlib import pyplot as plt
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

    def shifts_handler(
            self, shifts: np.ndarray, pw_rigid: bool=False
    ):
        """
        Handler function for processing shifts array

        Parameters:
        -----------
        shifts: np.ndarray of shifts from .npy file
        plot_type: str - napari-1d or matplotlib - determines how process shifts array
        pw_rigid: bool - flag for whether shifts are for rigid or nonrigid motion correction

        Returns:
        --------
        processed shifts results
        """
        if pw_rigid:
            n_pts = shifts.shape[1]
            n_lines = shifts.shape[2]
            xs = [np.linspace(0, n_pts, n_pts)]
            ys = []

            for i in range(shifts.shape[0]):
                for j in range(n_lines):
                    ys.append(shifts[i,:,j])
            return xs, ys
        else:
            n_pts = shifts.shape[0]
            n_lines = shifts.shape[1]
            xs = [np.linspace(0, n_pts, n_pts)]
            ys = []

            for i in range(n_lines):
                ys.append(shifts[:, i])
            return xs, ys

    @validate("mcorr")
    def get_shifts(self) -> np.ndarray:
        """
        Get x & y shifts

        Returns
        -------
        shifts: array of x and y shifts
            for piecewise MC, x and y shifts for all blocks
            for rigid MC, one pair of shifts for entire movie
        """
        path = get_full_data_path(self._series["outputs"]["shifts"])
        shifts = np.load(str(path))
        return shifts