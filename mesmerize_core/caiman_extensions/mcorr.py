from pathlib import Path

import numpy as np
import pandas as pd
from caiman import load_memmap

from .common import validate
from typing import *
from .cache import Cache

cache = Cache()


@pd.api.extensions.register_series_accessor("mcorr")
class MCorrExtensions:
    """
    Extensions for managing motion correction outputs
    """

    def __init__(self, s: pd.Series):
        self._series = s

    @validate("mcorr")
    @cache.use_cache
    def get_output_path(self) -> Path:
        """
        Get the path to the motion corrected output memmap file

        Returns
        -------
        Path
            path to the motion correction output memmap file
        """
        return self._series.paths.resolve(self._series["outputs"]["mcorr-output-path"])

    @validate("mcorr")
    @cache.use_cache
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
    @cache.use_cache
    def get_shifts(
        self, pw_rigid: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Gets file path to shifts array (.npy file) for item, processes shifts array
        into a list of x and y shifts based on whether rigid or nonrigid
        motion correction was performed.

        Parameters:
        -----------
        pw_rigid: bool - flag for whether shifts are for rigid or nonrigid motion correction
            True = Nonrigid (elastic/piecewise)
            False = Rigid
        Returns:
        --------
        List of Processed X and Y shifts arrays
        """
        path = self._series.paths.resolve(self._series["outputs"]["shifts"])
        shifts = np.load(str(path))

        if pw_rigid:
            n_pts = shifts.shape[1]
            n_lines = shifts.shape[2]
            xs = [np.linspace(0, n_pts, n_pts)]
            ys = []

            for i in range(shifts.shape[0]):
                for j in range(n_lines):
                    ys.append(shifts[i, :, j])
        else:
            n_pts = shifts.shape[0]
            n_lines = shifts.shape[1]
            xs = [np.linspace(0, n_pts, n_pts)]
            ys = []

            for i in range(n_lines):
                ys.append(shifts[:, i])
        return xs, ys
