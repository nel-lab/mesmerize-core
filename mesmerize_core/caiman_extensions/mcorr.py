from pathlib import Path

import numpy as np
import pandas as pd
from caiman import load_memmap

from .common import validate
from typing import *
from .cache import Cache


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
        return self._series.paths.resolve(self._series["outputs"]["mcorr-output-path"])

    @validate("mcorr")
    def get_output(self) -> np.ndarray:
        """
        Get the motion corrected output as a memmaped numpy array, allows fast random-access scrolling.

        Returns
        -------
        np.ndarray
            memmap numpy array of the motion corrected movie

        Examples
        --------

        View the raw movie and mcorr movie side by side

        .. code-block:: python
            from mesmerize_core import load_batch

            # needs fastplotlib and must be run in a notebook
            from fastplotlib import GridPlot, Image
            from ipywidgets import IntSlider, VBox

            df = load_batch("/path/to/batch_dataframe_file.pickle")

            # assumes item at 0th index is a mcorr batch item
            input_movie = df.iloc[0].caiman.get_input_movie()
            mcorr_movie = df.iloc[0].mcorr.get_output()

            # gridplot with 1 row 2 columns
            # synced controllers for pan and zoom
            gp = GridPlot(shape=(1, 2), controllers="sync")

            # create the graphics for input and mcorr movies
            input_graphic = Image(input_movie[0], cmap="gnuplot2")
            mcorr_graphic = Image(mcorr_movie[0], cmap="gnuplot2")

            # add the corresponding graphic to each subplot in the gridplot
            gp.subplots[0, 0].add_graphic(input_graphic)
            gp.subplots[0, 1].add_graphic(mcorr_graphic)

            slider = IntSlider(value=0, min=0, max=mcorr_graphic.shape[0] - 1, step=1)

            previous_slider_value = 0
            def update_frame():  # runs on each rendering cycle
                if previous_slider_value == slider.value:
                    return

                input_graphic.update_data(input_movie[slider.value])
                mcorr_graphic.update_data(mcorr_movie[slider.value])

            gp.add_animdations([update_frame])

            VBox([gp.show(), slider])
            
        """
        path = self.get_output_path()
        Yr, dims, T = load_memmap(str(path))
        mc_movie = np.reshape(Yr.T, [T] + list(dims), order="F")
        return mc_movie

    @validate("mcorr")
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
