from pathlib import Path

import numpy as np
import pandas as pd
from caiman import load_memmap

from ._utils import validate


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
    def get_output(self, mode: str = "r") -> np.ndarray:
        """
        Get the motion corrected output as a memmaped numpy array, allows fast random-access scrolling.

        Parameters
        ----------

        mode: str
            passed to numpy.memmap

            one of: `{'r+', 'r', 'w+', 'c'}`

            The file is opened in this mode:

            +------+-------------------------------------------------------------+
            | 'r'  | Open existing file for reading only.                        |
            +------+-------------------------------------------------------------+
            | 'r+' | Open existing file for reading and writing.                 |
            +------+-------------------------------------------------------------+
            | 'w+' | Create or overwrite existing file for reading and writing.  |
            +------+-------------------------------------------------------------+
            | 'c'  | Copy-on-write: assignments affect data in memory, but       |
            |      | changes are not saved to disk.  The file on disk is         |
            |      | read-only.                                                  |
            +------+-------------------------------------------------------------+

        Returns
        -------
        np.ndarray
            memmap numpy array of the motion corrected movie

        Examples
        --------

        This example visualizes the raw movie and mcorr movie side by side.
        Needs fastplotlib and must be run in a notebook

        .. code-block:: python

            from mesmerize_core import load_batch
            from fastplotlib import ImageWidget

            df = load_batch("/path/to/batch_dataframe_file.pickle")

            # assumes item at 0th index is a mcorr batch item
            input_movie = df.iloc[0].caiman.get_input_movie()
            mcorr_movie = df.iloc[0].mcorr.get_output()

            mcorr_iw = ImageWidget(
                data=[input_movie, mcorr_movie],
                vmin_vmax_sliders=True,
                cmap="gnuplot2"
            )
            mcorr_iw.show()

        """
        path = self.get_output_path()
        Yr, dims, T = load_memmap(str(path), mode=mode)
        mc_movie = np.reshape(Yr.T, [T] + list(dims), order="F")
        return mc_movie

    @validate("mcorr")
    def get_shifts(self, pw_rigid) -> list[np.ndarray]:
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
        List of Processed X and Y [and Z] shifts arrays
        - For rigid correction, each element is a vector of length n_frames
        - For pw_rigid correction, each element is an n_frames x n_patches matrix
        """
        path = self._series.paths.resolve(self._series["outputs"]["shifts"])
        shifts = np.load(str(path))

        if pw_rigid:
            shifts_by_dim = list(
                shifts
            )  # dims-length list of n_frames x n_patches matrices
        else:
            shifts_by_dim = list(
                shifts.T
            )  # dims-length list of n_frames-length vectors

        return shifts_by_dim
