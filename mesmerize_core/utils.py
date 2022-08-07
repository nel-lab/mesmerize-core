"""
Useful functions adapted from old mesmerize

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""


import numpy as np
from matplotlib import cm as matplotlib_color_map
from functools import wraps
import os
from stat import S_IEXEC
from typing import *
import re as regex
from pathlib import Path
from warnings import warn
import sys


if os.name == "nt":
    IS_WINDOWS = True
    HOME = "USERPROFILE"
else:
    IS_WINDOWS = False
    HOME = "HOME"

if "MESMERIZE_LRU_CACHE" in os.environ.keys():
    MESMERIZE_LRU_CACHE = os.environ["MESMERIZE_LRU_CACHE"]
else:
    MESMERIZE_LRU_CACHE = 10


qualitative_colormaps = [
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]


def warning_experimental(more_info: str = ""):
    """
    decorator to warn the user that the function is experimental
    """
    def catcher(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            warn(
                f"You are trying to use the following experimental feature, "
                f"this maybe change in the future without warning:\n"
                f"{func.__qualname__}\n"
                f"{more_info}\n",
                FutureWarning,
                stacklevel=2
            )
            return func(self, *args, **kwargs)
        return fn
    return catcher


def validate_path(path: Union[str, Path]):
    if not regex.match("^[A-Za-z0-9@\/\\\:._-]*$", str(path)):
        raise ValueError(
            "Paths must only contain alphanumeric characters, "
            "hyphens ( - ), underscores ( _ ) or periods ( . )"
        )
    return path


def auto_colormap(
    n_colors: int,
    cmap: str = "hsv",
    output: Union[str, type] = "float",
    spacing: str = "uniform",
    alpha: float = 1.0,
) -> List[Union[np.ndarray, str]]:
    """
    If non-qualitative map: returns list of colors evenly spread through the chosen colormap.
    If qualitative map: returns subsequent colors from the chosen colormap

    Parameters
    ----------
    n_colors: int
        Numbers of colors to return

    cmap: str
        name of colormap

    output: Union[str, type]
        option: "float" or ``float`` returns RGBA values between 0-1: [R, G, B, A],
        option: "hex" returns hex strings that correspond to the RGBA values

    spacing: str
        option: "uniform"'" returns evenly spaced colors across the entire cmap range
        option: "subsequent" returns subsequent colors from the cmap

    alpha: float
        alpha level, 0.0 - 1.0

    Returns
    -------
    List[Union[np.ndarray, str]]
        List of colors as either ``numpy.ndarray``, or hex ``str`` with length ``n_colors``
    """

    valid = ["float", float, "hex"]
    if output not in valid:
        raise ValueError(f"output must be one {valid}")

    valid = ["uniform", "subsequent"]
    if spacing not in valid:
        raise ValueError(f"spacing must be one of either {valid}")

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be within 0.0 and 1.0")

    cm = matplotlib_color_map.get_cmap(cmap)
    cm._init()

    lut = (cm._lut).view(np.ndarray)

    lut[:, 3] *= alpha

    if spacing == "uniform":
        if not cmap in qualitative_colormaps:
            if cmap == "hsv":
                cm_ixs = np.linspace(30, 210, n_colors, dtype=int)
            else:
                cm_ixs = np.linspace(0, 210, n_colors, dtype=int)
        else:
            if n_colors > len(lut):
                raise ValueError("Too many colors requested for the chosen cmap")
            cm_ixs = np.arange(0, len(lut), dtype=int)
    else:
        cm_ixs = range(n_colors)

    colors = []
    for ix in range(n_colors):
        c = lut[cm_ixs[ix]]

        if output == "hex":
            c = tuple(c[:3] * 255)
            hc = "#%02x%02x%02x" % tuple(map(int, c))
            colors.append(hc)

        else:  # floats
            colors.append(c)

    return colors


def make_runfile(
    module_path: str, args_str: Optional[str] = None, filename: Optional[str] = None
) -> str:
    """
    Make an executable bash script.
    Used for running python scripts in external processes within the same python environment as the main/parent process.

    Parameters
    ----------
    module_path: str
        absolute path to the python module/script that should be run externally

    args_str: Optional[str]
        optinal str of args that is directly passed to the script specified by ``module_path``

    filename: Optional[str]
        optional, filename of the executable bash script

    Returns
    -------
    str
        path to the shell script that can be executed
    """

    if filename is None:
        if IS_WINDOWS:
            sh_file = os.path.join(os.environ[HOME], "run.ps1")
        else:
            sh_file = os.path.join(os.environ[HOME], "run.sh")
    else:
        if IS_WINDOWS:
            if not filename.endswith(".ps1"):
                filename = filename + ".ps1"

    sh_file = filename

    if args_str is None:
        args_str = ""

    if not IS_WINDOWS:
        with open(sh_file, "w") as f:

            f.write(f"#!/bin/bash\n")

            if "CONDA_PREFIX" in os.environ.keys():
                f.write(
                    f'export CONDA_PREFIX={os.environ["CONDA_PREFIX"]}\n'
                    f'export CONDA_PYTHON_EXE={os.environ["CONDA_PYTHON_EXE"]}\n'
                    f'export CONDA_PREFIX_1={os.environ["CONDA_PREFIX_1"]}\n'
                )

            elif "VIRTUAL_ENV" in os.environ.keys():
                f.write(
                    f'export PATH={os.environ["PATH"]}\n'
                    f'export VIRTUAL_ENV={os.environ["VIRTUAL_ENV"]}\n'
                    f'export LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]}\n'
                )

            if "PYTHONPATH" in os.environ.keys():
                f.write(f'export PYTHONPATH={os.environ["PYTHONPATH"]}\n')

            # for k, v in os.environ.items():  # copy the current environment
            #     if '\n' in v:
            #         continue
            #
            # f.write(f'export {k}="{v}"\n')

            # User-setable n-processes
            if "MESMERIZE_N_PROCESSES" in os.environ.keys():
                f.write(
                    f'export MESMERIZE_N_PROCESSES={os.environ["MESMERIZE_N_PROCESSES"]}\n'
                )

            f.write(f"export OPENBLAS_NUM_THREADS=1\n" f"export MKL_NUM_THREADS=1\n")

            f.write(f"python {module_path} {args_str}")  # call the script to run

    else:
        with open(sh_file, "w") as f:
            for k, v in os.environ.items():  # copy the current environment
                if regex.match("^.*[\(\)]", str(k)) or regex.match("^.*[\(\)]", str(v)):
                    continue
                f.write(f'$env:{k}="{v}";\n')

            f.write(f"{sys.executable} {module_path} {args_str}")

    st = os.stat(sh_file)
    os.chmod(sh_file, st.st_mode | S_IEXEC)

    return sh_file


def quick_min_max(data: np.ndarray) -> Tuple[float, float]:
    # from pyqtgraph.ImageView
    # Estimate the min/max values of *data* by subsampling.
    # Returns [(min, max), ...] with one item per channel
    while data.size > 1e6:
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, 2)
        data = data[tuple(sl)]

    return float(np.nanmin(data)), float(np.nanmax(data))


def _organize_coordinates(contour: dict):
    coors = contour["coordinates"]
    coors = coors[~np.isnan(coors).any(axis=1)]

    return coors
