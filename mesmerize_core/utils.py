"""
Useful functions adapted from old mesmerize

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""

import numpy as np
from functools import wraps
import os
from stat import S_IEXEC
from typing import *
import re as regex
from pathlib import Path
from warnings import warn
import sys
from tempfile import NamedTemporaryFile
from subprocess import check_call
import shlex
import mslex

if os.name == "nt":
    IS_WINDOWS = True
    HOME = "USERPROFILE"
    lex = mslex
else:
    IS_WINDOWS = False
    HOME = "HOME"
    lex = shlex

if "MESMERIZE_LRU_CACHE" in os.environ.keys():
    MESMERIZE_LRU_CACHE = os.environ["MESMERIZE_LRU_CACHE"]
else:
    MESMERIZE_LRU_CACHE = 10


def warning_experimental(more_info: str = ""):
    """
    decorator to warn the user that the function is experimental
    """

    def catcher(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            warn(
                f"You are trying to use the following experimental feature, "
                f"this may change in the future without warning:\n"
                f"{func.__qualname__}\n"
                f"{more_info}\n",
                FutureWarning,
                stacklevel=2,
            )
            return func(self, *args, **kwargs)

        return fn

    return catcher


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
            filename = os.path.join(os.environ[HOME], "run.bat")
        else:
            filename = os.path.join(os.environ[HOME], "run.sh")
    else:
        if IS_WINDOWS:
            if not filename.endswith(".bat"):
                filename = filename + ".bat"

    sh_file = filename

    if args_str is None:
        args_str = ""

    if not IS_WINDOWS:
        # remove file first if it exists - avoid issues with ownership
        if os.path.exists(sh_file):
            os.remove(sh_file)

        with open(sh_file, "w") as f:

            f.write(f"#!/bin/bash\n")

            if "VIRTUAL_ENV" in os.environ.keys():
                f.write(
                    f'export PATH={lex.quote(os.environ["PATH"])}\n'
                    f'export VIRTUAL_ENV={lex.quote(os.environ["VIRTUAL_ENV"])}\n'
                    f'export LD_LIBRARY_PATH={lex.quote(os.environ["LD_LIBRARY_PATH"])}\n'
                )

            if "PYTHONPATH" in os.environ.keys():
                f.write(f'export PYTHONPATH={lex.quote(os.environ["PYTHONPATH"])}\n')

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

            if "CONDA_PREFIX" in os.environ.keys():
                # add command to run the python script in the conda environment
                # that was active at the time that this shell script was generated
                f.write(
                    f'{lex.quote(os.environ["CONDA_EXE"])} run -p {lex.quote(os.environ["CONDA_PREFIX"])} '
                    f"python {lex.quote(module_path)} {args_str}"
                )
            else:
                f.write(
                    f"python {lex.quote(module_path)} {args_str}"
                )  # call the script to run

    else:
        with open(sh_file, "w") as f:
            for k, v in os.environ.items():  # copy the current environment
                if regex.match(r"^.*[()]", str(k)) or regex.match(r"^.*[()]", str(v)):
                    continue
                f.write(f"SET {k}={lex.quote(v)}\n")
            f.write(f"{lex.quote(sys.executable)} {lex.quote(module_path)} {args_str}")

    st = os.stat(sh_file)
    os.chmod(sh_file, st.st_mode | S_IEXEC)

    print(sh_file)

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
