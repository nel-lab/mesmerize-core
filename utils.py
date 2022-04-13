from qtpy.QtWidgets import QWidget, QFileDialog, QMessageBox
from qtpy import QtGui
import numpy as np
from matplotlib import cm as matplotlib_color_map
from functools import wraps
import os
from stat import S_IEXEC
import traceback
from typing import *
import re as regex
from pathlib import Path


# Useful functions adapted from mesmerize


# to use powershell to run the CNMF process using QProcess
# napari's built in @thread_worker locks up the entire application
if os.name == 'nt':
    IS_WINDOWS = True
    HOME = 'USERPROFILE'
else:
    IS_WINDOWS = False
    HOME = 'HOME'

if 'MESMERIZE_LRU_CACHE' in os.environ.keys():
    MESMERIZE_LRU_CACHE = os.environ['MESMERIZE_LRU_CACHE']
else:
    MESMERIZE_LRU_CACHE = 10


qualitative_colormaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
              'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']


def validate_path(path: Union[str, Path]):
    if not regex.match("^[A-Za-z0-9\/\\:._-]*$", str(path)):
        raise ValueError("Paths must only contain alphanumeric characters, "
                         "hyphens ( - ), underscores ( _ ) or periods ( . )")
    return path


def use_open_file_dialog(title: str = 'Choose file', start_dir: Union[str, None] = None, exts: List[str] = None):
    """
    Use to pass a file path, for opening, into the decorated function using QFileDialog.getOpenFileName

    :param title:       Title of the dialog box
    :param start_dir:   Directory that is first shown in the dialog box.
    :param exts:        List of file extensions to set the filter in the dialog box
    """
    def wrapper(func):

        @wraps(func)
        def fn(self, *args, **kwargs):
            if 'qdialog' in kwargs.keys():
                if not kwargs['qdialog']:
                    func(self, *args, **kwargs)
                    return fn

            if exts is None:
                e = []
            else:
                e = exts

            if isinstance(self, QWidget):
                parent = self
            else:
                parent = None

            path = QFileDialog.getOpenFileName(parent, title, os.environ['HOME'], f'({" ".join(e)})')
            if not path[0]:
                return
            path = path[0]
            func(self, path, *args, **kwargs)
        return fn
    return wrapper


def use_save_file_dialog(title: str = 'Save file', start_dir: Union[str, None] = None, ext: str = None):
    """
    Use to pass a file path, for saving, into the decorated function using QFileDialog.getSaveFileName

    :param title:       Title of the dialog box
    :param start_dir:   Directory that is first shown in the dialog box.
    :param exts:        List of file extensions to set the filter in the dialog box
    """
    def wrapper(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            if ext is None:
                raise ValueError('Must specify extension')
            if ext.startswith('*'):
                ex = ext[1:]
            else:
                ex = ext

            if isinstance(self, QWidget):
                parent = self
            else:
                parent = None

            path = QFileDialog.getSaveFileName(parent, title, start_dir, f'(*{ex})')
            if not path[0]:
                return
            path = path[0]
            if not path.endswith(ex):
                path = f'{path}{ex}'

            path = validate_path(path)

            func(self, path, *args, **kwargs)

        return fn
    return wrapper


def use_open_dir_dialog(title: str = 'Open directory', start_dir: Union[str, None] = None):
    """
    Use to pass a dir path, to open, into the decorated function using QFileDialog.getExistingDirectory
    :param title:       Title of the dialog box
    :param start_dir:   Directory that is first shown in the dialog box.
    Example:
    .. code-block:: python
        @use_open_dir_dialog('Select Project Directory', '')
        def load_data(self, path, *args, **kwargs):
            my_func_to_do_stuff_and_load_data(path)
    """
    def wrapper(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            if isinstance(self, QWidget):
                parent = self
            else:
                parent = None

            path = QFileDialog.getExistingDirectory(parent, title)
            if not path:
                return
            func(self, path, *args, **kwargs)
        return fn
    return wrapper


def present_exceptions(title: str = 'error', msg: str = 'The following error occurred.'):
    """
    Use to catch exceptions and present them to the user in a QMessageBox warning dialog.
    The traceback from the exception is also shown.

    This decorator can be stacked on top of other decorators.

    Example:

    .. code-block: python

            @present_exceptions('Error loading file')
            @use_open_file_dialog('Choose file')
                def select_file(self, path: str, *args):
                    pass


    :param title:       Title of the dialog box
    :param msg:         Message to display above the traceback in the dialog box
    :param help_func:   A helper function which is called if the user clicked the "Help" button
    """

    def catcher(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc()

                mb = QMessageBox()
                mb.setIcon(QMessageBox.Warning)
                mb.setWindowTitle(title)
                mb.setText(msg)
                mb.setInformativeText(f"{e.__class__.__name__}: {e}")
                mb.setDetailedText(tb)
                mb.setStandardButtons(
                    QMessageBox.Ok | QMessageBox.Help
                )


                # getLogger().info(
                #     f"{e.__class__.__name__}: {e}\n"
                #     f"{traceback.format_exc()}"
                # )

        return fn

    return catcher


def auto_colormap(
        n_colors: int,
        cmap: str = 'hsv',
        output: str = 'mpl',
        spacing: str = 'uniform',
        alpha: float = 1.0
    ) \
        -> List[Union[QtGui.QColor, np.ndarray, str]]:
    """
    If non-qualitative map: returns list of colors evenly spread through the chosen colormap.
    If qualitative map: returns subsequent colors from the chosen colormap

    :param n_colors: Numbers of colors to return
    :param cmap:     name of colormap

    :param output:   option: 'mpl' returns RGBA values between 0-1 which matplotlib likes,
                     option: 'bokeh' returns hex strings that correspond to the RGBA values which bokeh likes

    :param spacing:  option: 'uniform' returns evenly spaced colors across the entire cmap range
                     option: 'subsequent' returns subsequent colors from the cmap

    :param alpha:    alpha level, 0.0 - 1.0

    :return:         List of colors as either ``QColor``, ``numpy.ndarray``, or hex ``str`` with length ``n_colors``
    """

    valid = ['mpl', 'pyqt', 'bokeh']
    if output not in valid:
        raise ValueError(f'output must be one {valid}')

    valid = ['uniform', 'subsequent']
    if spacing not in valid:
        raise ValueError(f'spacing must be one of either {valid}')

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be within 0.0 and 1.0')

    cm = matplotlib_color_map.get_cmap(cmap)
    cm._init()

    if output == 'pyqt':
        lut = (cm._lut * 255).view(np.ndarray)
    else:
        lut = (cm._lut).view(np.ndarray)

    lut[:, 3] *= alpha

    if spacing == 'uniform':
        if not cmap in qualitative_colormaps:
            if cmap =='hsv':
                cm_ixs = np.linspace(30, 210, n_colors, dtype=int)
            else:
                cm_ixs = np.linspace(0, 210, n_colors, dtype=int)
        else:
            if n_colors > len(lut):
                raise ValueError('Too many colors requested for the chosen cmap')
            cm_ixs = np.arange(0, len(lut), dtype=int)
    else:
        cm_ixs = range(n_colors)

    colors = []
    for ix in range(n_colors):
        c = lut[cm_ixs[ix]]

        if output == 'bokeh':
            c = tuple(c[:3] * 255)
            hc = '#%02x%02x%02x' % tuple(map(int, c))
            colors.append(hc)

        else:  # mpl
            colors.append(c)

    return colors


def make_runfile(module_path: str, args_str: Optional[str] = None, filename: Optional[str] = None) -> str:
    """
    Make an executable bash script. Used for running python scripts in external processes.

    :param module_path: absolute module path
    :type module_path:  str

    :param args_str:    str of args that is directly passed with the python command in the bash script
    :type args_str:     str

    :param savedir:     working directory
    :type savedir:      Optional[str]

    :param filename:    optional, specific filename for the script
    :type filename:     Optional[str]

    :param pre_run:     optional, str to run before module is ran
    :type pre_run:      Optional[str]

    :param post_run:    optional, str to run after module has run
    :type post_run:     Optional[str]

    :return: path to the shell script that can be run
    :rtype:  str
    """

    if filename is None:
        if IS_WINDOWS:
            sh_file = os.path.join(os.environ[HOME], 'run.ps1')
        else:
            sh_file = os.path.join(os.environ[HOME], 'run.sh')
    else:
        if IS_WINDOWS:
            if not filename.endswith('.ps1'):
                filename = filename + '.ps1'

    sh_file = filename

    if args_str is None:
        args_str = ''

    if not IS_WINDOWS:
        with open(sh_file, 'w') as f:
            if 'CONDA_PREFIX' in os.environ.keys():
                f.write(
                    f'#!/bin/bash\n'
                    f'export CONDA_PREFIX={os.environ["CONDA_PREFIX"]}\n'
                    f'export CONDA_PYTHON_EXE={os.environ["CONDA_PYTHON_EXE"]}\n'
                    f'export CONDA_PREFIX_1={os.environ["CONDA_PREFIX_1"]}\n'
                )

            elif 'VIRTUAL_ENV' in os.environ.keys():
                f.write(
                    f'#!/bin/bash\n'
                    f'export PATH={os.environ["PATH"]}\n'
                    f'export VIRTUAL_ENV={os.environ["VIRTUAL_ENV"]}\n'
                    f'export LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]}\n'
                )

            if 'PYTHONPATH' in os.environ.keys():
                f.write(f'export PYTHONPATH={os.environ["PYTHONPATH"]}\n')

            # for k, v in os.environ.items():  # copy the current environment
            #     if '\n' in v:
            #         continue
            #
                # f.write(f'export {k}="{v}"\n')

            # User-setable n-processes
            if 'MESMERIZE_N_PROCESSES' in os.environ.keys():
                f.write(f'export MESMERIZE_N_PROCESSES={os.environ["MESMERIZE_N_PROCESSES"]}\n')

            f.write(
                f'export OPENBLAS_NUM_THREADS=1\n'
                f'export MKL_NUM_THREADS=1\n'
            )

            f.write(f'python {module_path} {args_str}')  # call the script to run

    else:
        with open(sh_file, 'w') as f:
            for k, v in os.environ.items():  # copy the current environment
                f.write(f'$env:{k}="{v}";\n')

            f.write(f'python {module_path} {args_str}')

    st = os.stat(sh_file)
    os.chmod(sh_file, st.st_mode | S_IEXEC)

    return sh_file


def _organize_coordinates(contour: dict):
    coors = contour['coordinates']
    coors = coors[~np.isnan(coors).any(axis=1)]

    return coors

