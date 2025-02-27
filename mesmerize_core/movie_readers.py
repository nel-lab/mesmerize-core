import tifffile
from pathlib import Path
import numpy as np
from caiman import load_memmap

from .utils import warning_experimental
from .arrays import LazyTiff

try:
    import pims

    HAS_PIMS = True
except (ModuleNotFoundError, ImportError):
    HAS_PIMS = False


def default_reader(path: str, **kwargs):
    ext = Path(path).suffixes[-1]
    if ext in [".tiff", ".tif", ".btf"]:
        try:
            movie = tiff_memmap_reader(path, **kwargs)
        except:  # if file is not memmapable
            movie = tiff_lazyarray(path, **kwargs)

        return movie

    if ext in [".mmap", ".memmap"]:
        return caiman_memmap_reader(path, **kwargs)

    else:
        raise ValueError(f"No default movie reader for given file extension: '{ext}'")


def tiff_memmap_reader(path: str, **kwargs) -> np.memmap:
    return tifffile.memmap(path, **kwargs)


@warning_experimental("This feature is new and might change in the future")
def tiff_lazyarray(path: str) -> LazyTiff:
    # random access speed on a magnetic HDD is ~30Hz for simultaneously slicing of 20 frames from Teena's tiff files
    # much slower than tifffile.memmap but this is just a last resort anyways
    return LazyTiff(path)


def caiman_memmap_reader(path: str, **kwargs) -> np.memmap:
    Yr, dims, T = load_memmap(path, **kwargs)
    return np.reshape(Yr.T, [T] + list(dims), order="F")


def pims_reader(path: str, **kwargs):
    if not HAS_PIMS:
        raise ModuleNotFoundError("you must install `pims` to use the pims reader")
    return pims.open(path, **kwargs)
