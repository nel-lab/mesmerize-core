import tifffile
from pathlib import Path
import numpy as np
from caiman import load_memmap

try:
    import pims
    HAS_PIMS = True
except (ModuleNotFoundError, ImportError):
    HAS_PIMS = False


def default_reader(path: str):
    ext = Path(path).suffixes[-1]
    if ext in [".tiff", ".tif", ".btf"]:
        return tiff_memmap_reader(path)

    if ext in [".mmap", ".memmap"]:
        return caiman_memmap_reader(path)

    else:
        raise ValueError(
            f"No default movie reader for given file extension: '{ext}'"
        )


def tiff_memmap_reader(path: str) -> np.memmap:
    return tifffile.memmap(path)


def caiman_memmap_reader(path: str) -> np.memmap:
    Yr, dims, T = load_memmap(path)
    return np.reshape(Yr.T, [T] + list(dims), order="F")


def pims_reader(path: str):
    if not HAS_PIMS:
        raise ModuleNotFoundError(
            "you must install `pims` to use the pims reader"
        )
    return pims.open(path)
