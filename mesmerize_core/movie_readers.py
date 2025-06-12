import tifffile
from pathlib import Path
import numpy as np
from caiman import load_memmap
import h5py
import zarr

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

    
    if ext in ['.hdf5', '.h5', '.nwb', '.mat', '.n5', '.zarr']:
        return hdf5_reader(path, **kwargs)

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


def hdf5_reader(path: str, var_name_hdf5='mov'):
    # based on caiman.base.movies.load_iter
    extension = Path(path).suffix
    if extension in ['.n5', '.zarr']:  # Thankfully, the zarr library lines up closely with h5py past the initial open
        f = zarr.open(path, "r")
        if isinstance(f, zarr.Array):
            raise RuntimeError('Expected a zarr Group, not an Array')
    else:
        try:
            f = h5py.File(path, "r")
        except:
            if extension == '.mat':
                raise Exception(f"Problem loading {path}: Unknown format. This may be in the original version 1 (non-hdf5) mat format; please convert it first")
            else:
                raise Exception(f"Problem loading {path}: Unknown format.")
    
    ignore_keys = ['__DATA_TYPES__'] # Known metadata that tools provide, add to this as needed.
    fkeys = list(filter(lambda x: x not in ignore_keys, f.keys()))
    if len(fkeys) == 1: # If the hdf5 file we're parsing has only one dataset inside it,
                        # ignore the arg and pick that dataset
        var_name_hdf5 = fkeys[0]
    Y = f.get('acquisition/' + var_name_hdf5 + '/data'
            if extension == '.nwb' else var_name_hdf5)
    return Y