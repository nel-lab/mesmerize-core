import pims
import tifffile
from typing import *
from pathlib import Path


class MovieReader:
    def __init__(self):
        self.registry = {
            'append-tiff': AppendTiffReader
        }

    def register(self, func):
        pass

    def get_reader(self, reader: str):
        return self.registry[reader]


class AppendTiffReader:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.tif = tifffile.TiffReader(path)
        self.n_frames = len(self.tif.pages)

    def __getitem__(self, item):
        return self.tif.asarray(key=item)
