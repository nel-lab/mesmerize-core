from .batch_utils import (
    set_parent_raw_data_path,
    get_parent_raw_data_path,
    load_batch,
    create_batch,
    save_results_safely
)
from .caiman_extensions import *
from pathlib import Path


with open(Path(__file__).parent.joinpath("VERSION"), "r") as f:
    __version__ = f.read().split("\n")[0]

__all__ = [
    "set_parent_raw_data_path",
    "get_parent_raw_data_path",
    "load_batch",
    "create_batch",
    "save_results_safely",
    "CaimanDataFrameExtensions",
    "CaimanSeriesExtensions",
    "CNMFExtensions",
    "MCorrExtensions",
]
