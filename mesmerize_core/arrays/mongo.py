from typing import *
from pathlib import Path
from warnings import warn

import numpy as np

from ._base import LazyArray


class LazyMongo(LazyArray):
    def __init__(self, client: pymongo.MongoClient, collection: str):
        self._collection = getattr(db, collection)
        self._n_frames = db.frames.estimated_document_count()
        self._shape = (self._n_frames, 512, 512)
        self._dtype = np.int16

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def min(self) -> float:
        warn("min not implemented for LazyTiff, returning min of 0th index")
        return self[0].min()

    @property
    def max(self) -> float:
        warn("max not implemented for LazyTiff, returning min of 0th index")
        return self[0].max()

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        index = indices + 1
        _bytes = self._collection.find({"index": float(index)})[0]["frame0"]
        return np.frombuffer(_bytes, dtype=np.int16)[32:].reshape(512, 512)
