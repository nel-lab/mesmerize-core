from typing import *
from pathlib import Path
from warnings import warn

import numpy as np

try:
    from decord import VideoReader
except ImportError:
    HAS_DECORD = False
else:
    HAS_DECORD = True

from ._base import LazyArray


class LazyVideo(LazyArray):
    def __init__(
        self,
        path: Union[Path, str],
        min_max: Tuple[int, int] = None,
        as_grayscale: bool = False,
        rgb_weights: Tuple[float, float, float] = (0.299, 0.587, 0.114),
        **kwargs,
    ):
        """
        LazyVideo reader, basically just a wrapper for ``decord.VideoReader``.
        Should support opening anything that decord can open.

        **Important:** requires ``decord`` to be installed: https://github.com/dmlc/decord

        Parameters
        ----------
        path: Path or str
            path to video file

        min_max: Tuple[int, int], optional
            min and max vals of the entire video, uses min and max of 10th frame if not provided

        as_grayscale: bool, optional
            return grayscale frames upon slicing

        rgb_weights: Tuple[float, float, float], optional
            (r, g, b) weights used for grayscale conversion if ``as_graycale`` is ``True``.
            default is (0.299, 0.587, 0.114)

        kwargs
            passed to ``decord.VideoReader``

        Examples
        --------

        Lazy loading with CPU

        .. code-block:: python

            from mesmerize_core.arrays import LazyVideo

            vid = LazyVideo("path/to/video.mp4")

            # use fpl to visualize

            import fastplotlib as fpl

            iw = fpl.ImageWidget(vid)
            iw.show()


        Lazy loading with GPU, decord must be compiled with CUDA options to use this

        .. code-block:: python

            from decord import gpu
            from mesmerize_core.arrays import LazyVideo

            gpu_context = gpu(0)

            vid = LazyVideo("path/to/video.mp4", ctx=gpu_context)

        """
        if not HAS_DECORD:
            raise ImportError("You must install `decord` to use LazyVideo")

        self._video_reader = VideoReader(str(path), **kwargs)

        try:
            frame0 = self._video_reader[10].asnumpy()
        except IndexError:
            frame0 = self._video_reader[0].asnumpy()

        self._shape = (self._video_reader._num_frame, *frame0.shape[:-1])

        if len(frame0.shape) > 2:
            # we assume the shape of a frame is [x, y, RGB]
            self._is_color = True
        else:
            # we assume is already grayscale
            self._is_color = False

        self._dtype = frame0.dtype

        if min_max is not None:
            self._min, self._max = min_max
        else:
            self._min = frame0.min()
            self._max = frame0.max()

        self.as_grayscale = as_grayscale
        self.rgb_weights = rgb_weights

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        """[n_frames, x, y], RGB color dim not included in shape"""
        return self._shape

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def min(self) -> float:
        warn("min not implemented for LazyTiff, returning min of 0th index")
        return self._min

    @property
    def max(self) -> float:
        warn("max not implemented for LazyTiff, returning min of 0th index")
        return self._max

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        if not self.as_grayscale:
            return self._video_reader[indices].asnumpy()

        if self._is_color:
            a = self._video_reader[indices].asnumpy()

            # R + G + B -> grayscale
            gray = (
                a[..., 0] * self.rgb_weights[0]
                + a[..., 1] * self.rgb_weights[1]
                + a[..., 2] * self.rgb_weights[2]
            )

            return gray

        warn("Video is already grayscale, just returning")
        return self._video_reader[indices].asnumpy()
