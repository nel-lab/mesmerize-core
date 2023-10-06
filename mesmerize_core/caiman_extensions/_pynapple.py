import numpy as np
import pandas as pd
import pynapple as nap
from caiman.source_extraction.cnmf import CNMF


from ._utils import validate
from .cnmf import CNMFExtensions


@pd.api.extensions.register_series_accessor("nap")
class PynappleExtension:
    def __init__(self, series: pd.Series):
        self._series = series

    @validate("cnmf")
    def get_tsd_frame(self):
        cnmf_obj: CNMF = self._series.cnmf.get_output()

        framerate = cnmf_obj.params.data["fr"]

        n_frames = cnmf_obj.estimates.C[1]

        duration_seconds = n_frames / framerate

        timestamps = np.linspace(0, duration_seconds, n_frames)

        n_components = cnmf_obj.estimates.C.shape[0]

        tsdframe = nap.TsdFrame(
            t=timestamps,
            d=cnmf_obj.estimates.C.T,
            time_units="s",
            columns=list(map(range(n_components)))
        )

        return tsdframe
