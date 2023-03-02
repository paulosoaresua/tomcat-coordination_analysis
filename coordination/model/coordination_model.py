from __future__ import annotations

from typing import Any

import xarray

from coordination.common.functions import sigmoid


"""
Generic classes used by any model of coordination
"""


class CoordinationPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination

    @classmethod
    def from_inference_data(cls, idata: Any) -> CoordinationPosteriorSamples:
        unbounded_coordination = idata.posterior["unbounded_coordination"]
        coordination = sigmoid(unbounded_coordination)

        return cls(unbounded_coordination, coordination)
