from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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

    def plot(self, ax: Any, show_samples: bool):
        T = self.coordination.sizes["coordination_time"]
        C = self.coordination.sizes["chain"]
        N = self.coordination.sizes["draw"]
        stacked_coordination_samples = self.coordination.stack(chain_plus_draw=("chain", "draw"))

        mean_coordination = self.coordination.mean(dim=["chain", "draw"])
        sd_coordination = self.coordination.std(dim=["chain", "draw"])
        lower_band = np.maximum(mean_coordination - sd_coordination, 0)
        upper_band = np.minimum(mean_coordination + sd_coordination, 1)

        if show_samples:
            ax.plot(np.arange(T)[:, None].repeat(N * C, axis=1), stacked_coordination_samples, color="tab:blue",
                    alpha=0.3, zorder=1)
        ax.fill_between(np.arange(T), lower_band, upper_band, color="tab:pink", alpha=0.8, zorder=2)
        ax.plot(np.arange(T), mean_coordination, color="white", alpha=1, marker="o", markersize=5, linewidth=1,
                linestyle="--", zorder=3)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Coordination")
        ax.set_xlim([-0.5, T + 0.5])
        ax.set_ylim([0, 1.05])
