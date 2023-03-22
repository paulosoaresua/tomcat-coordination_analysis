from __future__ import annotations

from typing import Any, Optional

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
        coordination = idata.posterior["coordination"]

        return cls(unbounded_coordination, coordination)

    def plot(self, ax: Any, show_samples: bool = False, color: str = "tab:blue", line_width: int = 1,
             marker_size: Optional[int] = None):
        T = self.coordination.sizes["coordination_time"]
        C = self.coordination.sizes["chain"]
        N = self.coordination.sizes["draw"]
        stacked_coordination_samples = self.coordination.stack(chain_plus_draw=("chain", "draw"))

        mean_coordination = self.coordination.mean(dim=["chain", "draw"])
        sd_coordination = self.coordination.std(dim=["chain", "draw"])
        lower_band = np.maximum(mean_coordination - sd_coordination, 0)
        upper_band = np.minimum(mean_coordination + sd_coordination, 1)

        if show_samples:
            ax.plot(np.arange(T)[:, None].repeat(N * C, axis=1), stacked_coordination_samples, color="tab:pink",
                    alpha=0.3, zorder=1)
        ax.fill_between(np.arange(T), lower_band, upper_band, color=color, alpha=0.5, zorder=2)
        if marker_size is None:
            ax.plot(np.arange(T), mean_coordination, color=color, linewidth=line_width, linestyle="-", zorder=3)
        else:
            ax.plot(np.arange(T), mean_coordination, color=color, linewidth=line_width, linestyle="-",
                    markersize=marker_size, marker="o", zorder=3)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Coordination")
        ax.set_xlim([-0.5, T + 0.5])
        ax.set_ylim([0, 1.05])
