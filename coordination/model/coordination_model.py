from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from coordination.common.log import image_to_tensorboard
from sklearn.base import BaseEstimator
from torch.utils.tensorboard import SummaryWriter

from coordination.common.dataset import EvidenceDataset, SeriesData
from coordination.plot.coordination import add_discrete_coordination_bar


class CoordinationModel(BaseEstimator):

    def __init__(self):
        self.tb_writer: Optional[SummaryWriter] = None

    def fit(self, input_features: EvidenceDataset, *args, **kwargs):
        raise NotImplementedError

    def predict(self, input_features: EvidenceDataset, *args, **kwargs) -> Any:
        raise NotImplementedError

    def configure_tensorboard(self, out_dir: str):
        self.tb_writer = SummaryWriter(log_dir=out_dir)

    def log_coordination_inference_plot(self, series: SeriesData, params: np.ndarray, series_id: str):
        mean_cs = params[0]
        var_cs = params[1]
        time_steps = len(mean_cs)

        fig = plt.figure(figsize=(20, 6))

        xs = range(time_steps)
        plt.plot(xs, mean_cs, marker="o", color="tab:orange", linestyle="--", markersize=3, linewidth=0.5)
        lb = np.clip(mean_cs - np.sqrt(var_cs), a_min=0, a_max=1)
        ub = np.clip(mean_cs + np.sqrt(var_cs), a_min=0, a_max=1)
        plt.fill_between(xs, lb, ub, color='tab:orange', alpha=0.2)

        times, masks = list(
            zip(*[(t, mask + 0.02) for t, mask in enumerate(series.vocalics.mask) if mask > 0 and t < time_steps]))
        plt.scatter(times, masks, color="tab:green", marker="+")
        plt.xlabel("Time Steps (seconds)")
        plt.ylabel("Coordination")
        plt.ylim([-0.05, 1.05])
        plt.title(f"Coordination Inference - {series_id}", fontsize=14, weight="bold")

        add_discrete_coordination_bar(main_ax=fig.gca(),
                                      coordination_series=[np.where(mean_cs > 0.5, 1, 0)],
                                      coordination_colors=["tab:orange"],
                                      labels=["Coordination"])

        image = image_to_tensorboard(fig)
        self.tb_writer.add_image(f"coordination-{series_id}", image)



