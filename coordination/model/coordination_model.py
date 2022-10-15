from typing import Any

from sklearn.base import BaseEstimator
from torch.utils.tensorboard import SummaryWriter

from coordination.common.dataset import Dataset


class CoordinationModel(BaseEstimator):

    tb_writer: SummaryWriter

    def fit(self, input_features: Dataset, *args, **kwargs):
        raise NotImplementedError

    def predict(self, input_features: Dataset, *args, **kwargs) -> Any:
        raise NotImplementedError

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self

    def configure_tensorboard(self, out_dir: str):
        self.tb_writer = SummaryWriter(log_dir=out_dir)
