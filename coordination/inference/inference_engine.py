from typing import Any

from coordination.common.dataset import Dataset
from sklearn.base import BaseEstimator


class InferenceEngine(BaseEstimator):

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
