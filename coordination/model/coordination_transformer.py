from typing import Any

import numpy as np

from coordination.common.dataset import Dataset
from coordination.model.coordination_model import CoordinationModel
from sklearn.base import BaseEstimator, TransformerMixin


class CoordinationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, coordination_model: CoordinationModel):
        self.coordination_model = coordination_model

    def fit(self, input_features: Dataset, *args, **kwargs):
        self.coordination_model.fit(input_features, *args, **kwargs)
        return self

    def transform(self, input_features: Dataset, *args, **kwargs) -> Any:
        params = self.coordination_model.predict(input_features, *args, **kwargs)
        # We pass to the next step in the pipeline the mean coordination over an entire trial
        # for all trials in the dataset
        return np.array([np.mean(coordinations[0]) for coordinations in params])[:, np.newaxis]

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self
