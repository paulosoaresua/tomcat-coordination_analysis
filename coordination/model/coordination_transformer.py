from typing import Any

import numpy as np

from coordination.common.dataset import Dataset
from coordination.model.coordination_model import CoordinationModel
from sklearn.base import BaseEstimator, TransformerMixin


class CoordinationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, coordination_model: CoordinationModel, num_particles: int = 10000):
        self.coordination_model = coordination_model
        self.num_particles = num_particles
        self.estimates = []

    def fit(self, X: Dataset, y: Any = None, *args, **kwargs):
        self.coordination_model.fit(X, *args, **kwargs)
        return self

    def transform(self, X: Dataset, y: Any = None, *args, **kwargs) -> Any:
        # We keep estimated coordination so it can be retrieved for later assessment
        self.estimates = self.coordination_model.predict(X, self.num_particles, *args, **kwargs)
        # We pass to the next step in the pipeline the mean coordination over an entire trial
        # for all trials in the dataset
        return np.array([np.mean(coordinations[0]) for coordinations in self.estimates])[:, np.newaxis]

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self
