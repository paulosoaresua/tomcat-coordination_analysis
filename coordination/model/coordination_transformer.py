from typing import Any

import numpy as np

from coordination.common.dataset import EvidenceDataset
from coordination.model.coordination_model import CoordinationModel
from sklearn.base import BaseEstimator, TransformerMixin


class CoordinationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, coordination_model: CoordinationModel):
        self.coordination_model = coordination_model
        self.estimates_ = []
        self.output_ = None

    def fit(self, X: EvidenceDataset, y: Any = None, *args, **kwargs):
        self.coordination_model.fit(X, *args, **kwargs)
        return self

    def transform(self, X: EvidenceDataset, y: Any = None, *args, **kwargs) -> Any:
        # We keep estimated coordination so it can be retrieved for later assessment
        self.estimates_ = self.coordination_model.predict(X, *args, **kwargs)
        # We pass to the next step in the pipeline the mean coordination over an entire trial
        # for all trials in the dataset
        self.output_ = np.array([np.mean(coordination_estimates[0]) for coordination_estimates in self.estimates_])[:,
                      np.newaxis]

        return self.output_

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self.coordination_model, key):
                setattr(self.coordination_model, key, value)

        return self
