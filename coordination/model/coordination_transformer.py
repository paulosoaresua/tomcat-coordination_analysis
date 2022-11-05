from typing import Any

import numpy as np

from coordination.common.dataset import EvidenceDataset
from coordination.model.pgm import PGM
from sklearn.base import BaseEstimator, TransformerMixin


class CoordinationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model: PGM):
        self.model = model
        self.estimates_ = []
        self.output_ = None

    def fit(self, X: EvidenceDataset, y: Any = None, *args, **kwargs):
        self.model.fit(X, *args, **kwargs)
        return self

    def transform(self, X: EvidenceDataset, y: Any = None, *args, **kwargs) -> Any:
        # We keep estimated coordination so it can be retrieved for later assessment
        self.estimates_ = self.model.predict(X, *args, **kwargs)
        # We pass to the next step in the pipeline the mean coordination over an entire trial
        # for all trials in the dataset
        self.output_ = np.array([np.mean(coordination_estimates[0]) for coordination_estimates in self.estimates_])[:,
                      np.newaxis]

        return self.output_

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)

        return self
