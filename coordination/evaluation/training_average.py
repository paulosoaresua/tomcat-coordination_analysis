from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class TrainingAverageModel(BaseEstimator):
    """
    This is a naive model that uses the average of the target training samples as prediction over
    the test samples.
    """

    def __init__(self):
        """
        Creates a training average model.
        """
        self._mean_y = None

    def fit(self, X: np.ndarray, y: Any = None) -> TrainingAverageModel:
        """
        Fits the model by storing the mean of the target values.

        @param X: input features per samples.
        @param y: target values per samples.
        @return: model.
        """
        self._mean_y = np.mean(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the mean of the targets used during fit as prediction regardless of the input
        features.

        @param X: input features per sample.
        @return:
        """
        return np.full(X.shape[0], fill_value=self._mean_y)
