from __future__ import annotations
from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from coordination.component.speech.vocalics_component import VocalicsSparseSeries


class SeriesData:

    def __init__(self, vocalics: VocalicsSparseSeries):
        self.vocalics = vocalics

    @property
    def num_time_steps(self):
        return self.vocalics.num_time_steps


class Dataset:

    def __init__(self, series: List[SeriesData], series_ids: Optional[List[str]] = None):
        self.series = series
        self.series_ids = series_ids

    @property
    def num_trials(self):
        return len(self.series)

    def get_id(self, series_idx: int) -> str:
        return self.series_ids[series_idx] if self.series_ids is not None else str(series_idx)

    def get_subset(self, indices: List[int]) -> Dataset:
        series_subset = [self.series[i] for i in indices]
        series_ids_subset = [self.series_ids[i] for i in indices] if self.series_ids is not None else None

        return Dataset(series_subset, series_ids_subset)


class IndexToDatasetTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, complete_dataset: Dataset):
        self.complete_dataset = complete_dataset

    def fit(self, X: np.ndarray, y: Any = None, *args, **kwargs):
        return self

    def transform(self, X: np.ndarray, y: Any = None, *args, **kwargs) -> Dataset:
        return self.complete_dataset.get_subset(X.flatten().tolist())


def train_test_split(X: Dataset, y: np.ndarray, test_size: float = 0.2, seed: int = 0):
    train_indices, test_indices, train_y, test_y = sklearn_train_test_split(np.arange(X.num_trials), y,
                                                                            test_size=test_size, random_state=seed)

    train_X = X.get_subset(train_indices)
    test_X = X.get_subset(test_indices)

    return train_X, test_X, train_y, test_y
