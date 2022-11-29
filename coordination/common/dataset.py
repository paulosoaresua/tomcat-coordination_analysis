from __future__ import annotations
from typing import Any, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split as sklearn_train_test_split


class EvidenceDataSeries:

    def __init__(self, uuid: str):
        self.uuid = uuid

    @property
    def num_time_steps(self):
        raise NotImplementedError


class EvidenceDataset:

    def __init__(self, series: List[EvidenceDataSeries]):
        self.series = series

    @property
    def num_trials(self):
        return len(self.series)

    @property
    def num_time_steps(self):
        return 0 if len(self.series) == 0 else self.series[0].num_time_steps

    def get_subset(self, indices: List[int]) -> EvidenceDataset:
        return self.__class__([self.series[i] for i in indices])

    @staticmethod
    def merge_list(datasets: List[EvidenceDataset]) -> EvidenceDataset:
        merged_dataset = None
        for dataset in datasets:
            if merged_dataset is None:
                merged_dataset = dataset
            else:
                merged_dataset = EvidenceDataset.merge(merged_dataset, dataset)

        return merged_dataset

    def merge(self, dataset2: EvidenceDataset) -> EvidenceDataset:
        return self.__class__(self.series + dataset2.series)


class IndexToDatasetTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, complete_dataset: EvidenceDataset):
        self.complete_dataset = complete_dataset

    def fit(self, X: np.ndarray, y: Any = None, *args, **kwargs):
        # Do nothing
        return self

    def transform(self, X: np.ndarray, y: Any = None, *args, **kwargs) -> EvidenceDataset:
        return self.complete_dataset.get_subset(X.flatten().tolist())


def train_test_split(X: EvidenceDataset, y: np.ndarray, test_size: float = 0.2, seed: int = 0):
    train_indices, test_indices, train_y, test_y = sklearn_train_test_split(np.arange(X.num_trials), y,
                                                                            test_size=test_size, random_state=seed)

    train_X = X.get_subset(train_indices)
    test_X = X.get_subset(test_indices)

    return train_X, test_X, train_y, test_y
