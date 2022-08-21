from typing import List, Optional

from datetime import datetime

import numpy as np


class SparseSeries:
    def __init__(self, values: np.ndarray, mask: np.ndarray, timestamps: Optional[List[Optional[datetime]]] = None):
        self.values = values
        self.mask = mask
        self.timestamps = timestamps

    @property
    def num_series(self):
        if len(self.values) == 0:
            return 0

        return self.values.shape[0]

    @property
    def num_time_steps(self):
        if len(self.values) == 0:
            return 0

        return self.values.shape[1]

    def normalize(self):
        """
        Make values of series have mean 0 and standard deviation 1
        """
        self.values = self.values.astype(float)
        valid_indices = [t for t, mask in enumerate(self.mask) if mask == 1]
        mean = self.values[:, valid_indices].mean(axis=1)[:, np.newaxis]
        std = self.values[:, valid_indices].std(axis=1)[:, np.newaxis]
        self.values[:, valid_indices] = (self.values[:, valid_indices] - mean) / std
