from typing import List

from datetime import datetime
import numpy as np


class VocalicsSeries:

    def __init__(self, values: np.ndarray, timestamps: List[datetime]):
        # A matrix containing values per each vocalic feature over time.
        # Each row contains the series for a different feature.
        self.values = values
        self.timestamps = timestamps

    @property
    def num_series(self):
        """
        Number of vocalic features
        """
        if len(self.values) == 0:
            return 0

        return self.values.shape[0]

    @property
    def size(self):
        """
        Number of vocalic feature values
        """
        if len(self.values) == 0:
            return 0

        return self.values.shape[1]
