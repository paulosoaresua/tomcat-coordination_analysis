from typing import List

from datetime import datetime
import numpy as np

"""
This class is on its own file because Vocalics depends on it and VocalicsReader too. 
And Vocalics depends on VocalicsReader. Therefore, adding it to Vocalics would cause circular
dependency.
"""


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
        Number of values per feature
        """
        if len(self.values) == 0:
            return 0

        return self.values.shape[1]
