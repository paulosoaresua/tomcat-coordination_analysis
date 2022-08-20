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
