import numpy as np


class SparseSeries:
    """
    Represents a series of sparse values. The mask attribute has 1 in the time steps for which there are valid values
    in the series, and 0 otherwise.
    """

    def __init__(self, values: np.ndarray, mask: np.ndarray):
        self.values = values
        self.mask = mask

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
        Number of time steps
        """
        if len(self.values) == 0:
            return 0

        return self.values.shape[1]

