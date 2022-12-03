from copy import deepcopy

import numpy as np

from coordination.callback.callback import Callback
from coordination.model.pgm import PGM


class EarlyStoppingCallback(Callback):

    def __init__(self, monitor: str = "nll", patience: int = 5):
        # For now, NLL is the only measure we compute to test improvement.
        assert monitor == "nll"

        self.monitor = monitor
        self.patience = patience

        self._num_no_improvement_iter = 0
        self._best_measurement = np.inf
        self.best_model_ = None

    def reset(self):
        self._num_no_improvement_iter = 0
        self._best_measurement = np.inf

    def check(self, model: PGM, iter: int):
        current_measurement = None
        if self.monitor == "nll":
            current_measurement = model.nll_[iter]

        if current_measurement >= self._best_measurement:
            self._num_no_improvement_iter += 1

            if self.patience == self._num_no_improvement_iter:
                model.train = False
        else:
            self._best_measurement = current_measurement
