import numpy as np

from coordination.callback.callback import Callback
from coordination.model.pgm import PGM


class EarlyStoppingCallback(Callback):

    def __init__(self, monitor: str = "nll", patience: int = 5):
        # For now, NLL is the only measure we compute to test improvement.
        assert monitor == "nll"

        self.monitor = monitor
        self.patience = patience

        self.num_no_improvement_iter = 0
        self.last_measurement = np.inf

    def reset(self):
        self.num_no_improvement_iter = 0
        self.last_measurement = np.inf

    def check(self, model: PGM):
        current_measurement = None
        if self.monitor == "nll":
            current_measurement = model.nll_[-1]

        if current_measurement >= self.last_measurement:
            self.num_no_improvement_iter += 1

            if self.patience == self.num_no_improvement_iter:
                model.train = False

        self.last_measurement = current_measurement
