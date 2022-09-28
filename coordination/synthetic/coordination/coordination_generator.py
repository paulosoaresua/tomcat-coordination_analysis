from typing import Optional, Union

from scipy.stats import bernoulli, truncnorm
import numpy as np
import random


class CoordinationGenerator:
    """
    This class generates synthetic values for a latent coordination variable over time.
    """

    def __init__(self, num_time_steps: int):
        self._num_time_steps = num_time_steps

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        cs = []
        for t in range(self._num_time_steps):
            if t == 0:
                cs.append(self._sample_from_prior())
            else:
                cs.append(self._sample_from_transition(cs[t - 1]))

        return np.array(cs)

    def _sample_from_prior(self) -> float:
        raise Exception("Not implemented in this class.")

    def _sample_from_transition(self, previous_coordination: Union[float, np.ndarray]) -> float:
        raise Exception("Not implemented in this class.")
