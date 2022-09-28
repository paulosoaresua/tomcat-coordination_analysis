from scipy.stats import beta

from coordination.synthetic.coordination.coordination_generator import CoordinationGenerator

from scipy.special import expit, logit
from scipy.stats import norm
import numpy as np

from typing import Optional


class LogisticCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a continuous coordination sampled from a beta distribution.
    """

    def __init__(self, transition_std: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._transition_std = transition_std

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        cs = super().generate(seed)
        return cs[:, 1]

    def _sample_from_prior(self) -> float:
        d = -3
        return np.array([d, expit(d)])

    def _sample_from_transition(self, previous_coordination: np.ndarray) -> float:
        d = norm(loc=previous_coordination[0], scale=self._transition_std).rvs()
        return np.array([d, expit(d)])
