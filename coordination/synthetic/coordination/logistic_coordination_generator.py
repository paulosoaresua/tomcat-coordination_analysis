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

    def __init__(self, mean_prior_coordination_logit: float, std_prior_coordination_logit: float,
                 std_coordination_logit_drifting: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_coordination_logit = mean_prior_coordination_logit
        self._std_prior_coordination_logit = std_prior_coordination_logit
        self._std_coordination_logit_drifting = std_coordination_logit_drifting

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        cs = super().generate(seed)
        return cs[:, 1]

    def _sample_from_prior(self) -> float:
        v = norm(loc=self._mean_prior_coordination_logit, scale=self._std_prior_coordination_logit).rvs()
        return np.array([v, expit(v)])

    def _sample_from_transition(self, previous_coordination: np.ndarray) -> float:
        d = norm(loc=previous_coordination[0], scale=self._std_coordination_logit_drifting).rvs()
        return np.array([d, expit(d)])
