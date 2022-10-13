import numpy as np
from scipy.stats import truncnorm

from coordination.inference.coordination_blending_latent_vocalics import CoordinationBlendingInferenceLatentVocalics

MIN_VALUE = 0
MAX_VALUE = 1


class TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(CoordinationBlendingInferenceLatentVocalics):

    def __init__(self, mean_prior_coordination: float, std_prior_coordination: float, std_coordination_drifting: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting

    def _sample_coordination_from_prior(self) -> np.ndarray:
        mean = np.ones(self.num_particles) * self._mean_prior_coordination
        a = (MIN_VALUE - mean) / self._std_prior_coordination
        b = (MAX_VALUE - mean) / self._std_prior_coordination
        return truncnorm(loc=mean, scale=self._std_prior_coordination, a=a, b=b).rvs()

    def _sample_coordination_from_transition(self, previous_coordination_particles: np.ndarray):
        a = (MIN_VALUE - previous_coordination_particles) / self._std_coordination_drifting
        b = (MAX_VALUE - previous_coordination_particles) / self._std_coordination_drifting
        return truncnorm(loc=previous_coordination_particles, scale=self._std_coordination_drifting, a=a,
                         b=b).rvs()
