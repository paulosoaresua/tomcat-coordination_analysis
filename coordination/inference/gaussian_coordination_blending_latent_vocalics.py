import numpy as np
from scipy.stats import norm

from coordination.inference.coordination_blending_latent_vocalics import CoordinationBlendingInferenceLatentVocalics
from coordination.inference.inference_engine import InferenceEngine
from coordination.inference.particle_filter import ParticleFilter


class GaussianCoordinationBlendingInferenceLatentVocalics(CoordinationBlendingInferenceLatentVocalics):

    def __init__(self, mean_prior_coordination: float, std_prior_coordination: float, std_coordination_drifting: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting

    def _sample_coordination_from_prior(self) -> np.ndarray:
        mean = np.ones(self.num_particles) * self._mean_prior_coordination
        return norm(loc=mean, scale=self._std_prior_coordination).rvs()

    def _sample_coordination_from_transition(self, previous_coordination_particles: np.ndarray):
        return norm(loc=previous_coordination_particles, scale=self._std_coordination_drifting).rvs()

    def _transform_coordination(self, coordination_particles: np.ndarray) -> np.ndarray:
        return np.clip(coordination_particles, a_min=0, a_max=1)
