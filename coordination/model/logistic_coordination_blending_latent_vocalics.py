from typing import Callable, List, Optional

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from coordination.common.dataset import InputFeaturesDataset
from coordination.model.gaussian_coordination_blending_latent_vocalics import \
    GaussianCoordinationBlendingInferenceLatentVocalics, LatentVocalicsParticles


class LogisticLatentVocalicsParticles(LatentVocalicsParticles):
    coordination_logit: np.ndarray

    def _keep_particles_at(self, indices: np.ndarray):
        super()._keep_particles_at(indices)
        self.coordination_logit = self.coordination_logit[indices]

    def squeeze(self):
        self.coordination = expit(self.coordination_logit)


class LogisticCoordinationBlendingInferenceLatentVocalics(GaussianCoordinationBlendingInferenceLatentVocalics):

    def __init__(self,
                 mean_prior_coordination: float,
                 std_prior_coordination: float,
                 std_coordination_drifting: float,
                 mean_prior_latent_vocalics: np.array,
                 std_prior_latent_vocalics: np.array,
                 std_coordinated_latent_vocalics: np.ndarray,
                 std_observed_vocalics: np.ndarray,
                 f: Callable = lambda x, s: x,
                 g: Callable = lambda x: x,
                 fix_coordination_on_second_half: bool = True,
                 num_particles: int = 10000,
                 seed: Optional[int] = None):
        super().__init__(mean_prior_coordination, std_prior_coordination, std_coordination_drifting,
                         mean_prior_latent_vocalics, std_prior_latent_vocalics, std_coordinated_latent_vocalics,
                         std_observed_vocalics, f, g, fix_coordination_on_second_half, num_particles, seed)

        self.states: List[LogisticLatentVocalicsParticles] = []

    def fit(self, input_features: InputFeaturesDataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        return self

    def _sample_coordination_from_prior(self, new_particles: LogisticLatentVocalicsParticles):
        mean = np.ones(self.num_particles) * self.mean_prior_coordination
        new_particles.coordination_logit = norm(loc=mean, scale=self.std_prior_coordination).rvs()
        new_particles.squeeze()

    def _sample_coordination_from_transition(self, previous_particles: LogisticLatentVocalicsParticles,
                                             new_particles: LogisticLatentVocalicsParticles):
        new_particles.coordination_logit = norm(loc=previous_particles.coordination_logit,
                                                scale=self.std_coordination_drifting).rvs()
        new_particles.squeeze()

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return LogisticLatentVocalicsParticles()
