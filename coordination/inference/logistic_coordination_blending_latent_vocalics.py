from typing import List

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from coordination.common.dataset import Dataset
from coordination.inference.gaussian_coordination_blending_latent_vocalics import \
    GaussianCoordinationBlendingInferenceLatentVocalics, LatentVocalicsParticles


class LogisticLatentVocalicsParticles(LatentVocalicsParticles):
    coordination_logit: np.ndarray

    def _keep_particles_at(self, indices: np.ndarray):
        super()._keep_particles_at(indices)
        self.coordination_logit = self.coordination_logit[indices]

    def squeeze(self):
        self.coordination = expit(self.coordination_logit)


class LogisticCoordinationBlendingInferenceLatentVocalics(GaussianCoordinationBlendingInferenceLatentVocalics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.states: List[LogisticLatentVocalicsParticles] = []

    def fit(self, input_features: Dataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        raise NotImplementedError

    def _sample_coordination_from_prior(self, new_particles: LogisticLatentVocalicsParticles):
        mean = np.ones(self.num_particles) * self._mean_prior_coordination
        new_particles.coordination_logit = norm(loc=mean, scale=self._std_prior_coordination).rvs()
        new_particles.squeeze()

    def _sample_coordination_from_transition(self, previous_particles: LogisticLatentVocalicsParticles,
                                             new_particles: LogisticLatentVocalicsParticles):
        new_particles.coordination_logit = norm(loc=previous_particles.coordination_logit,
                                                scale=self._std_coordination_drifting).rvs()
        new_particles.squeeze()

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return LogisticLatentVocalicsParticles()
