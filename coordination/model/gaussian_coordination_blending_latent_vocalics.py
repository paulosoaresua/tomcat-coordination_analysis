from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from coordination.common.dataset import Dataset
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingInferenceLatentVocalics, \
    LatentVocalicsParticles


class GaussianLatentVocalicsParticles(LatentVocalicsParticles):
    latent_vocalics: Dict[str, np.ndarray]
    unbounded_coordination: np.ndarray

    def _keep_particles_at(self, indices: np.ndarray):
        super()._keep_particles_at(indices)
        self.unbounded_coordination = self.unbounded_coordination[indices]

    def clip(self):
        self.coordination = np.clip(self.unbounded_coordination, a_min=0, a_max=1)

    def mean(self):
        return np.clip(self.unbounded_coordination.mean(), a_min=0, a_max=1)

    def var(self):
        return self.unbounded_coordination.var()


class GaussianCoordinationBlendingInferenceLatentVocalics(CoordinationBlendingInferenceLatentVocalics):

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
        super().__init__(mean_prior_latent_vocalics, std_prior_latent_vocalics, std_coordinated_latent_vocalics,
                         std_observed_vocalics, f, g, fix_coordination_on_second_half, num_particles, seed)

        self.mean_prior_coordination = mean_prior_coordination
        self.std_prior_coordination = std_prior_coordination
        self.std_coordination_drifting = std_coordination_drifting

        self.states: List[GaussianLatentVocalicsParticles] = []

    def fit(self, input_features: Dataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        return self

    def _sample_coordination_from_prior(self, new_particles: GaussianLatentVocalicsParticles):
        mean = np.ones(self.num_particles) * self.mean_prior_coordination
        new_particles.unbounded_coordination = norm(loc=mean, scale=self.std_prior_coordination).rvs()
        new_particles.clip()

    def _sample_coordination_from_transition(self, previous_particles: GaussianLatentVocalicsParticles,
                                             new_particles: GaussianLatentVocalicsParticles):
        new_particles.unbounded_coordination = norm(loc=previous_particles.unbounded_coordination,
                                                    scale=self.std_coordination_drifting).rvs()
        new_particles.clip()

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return GaussianLatentVocalicsParticles()
