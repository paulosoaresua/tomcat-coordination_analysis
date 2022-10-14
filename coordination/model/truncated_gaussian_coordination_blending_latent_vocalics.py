import numpy as np
from scipy.stats import truncnorm

from coordination.common.dataset import Dataset
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingInferenceLatentVocalics, \
    LatentVocalicsParticles

MIN_VALUE = 0
MAX_VALUE = 1


class TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(CoordinationBlendingInferenceLatentVocalics):

    def __init__(self,
                 mean_prior_coordination: float,
                 std_prior_coordination: float,
                 std_coordination_drifting: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting

    def fit(self, input_features: Dataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        raise NotImplementedError

    def _sample_coordination_from_prior(self, new_particles: LatentVocalicsParticles):
        mean = np.ones(self.num_particles) * self._mean_prior_coordination
        a = (MIN_VALUE - mean) / self._std_prior_coordination
        b = (MAX_VALUE - mean) / self._std_prior_coordination
        new_particles.coordination = truncnorm(loc=mean, scale=self._std_prior_coordination, a=a, b=b).rvs()

    def _sample_coordination_from_transition(self, previous_particles: LatentVocalicsParticles,
                                             new_particles: LatentVocalicsParticles):
        a = (MIN_VALUE - previous_particles.coordination) / self._std_coordination_drifting
        b = (MAX_VALUE - previous_particles.coordination) / self._std_coordination_drifting
        new_particles.coordination = truncnorm(loc=previous_particles.coordination,
                                               scale=self._std_coordination_drifting, a=a,
                                               b=b).rvs()
