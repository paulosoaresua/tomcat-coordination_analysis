from typing import Callable, List, Optional

import numpy as np

from coordination.common.dataset import EvidenceDataSeries
from coordination.common.utils import set_seed


class Particles:

    def resample(self, importance_weights: np.ndarray):
        indices = Particles.resample_particle_indices(importance_weights)
        self._keep_particles_at(indices)

    def _keep_particles_at(self, indices: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def resample_particle_indices(importance_weights: np.ndarray) -> np.ndarray:
        num_particles = len(importance_weights)
        return np.random.choice(num_particles, num_particles, replace=True, p=importance_weights)


class ParticleFilter:

    def __init__(self,
                 num_particles: int,
                 resample_at_fn: Callable,
                 sample_from_prior_fn: Callable,
                 sample_from_transition_fn: Callable,
                 calculate_log_likelihood_fn: Callable,
                 seed: Optional[int] = None):
        self.num_particles = num_particles
        self.seed = seed

        # Functions
        self.resample_at_fn = resample_at_fn
        self.sample_from_prior_fn = sample_from_prior_fn
        self.sample_from_transition_fn = sample_from_transition_fn
        self.calculate_log_likelihood_fn = calculate_log_likelihood_fn

        self.states: List[Particles] = []

    def reset_state(self):
        self.states = []
        set_seed(self.seed)

    def next(self, series: EvidenceDataSeries):
        next_time_step = len(self.states)
        self.__transition(series)
        if self.resample_at_fn(next_time_step, series):
            self.__resample(series)

    def __transition(self, series: EvidenceDataSeries):
        if len(self.states) == 0:
            self.states.append(self.sample_from_prior_fn(self.num_particles, series))
        else:
            t = len(self.states)
            self.states.append(self.sample_from_transition_fn(t, self.num_particles, self.states, series))

    def __resample(self, series: EvidenceDataSeries):
        t = len(self.states) - 1
        log_weights = self.calculate_log_likelihood_fn(t, self.states, series)
        log_weights -= np.max(log_weights)
        log_weights = np.exp(log_weights)
        importance_weights = log_weights / np.sum(log_weights)
        self.states[t].resample(importance_weights)
