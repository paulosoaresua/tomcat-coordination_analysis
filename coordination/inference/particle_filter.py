from typing import List

import numpy as np


class Particles:

    def __init__(self, coordination: np.ndarray):
        self.coordination = coordination

    def resample(self, importance_weights: np.ndarray):
        num_particles = len(importance_weights)
        new_particles = np.random.choice(num_particles, num_particles, replace=True, p=importance_weights)
        self.coordination = self.coordination[new_particles]


class ParticleFilter:

    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        self.states: List[Particles] = []

    def next(self):
        next_time_step = len(self.states)
        self.__predict()
        if self._resample_at(next_time_step):
            self.__resample()

    def __predict(self):
        if len(self.states) == 0:
            self.states.append(self._sample_from_prior())
        else:
            t = len(self.states)
            self.states.append(self._sample_from_transition_to(t))

    def __resample(self):
        t = len(self.states) - 1
        log_weights = self._calculate_log_likelihood_at(t)
        log_weights -= np.max(log_weights)
        log_weights = np.exp(log_weights)
        importance_weights = log_weights / np.sum(log_weights)
        self.states[t].resample(importance_weights)

    def _resample_at(self, time_step: int):
        return True

    def _sample_from_prior(self) -> Particles:
        raise NotImplementedError

    def _sample_from_transition_to(self, time_step: int) -> Particles:
        raise NotImplementedError

    def _calculate_log_likelihood_at(self, time_step: int) -> np.ndarray:
        raise NotImplementedError
