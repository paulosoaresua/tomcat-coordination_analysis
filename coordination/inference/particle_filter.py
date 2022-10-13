from typing import List

import numpy as np

from coordination.common.dataset import SeriesData


class Particles:

    def __init__(self, coordination: np.ndarray):
        self.coordination = coordination

    def resample(self, importance_weights: np.ndarray):
        num_particles = len(importance_weights)
        new_particles = np.random.choice(num_particles, num_particles, replace=True, p=importance_weights)
        self.coordination = self.coordination[new_particles]


class ParticleFilter:

    def __init__(self):
        self.num_particles = 0
        self.states: List[Particles] = []

    def reset_particles(self):
        self.states = []

    def next(self, series: SeriesData):
        next_time_step = len(self.states)
        self.__transition(series)
        if self._resample_at(next_time_step, series):
            self.__resample(series)

    def __transition(self, series: SeriesData):
        if len(self.states) == 0:
            self.states.append(self._sample_from_prior(series))
        else:
            t = len(self.states)
            self.states.append(self._sample_from_transition_to(t, series))

    def __resample(self, series: SeriesData):
        t = len(self.states) - 1
        log_weights = self._calculate_log_likelihood_at(t, series)
        log_weights -= np.max(log_weights)
        log_weights = np.exp(log_weights)
        importance_weights = log_weights / np.sum(log_weights)
        self.states[t].resample(importance_weights)

    def _resample_at(self, time_step: int, series: SeriesData):
        return True

    def _sample_from_prior(self, series: SeriesData) -> Particles:
        raise NotImplementedError

    def _sample_from_transition_to(self, time_step: int, series: SeriesData) -> Particles:
        raise NotImplementedError

    def _calculate_log_likelihood_at(self, time_step: int, series: SeriesData) -> np.ndarray:
        raise NotImplementedError
