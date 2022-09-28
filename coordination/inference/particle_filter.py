import random

import numpy as np


class ParticleFilter:

    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        self.states = []
        self.importance_weights = np.zeros(num_particles)

    def next(self):
        next_time_step = len(self.states)
        self.__predict()
        if self._resample_at(next_time_step):
            self.__weight()
            self.__resample()

    def __predict(self):
        if len(self.states) == 0:
            self.states.append(self._sample_from_prior())
        else:
            t = len(self.states)
            self.states.append(self._sample_from_transition_to(t))

    def __weight(self):
        t = len(self.states) - 1
        log_weights = self._calculate_log_likelihood_at(t)
        log_weights -= np.max(log_weights)
        log_weights = np.exp(log_weights)
        self.importance_weights = log_weights / np.sum(log_weights)

    def __resample(self):
        new_particles = np.random.choice(self.num_particles, self.num_particles, replace=True,
                                         p=self.importance_weights)
        self.states[-1] = self.states[-1][new_particles]

    def _resample_at(self, time_step: int):
        return True

    def _sample_from_prior(self):
        raise Exception("Not implemented.")

    def _sample_from_transition_to(self, time_step: int):
        raise Exception("Not implemented.")

    def _calculate_log_likelihood_at(self, time_step: int):
        raise Exception("Not implemented.")
