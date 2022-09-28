from typing import Callable

import numpy as np
from scipy.stats import norm, truncnorm

from coordination.component.speech.common import VocalicsSparseSeries
from coordination.inference.inference_engine import InferenceEngine
from coordination.inference.particle_filter import ParticleFilter

MIN_VALUE = 0
MAX_VALUE = 1


class TruncatedGaussianCoordinationMixtureInference(InferenceEngine, ParticleFilter):

    def __init__(self, vocalic_series: VocalicsSparseSeries, mean_prior_coordination: float,
                 std_prior_coordination: float, std_coordination_drifting: float, mean_prior_vocalics: np.array,
                 std_prior_vocalics: np.array, std_uncoordinated_vocalics: np.ndarray,
                 std_coordinated_vocalics: np.ndarray, f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_vocalics) == vocalic_series.num_series
        assert len(std_prior_vocalics) == vocalic_series.num_series
        assert len(std_uncoordinated_vocalics) == vocalic_series.num_series
        assert len(std_coordinated_vocalics) == vocalic_series.num_series

        self._vocalic_series = vocalic_series
        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_uncoordinated_vocalics = std_uncoordinated_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

        self._num_features, self._time_steps = vocalic_series.values.shape  # n and T

    def estimate_means_and_variances(self) -> np.ndarray:
        M = int(self._time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else self._time_steps

        params = np.zeros((2, num_time_steps))
        for t in range(0, num_time_steps):
            self.next()
            mean = self.states[-1].mean()
            variance = self.states[-1].var()
            params[:, t] = [mean, variance]

        return params

    def _sample_from_prior(self):
        if self._std_prior_coordination == 0:
            return np.ones(self.num_particles) * self._mean_prior_coordination
        else:
            mean = np.ones(self.num_particles) * self._mean_prior_coordination
            std = np.ones(self.num_particles) * self._std_prior_coordination
            a = (MIN_VALUE - self._mean_prior_coordination) / self._std_prior_coordination
            b = (MAX_VALUE - self._mean_prior_coordination) / self._std_prior_coordination

            return truncnorm(loc=mean, scale=std, a=a, b=b).rvs()

    def _sample_from_transition_to(self, time_step: int):
        std = np.ones(self.num_particles) * self._std_coordination_drifting
        a = (MIN_VALUE - self.states[time_step - 1]) / std
        b = (MAX_VALUE - self.states[time_step - 1]) / std
        return truncnorm(loc=self.states[time_step - 1], scale=std, a=a, b=b).rvs()

    def _calculate_log_likelihood_at(self, time_step: int):
        final_time_step = time_step
        M = int(self._time_steps / 2)
        if self._fix_coordination_on_second_half and time_step == M:
            final_time_step = self._time_steps - 1

        log_likelihoods = 0
        for t in range(time_step, final_time_step + 1):
            A_t = self._f(self._vocalic_series.values[:, t], 0)
            A_prev = None if self._vocalic_series.previous_from_self[t] is None else self._f(
                self._vocalic_series.values[:, self._vocalic_series.previous_from_self[t]], 0)
            B_prev = None if self._vocalic_series.previous_from_other[t] is None else self._f(
                self._vocalic_series.values[:, self._vocalic_series.previous_from_other[t]], 1)

            if self._vocalic_series.mask[t] == 1 and B_prev is not None:
                u = np.random.rand(self.num_particles)
                means = np.zeros((self.num_particles, len(A_t)))
                stds = np.zeros_like(means)

                if A_prev is None:
                    means[u <= self.states[time_step]] = B_prev
                    means[u > self.states[time_step]] = self._mean_prior_vocalics
                    stds[u <= self.states[time_step]] = self._std_coordinated_vocalics
                    stds[u > self.states[time_step]] = self._std_prior_vocalics
                else:
                    means[u <= self.states[time_step]] = B_prev
                    means[u > self.states[time_step]] = A_prev
                    stds[u <= self.states[time_step]] = self._std_coordinated_vocalics
                    stds[u > self.states[time_step]] = self._std_uncoordinated_vocalics

                log_likelihoods += norm(loc=means, scale=stds).logpdf(A_t).sum(axis=1)

        return log_likelihoods

    def _resample_at(self, time_step: int):
        B_prev = None if self._vocalic_series.previous_from_other[time_step] is None else self._f(
            self._vocalic_series.values[:, self._vocalic_series.previous_from_other[time_step]], 1)
        return self._vocalic_series.mask[time_step] == 1 and B_prev is not None
