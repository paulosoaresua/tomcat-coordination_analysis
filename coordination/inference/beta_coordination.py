from typing import Callable, Optional

import random

import numpy as np
from scipy.stats import beta, norm

from coordination.component.speech.common import VocalicsSparseSeries
from coordination.inference.particle_filter import ParticleFilter


class BetaCoordinationInferenceFromVocalics(ParticleFilter):

    def __init__(self, vocalic_series: VocalicsSparseSeries, prior_a: float, prior_b: float,
                 std_coordination_drifting: float, mean_prior_vocalics: np.array, std_prior_vocalics: np.array,
                 std_coordinated_vocalics: np.ndarray, f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_vocalics) == vocalic_series.num_series
        assert len(std_prior_vocalics) == vocalic_series.num_series
        assert len(std_coordinated_vocalics) == vocalic_series.num_series

        self._prior_a = prior_a
        self._prior_b = prior_b
        self._vocalic_series = vocalic_series
        self._var_coordination_drifting = std_coordination_drifting ** 2
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

        self._num_features, self._time_steps = vocalic_series.values.shape  # n and T

    def _sample_from_prior(self):
        a = np.ones(self.num_particles) * self._prior_a
        b = np.ones(self.num_particles) * self._prior_b
        return beta(a, b).rvs()

    def _sample_from_transition_to(self, time_step: int):
        return self._get_transition_distribution(self.states[time_step - 1]).rvs()

    def _get_transition_distribution(self, previous_coordination: np.ndarray):
        """
        The transition distribution is a Beta distribution with mean equals to the previous coordination and
        fixed variance. Therefore, the parameters a and b of the resulting distribution depends on the previous
        coordination and fixed variance and can be determined analytically by solving the system of equations that
        define the mean and variance of a beta distribution:

        mean (= previous coordination) = a / (a + b)
        variance = (a + b)^2(a + b + 1)

        See section Mean and Variance:
        https://en.wikipedia.org/wiki/Beta_distribution
        """
        x = previous_coordination * (1 - previous_coordination)

        indices1 = np.where(x < 0.1)
        indices2 = np.where(x > 0.9)
        indices3 = np.where(x > self._var_coordination_drifting)

        a = np.zeros_like(previous_coordination)
        b = np.zeros_like(previous_coordination)

        b[indices1] = 1
        b[indices2] = 0.1
        b[indices3] = previous_coordination[indices3] * (
            (1 - previous_coordination[indices3])) ** 2 / self._var_coordination_drifting + previous_coordination[
                          indices3] - 1

        a[indices1] = 0.1
        a[indices2] = 1
        a[indices3] = b[indices3] * previous_coordination[indices3] / (1 - previous_coordination[indices3])

        if ((self._var_coordination_drifting >= x) & (previous_coordination > 0.1) & (
                previous_coordination < 0.9)).any():
            raise Exception(f"Unsupported: {previous_coordination}")

        return beta(a, b)

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
                if A_prev is None:
                    A_prev = self._mean_prior_vocalics

                D = B_prev - A_prev
                log_likelihoods += norm(loc=D * self.states[time_step][:, np.newaxis] + A_prev,
                                        scale=self._std_coordinated_vocalics).logpdf(A_t).sum(axis=1)

        return log_likelihoods

    def _resample_at(self, time_step: int):
        B_prev = None if self._vocalic_series.previous_from_other[time_step] is None else self._f(
            self._vocalic_series.values[:, self._vocalic_series.previous_from_other[time_step]], 1)
        return self._vocalic_series.mask[time_step] == 1 and B_prev is not None

    def estimate_means_and_variances(self, seed: Optional[float]) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        M = int(self._time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else self._time_steps

        params = np.zeros((2, num_time_steps))
        for t in range(0, num_time_steps):
            self.next()
            mean = self.states[-1].mean()
            variance = self.states[-1].var()
            params[:, t] = [mean, variance]

        return params
