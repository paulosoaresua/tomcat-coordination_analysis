from abc import ABC
from typing import Callable, List

import numpy as np
from scipy.stats import beta, norm

from coordination.common.dataset import Dataset, SeriesData
from coordination.inference.inference_engine import InferenceEngine
from coordination.inference.particle_filter import Particles, ParticleFilter


class BetaCoordinationInferenceFromVocalics(InferenceEngine, ParticleFilter):

    def __init__(self,
                 prior_a: float,
                 prior_b: float,
                 std_coordination_drifting: float,
                 mean_prior_vocalics: np.array,
                 std_prior_vocalics: np.array,
                 std_coordinated_vocalics: np.ndarray,
                 f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._prior_a = prior_a
        self._prior_b = prior_b
        self._var_coordination_drifting = std_coordination_drifting ** 2
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

    def fit(self, input_features: Dataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        raise NotImplementedError

    def predict(self, input_features: Dataset, num_particles: int = 0, *args, **kwargs) -> List[np.ndarray]:
        if input_features.num_trials > 0:
            assert len(self._mean_prior_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self._std_prior_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self._std_coordinated_vocalics) == input_features.series[0].vocalics.num_features

        # Set the number of particles to be used by the particle filter estimator
        self.num_particles = num_particles

        result = []
        for d in range(input_features.num_trials):
            self.reset_particles()
            series = input_features.series[d]

            M = int(series.num_time_steps / 2)
            num_time_steps = M + 1 if self._fix_coordination_on_second_half else series.num_time_steps

            params = np.zeros((2, num_time_steps))
            for t in range(0, num_time_steps):
                self.next(series)
                mean = self.states[-1].coordination.mean()
                variance = self.states[-1].coordination.var()
                params[:, t] = [mean, variance]

            result.append(params)

        return result

    def _sample_from_prior(self, series: SeriesData) -> Particles:
        a = np.ones(self.num_particles) * self._prior_a
        b = np.ones(self.num_particles) * self._prior_b
        return Particles(beta(a, b).rvs())

    def _sample_from_transition_to(self, time_step: int, series: SeriesData) -> Particles:
        return Particles(self._get_transition_distribution(self.states[time_step - 1].coordination).rvs())

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

    def _calculate_log_likelihood_at(self, time_step: int, series: SeriesData) -> np.ndarray:
        final_time_step = time_step
        M = int(series.num_time_steps / 2)
        if self._fix_coordination_on_second_half and time_step == M:
            final_time_step = series.num_time_steps - 1

        log_likelihoods = 0
        for t in range(time_step, final_time_step + 1):
            A_t = self._f(series.vocalics.values[:, t], 0)
            A_prev = None if series.vocalics.previous_from_self[t] is None else self._f(
                series.vocalics.values[:, series.vocalics.previous_from_self[t]], 0)
            B_prev = None if series.vocalics.previous_from_other[t] is None else self._f(
                series.vocalics.values[:, series.vocalics.previous_from_other[t]], 1)

            if series.vocalics.mask[t] == 1 and B_prev is not None:
                if A_prev is None:
                    A_prev = self._mean_prior_vocalics

                D = B_prev - A_prev
                log_likelihoods += norm(loc=D * self.states[time_step].coordination[:, np.newaxis] + A_prev,
                                        scale=self._std_coordinated_vocalics).logpdf(A_t).sum(axis=1)

        return log_likelihoods

    def _resample_at(self, time_step: int, series: SeriesData):
        return series.vocalics.mask[time_step] == 1 and series.vocalics.previous_from_other[
            time_step] is not None
