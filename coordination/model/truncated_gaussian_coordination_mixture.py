from typing import Callable, List

import numpy as np
from scipy.stats import norm, truncnorm

from coordination.common.dataset import InputFeaturesDataset, SeriesData
from coordination.model.coordination_model import CoordinationModel
from coordination.model.particle_filter import Particles, ParticleFilter

MIN_VALUE = 0
MAX_VALUE = 1


class TruncatedGaussianCoordinationMixtureInference(CoordinationModel, ParticleFilter):

    def __init__(self,
                 mean_prior_coordination: float,
                 std_prior_coordination: float,
                 std_coordination_drifting: float,
                 mean_prior_vocalics: np.array,
                 std_prior_vocalics: np.array,
                 std_uncoordinated_vocalics: np.ndarray,
                 std_coordinated_vocalics: np.ndarray,
                 f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_uncoordinated_vocalics = std_uncoordinated_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

    def fit(self, input_features: InputFeaturesDataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        return self

    def predict(self, input_features: InputFeaturesDataset, num_particles: int = 0, *args, **kwargs) -> List[np.ndarray]:
        if input_features.num_trials > 0:
            assert len(self._mean_prior_vocalics) == input_features.series[0].vocalics.num_series
            assert len(self._std_prior_vocalics) == input_features.series[0].vocalics.num_series
            assert len(self._std_uncoordinated_vocalics) == input_features.series[0].vocalics.num_series
            assert len(self._std_coordinated_vocalics) == input_features.series[0].vocalics.num_series

        # Set the number of particles to be used by the particle filter estimator
        self.num_particles = num_particles

        result = []
        for d in range(input_features.num_trials):
            self.reset_state()
            series = input_features.series[d]

            M = int(series.num_time_steps / 2)
            num_time_steps = M + 1 if self._fix_coordination_on_second_half else series.num_time_steps

            params = np.zeros((2, num_time_steps))
            for t in range(0, num_time_steps):
                self.next(series)
                mean = self.states[-1].mean()
                variance = self.states[-1].var()
                params[:, t] = [mean, variance]

            result.append(params)

        return result

    def _sample_from_prior(self, series: SeriesData) -> Particles:
        new_particles = Particles()
        if self._std_prior_coordination == 0:
            new_particles.coordination = np.ones(self.num_particles) * self._mean_prior_coordination
        else:
            mean = np.ones(self.num_particles) * self._mean_prior_coordination
            std = np.ones(self.num_particles) * self._std_prior_coordination
            a = (MIN_VALUE - self._mean_prior_coordination) / self._std_prior_coordination
            b = (MAX_VALUE - self._mean_prior_coordination) / self._std_prior_coordination

            new_particles.coordination = truncnorm(loc=mean, scale=std, a=a, b=b).rvs()

        return new_particles

    def _sample_from_transition_to(self, time_step: int, series: SeriesData) -> Particles:
        std = np.ones(self.num_particles) * self._std_coordination_drifting
        a = (MIN_VALUE - self.states[time_step - 1].coordination) / std
        b = (MAX_VALUE - self.states[time_step - 1].coordination) / std

        new_particles = Particles()
        new_particles.coordination = truncnorm(loc=self.states[time_step - 1], scale=std, a=a, b=b).rvs()

        return new_particles

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
                u = np.random.rand(self.num_particles)
                means = np.zeros((self.num_particles, len(A_t)))
                stds = np.zeros_like(means)

                if A_prev is None:
                    means[u <= self.states[time_step].coordination] = B_prev
                    means[u > self.states[time_step].coordination] = self._mean_prior_vocalics
                    stds[u <= self.states[time_step].coordination] = self._std_coordinated_vocalics
                    stds[u > self.states[time_step].coordination] = self._std_prior_vocalics
                else:
                    means[u <= self.states[time_step].coordination] = B_prev
                    means[u > self.states[time_step].coordination] = A_prev
                    stds[u <= self.states[time_step].coordination] = self._std_coordinated_vocalics
                    stds[u > self.states[time_step].coordination] = self._std_uncoordinated_vocalics

                log_likelihoods += norm(loc=means, scale=stds).logpdf(A_t).sum(axis=1)

        return log_likelihoods

    def _resample_at(self, time_step: int, series: SeriesData):
        return series.vocalics.mask[time_step] == 1 and series.vocalics.previous_from_other[
            time_step] is not None
