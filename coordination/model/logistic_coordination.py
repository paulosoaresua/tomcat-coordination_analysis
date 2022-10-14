from typing import Callable, List

import numpy as np
from scipy.stats import norm

from coordination.common.dataset import Dataset, SeriesData
from coordination.model.coordination_model import CoordinationModel
from coordination.model.particle_filter import Particles, ParticleFilter

from scipy.special import expit


class LogisticCoordinationParticles(Particles):

    def __init__(self, coordination: np.ndarray, coordination_logit: np.ndarray):
        super().__init__(coordination)
        self.coordination_logit = coordination_logit

    def resample(self, importance_weights: np.ndarray):
        num_particles = len(importance_weights)
        new_particles = np.random.choice(num_particles, num_particles, replace=True, p=importance_weights)
        self.coordination = self.coordination[new_particles]
        self.coordination_logit = self.coordination_logit[new_particles]


class LogisticCoordinationInferenceFromVocalics(CoordinationModel, ParticleFilter):

    def __init__(self,
                 mean_prior_coordination_logit: float,
                 std_prior_coordination_logit: float,
                 std_coordination_logit_drifting: float,
                 mean_prior_vocalics: np.array,
                 std_prior_vocalics: np.array,
                 std_coordinated_vocalics: np.ndarray,
                 f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_coordination_logit = mean_prior_coordination_logit
        self._std_prior_coordination_logit = std_prior_coordination_logit
        self._std_coordination_logit_drifting = std_coordination_logit_drifting
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

        self.states: List[LogisticCoordinationParticles] = []

    def fit(self, input_features: Dataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        return self

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
        mean = np.ones(self.num_particles) * self._mean_prior_coordination_logit
        std = np.ones(self.num_particles) * self._std_prior_coordination_logit
        coordination_logit_particles = norm(loc=mean, scale=std).rvs()
        coordination_particles = expit(coordination_logit_particles)

        return LogisticCoordinationParticles(coordination_particles, coordination_logit_particles)

    def _sample_from_transition_to(self, time_step: int, series: SeriesData) -> Particles:
        coordination_logit_particles = norm(loc=self.states[time_step - 1].coordination_logit,
                                            scale=self._std_coordination_logit_drifting).rvs()
        coordination_particles = expit(coordination_logit_particles)

        return LogisticCoordinationParticles(coordination_particles, coordination_logit_particles)

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
