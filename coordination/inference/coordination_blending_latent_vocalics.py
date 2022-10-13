from __future__ import annotations
from typing import Callable, Dict, List, Optional

import copy
import numpy as np
from scipy.stats import norm

from coordination.common.dataset import Dataset, SeriesData
from coordination.inference.inference_engine import InferenceEngine
from coordination.inference.particle_filter import Particles, ParticleFilter


class LatentVocalicsParticles(Particles):

    latent_vocalics: Dict[str, np.ndarray]

    def _keep_particles_at(self, indices: np.ndarray):
        super()._keep_particles_at(indices)

        for speaker, latent_vocalics in self.latent_vocalics.items():
            if latent_vocalics is not None:
                self.latent_vocalics[speaker] = latent_vocalics[indices, :]


class CoordinationBlendingInferenceLatentVocalics(InferenceEngine, ParticleFilter):

    def __init__(self,
                 mean_prior_latent_vocalics: np.array,
                 std_prior_latent_vocalics: np.array,
                 std_coordinated_latent_vocalics: np.ndarray,
                 std_observed_vocalics: np.ndarray,
                 f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True,
                 g: Callable = lambda x: x,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_latent_vocalics = mean_prior_latent_vocalics
        self._std_prior_latent_vocalics = std_prior_latent_vocalics
        self._std_coordinated_latent_vocalics = std_coordinated_latent_vocalics
        self._std_observed_vocalics = std_observed_vocalics
        self._f = f
        self._g = g
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

        self.states: List[LatentVocalicsParticles] = []

    def fit(self, input_features: Dataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        raise NotImplementedError

    def predict(self, input_features: Dataset, num_particles: int = 0, *args, **kwargs) -> List[np.ndarray]:
        if input_features.num_trials > 0:
            assert len(self._mean_prior_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self._std_prior_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self._std_coordinated_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self._std_observed_vocalics) == input_features.series[0].vocalics.num_features

        # Set the number of particles to be used by the particle filter estimator
        self.num_particles = num_particles

        result = []
        for d in range(input_features.num_trials):
            self.reset_particles()
            series = input_features.series[d]

            M = int(series.num_time_steps / 2)
            num_time_steps = M + 1 if self._fix_coordination_on_second_half else series.num_time_steps

            params = np.zeros((2, num_time_steps))
            for t in range(0, series.num_time_steps):
                self.next(series)

                # We keep generating latent vocalics after M but not coordination. The fixed coordination is given by
                # the set of particles after the last latent vocalics was generated
                real_time = min(t, M) if self._fix_coordination_on_second_half else t
                mean = self.states[-1].mean()
                variance = self.states[-1].var()
                params[:, real_time] = [mean, variance]

            result.append(params)

        return result

    def _sample_from_prior(self, series: SeriesData) -> Particles:
        new_particles = self._create_new_particles()
        self._sample_coordination_from_prior(new_particles)
        self._sample_vocalics_from_prior(series, new_particles)

        return new_particles

    def _sample_vocalics_from_prior(self, series: SeriesData, new_particles: LatentVocalicsParticles):
        new_particles.latent_vocalics = {subject: None for subject in series.vocalics.subjects}
        if series.vocalics.mask[0] == 1:
            speaker = series.vocalics.utterances[0].subject_id
            mean = np.ones((self.num_particles, series.vocalics.num_series)) * self._mean_prior_latent_vocalics
            new_particles.latent_vocalics[speaker] = norm(loc=mean, scale=self._std_prior_latent_vocalics).rvs()

    def _sample_from_transition_to(self, time_step: int, series: SeriesData) -> Particles:
        M = int(series.num_time_steps / 2)
        if not self._fix_coordination_on_second_half or time_step <= M:
            new_particles = self._create_new_particles()
            self._sample_coordination_from_transition(self.states[time_step - 1], new_particles)
        else:
            # Coordination if fixed
            new_particles = copy.deepcopy(self.states[time_step - 1])

        self._sample_vocalics_from_transition_to(time_step, series, new_particles)

        return new_particles

    def _sample_vocalics_from_transition_to(self, time_step: int, series: SeriesData,
                                            new_particles: LatentVocalicsParticles):
        new_particles.latent_vocalics = self.states[time_step - 1].latent_vocalics.copy()
        if series.vocalics.mask[time_step] == 1:
            speaker = series.vocalics.utterances[time_step].subject_id
            A_prev = self.states[time_step - 1].latent_vocalics[speaker]

            A_prev = self._f(A_prev, 0) if A_prev is not None else np.ones(
                (self.num_particles, series.vocalics.num_series)) * self._mean_prior_latent_vocalics
            if series.vocalics.previous_from_other[time_step] is None:
                new_particles.latent_vocalics[speaker] = norm(loc=A_prev, scale=self._std_prior_latent_vocalics).rvs()
            else:
                other_speaker = series.vocalics.utterances[
                    series.vocalics.previous_from_other[time_step]].subject_id
                B_prev = self._f(self.states[time_step - 1].latent_vocalics[other_speaker], 1)
                D = B_prev - A_prev
                mean = D * new_particles.coordination[:, np.newaxis] + A_prev
                new_particles.latent_vocalics[speaker] = norm(loc=mean,
                                                              scale=self._std_coordinated_latent_vocalics).rvs()

    def _calculate_log_likelihood_at(self, time_step: int, series: SeriesData):
        if series.vocalics.mask[time_step] == 1:
            speaker = series.vocalics.utterances[time_step].subject_id
            A_t = self.states[time_step].latent_vocalics[speaker]
            O_t = series.vocalics.values[:, time_step]
            log_likelihoods = norm(loc=self._g(A_t), scale=self._std_observed_vocalics).logpdf(O_t).sum(axis=1)
        else:
            log_likelihoods = 0

        return log_likelihoods

    def _resample_at(self, time_step: int, series: SeriesData):
        return series.vocalics.mask[time_step] == 1 and series.vocalics.previous_from_other[
            time_step] is not None

    def _sample_coordination_from_prior(self, new_particles: LatentVocalicsParticles):
        raise NotImplementedError

    def _sample_coordination_from_transition(self, previous_particles: LatentVocalicsParticles,
                                             new_particles: LatentVocalicsParticles) -> Particles:
        raise NotImplementedError

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return LatentVocalicsParticles()
