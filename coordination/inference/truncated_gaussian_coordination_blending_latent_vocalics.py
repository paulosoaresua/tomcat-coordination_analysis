from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import norm, truncnorm

from coordination.component.speech.common import VocalicsSparseSeries
from coordination.inference.inference_engine import InferenceEngine
from coordination.inference.particle_filter import Particles, ParticleFilter


MIN_VALUE = 0
MAX_VALUE = 1


class LatentVocalicsParticles(Particles):

    def __init__(self, coordination: np.ndarray, latent_vocalics: Dict[str, Optional[np.ndarray]]):
        super().__init__(coordination)
        self.latent_vocalics: Dict[str, np.ndarray] = latent_vocalics

    def resample(self, importance_weights: np.ndarray):
        num_particles = len(importance_weights)
        new_particles = np.random.choice(num_particles, num_particles, replace=True, p=importance_weights)
        self.coordination = self.coordination[new_particles]
        for speaker, latent_vocalics in self.latent_vocalics.items():
            if latent_vocalics is not None:
                self.latent_vocalics[speaker] = latent_vocalics[new_particles, :]


class TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(InferenceEngine, ParticleFilter):

    def __init__(self, vocalic_series: VocalicsSparseSeries,
                 mean_prior_coordination: float, std_prior_coordination: float,
                 std_coordination_drifting: float, mean_prior_latent_vocalics: np.array,
                 std_prior_latent_vocalics: np.array, std_coordinated_latent_vocalics: np.ndarray,
                 std_observed_vocalics: np.ndarray, f: Callable = lambda x: x,
                 fix_coordination_on_second_half: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_latent_vocalics) == vocalic_series.num_series
        assert len(std_prior_latent_vocalics) == vocalic_series.num_series
        assert len(std_coordinated_latent_vocalics) == vocalic_series.num_series
        assert len(std_observed_vocalics) == vocalic_series.num_series

        self._vocalic_series = vocalic_series
        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting
        self._mean_prior_latent_vocalics = mean_prior_latent_vocalics
        self._std_prior_latent_vocalics = std_prior_latent_vocalics
        self._std_coordinated_latent_vocalics = std_coordinated_latent_vocalics
        self._std_observed_vocalics = std_observed_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

        self._num_features, self._time_steps = vocalic_series.values.shape  # n and T

        self.states: List[LatentVocalicsParticles] = []

    def estimate_means_and_variances(self) -> np.ndarray:
        M = int(self._time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else self._time_steps

        params = np.zeros((2, num_time_steps))
        for t in range(0, self._time_steps):
            self.next()

            # We keep generating latent vocalics after M but not coordination. The fixed coordination is given by
            # the set of particles after the last latent vocalics was generated
            real_time = min(t, M) if self._fix_coordination_on_second_half else t
            mean = self.states[-1].coordination.mean()
            variance = self.states[-1].coordination.var()
            params[:, real_time] = [mean, variance]

        return params

    def _sample_from_prior(self) -> Particles:
        mean = np.ones(self.num_particles) * self._mean_prior_coordination
        a = (MIN_VALUE - mean) / self._std_prior_coordination
        b = (MAX_VALUE - mean) / self._std_prior_coordination
        coordination_particles = truncnorm(loc=mean, scale=self._std_prior_coordination, a=a, b=b).rvs()

        latent_vocalics_particles = {subject: None for subject in self._vocalic_series.subjects}
        if self._vocalic_series.mask[0] == 1:
            speaker = self._vocalic_series.utterances[0].subject_id
            mean = np.ones((self.num_particles, self._vocalic_series.num_series)) * self._mean_prior_latent_vocalics
            latent_vocalics_particles[speaker] = norm(loc=mean, scale=self._std_prior_latent_vocalics).rvs()

        return LatentVocalicsParticles(coordination_particles, latent_vocalics_particles)

    def _sample_from_transition_to(self, time_step: int):
        M = int(self._time_steps / 2)
        previous_coordination = self.states[time_step - 1].coordination
        if not self._fix_coordination_on_second_half or time_step <= M:
            # Coordination drifts
            a = (MIN_VALUE - previous_coordination) / self._std_coordination_drifting
            b = (MAX_VALUE - previous_coordination) / self._std_coordination_drifting
            coordination_particles = truncnorm(loc=previous_coordination, scale=self._std_coordination_drifting, a=a,
                                               b=b).rvs()
        else:
            # Coordination if fixed
            coordination_particles = previous_coordination

        latent_vocalics_particles = self.states[time_step - 1].latent_vocalics.copy()
        if self._vocalic_series.mask[time_step] == 1:
            speaker = self._vocalic_series.utterances[time_step].subject_id
            A_prev = self.states[time_step - 1].latent_vocalics[speaker]

            A_prev = A_prev if A_prev is not None else np.ones(
                (self.num_particles, self._vocalic_series.num_series)) * self._mean_prior_latent_vocalics
            if self._vocalic_series.previous_from_other[time_step] is None:
                latent_vocalics_particles[speaker] = norm(loc=A_prev, scale=self._std_prior_latent_vocalics).rvs()
            else:
                other_speaker = self._vocalic_series.utterances[
                    self._vocalic_series.previous_from_other[time_step]].subject_id
                B_prev = self.states[time_step - 1].latent_vocalics[other_speaker]
                D = B_prev - A_prev
                mean = D * coordination_particles[:, np.newaxis] + A_prev
                latent_vocalics_particles[speaker] = norm(loc=mean, scale=self._std_coordinated_latent_vocalics).rvs()

        return LatentVocalicsParticles(coordination_particles, latent_vocalics_particles)

    def _calculate_log_likelihood_at(self, time_step: int):
        if self._vocalic_series.mask[time_step] == 1:
            speaker = self._vocalic_series.utterances[time_step].subject_id
            A_t = self.states[time_step].latent_vocalics[speaker]
            O_t = self._vocalic_series.values[:, time_step]
            log_likelihoods = norm(loc=self._f(A_t), scale=self._std_observed_vocalics).logpdf(O_t).sum(axis=1)
        else:
            log_likelihoods = 0

        return log_likelihoods

    def _resample_at(self, time_step: int):
        return self._vocalic_series.mask[time_step] == 1 and self._vocalic_series.previous_from_other[
            time_step] is not None
