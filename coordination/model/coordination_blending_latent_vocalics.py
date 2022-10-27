from __future__ import annotations
from typing import Callable, Dict, List, Optional

import copy
import inspect

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from coordination.common.dataset import EvidenceDataset, SeriesData
from coordination.model.coordination_model import CoordinationModel
from coordination.model.particle_filter import Particles, ParticleFilter


class LatentVocalicsParticles(Particles):
    coordination: np.ndarray
    latent_vocalics: Dict[str, np.ndarray]

    def _keep_particles_at(self, indices: np.ndarray):
        self.coordination = self.coordination[indices]
        for speaker, latent_vocalics in self.latent_vocalics.items():
            if latent_vocalics is not None:
                self.latent_vocalics[speaker] = latent_vocalics[indices, :]

    def mean(self):
        return self.coordination.mean()

    def var(self):
        return self.coordination.var()


def default_f(x: np.ndarray, s: int) -> np.ndarray:
    return x


def default_g(x: np.ndarray) -> np.ndarray:
    return x


class CoordinationBlendingInferenceLatentVocalics(CoordinationModel):

    def __init__(self,
                 mean_prior_latent_vocalics: np.array,
                 std_prior_latent_vocalics: np.array,
                 std_coordinated_latent_vocalics: np.ndarray,
                 std_observed_vocalics: np.ndarray,
                 f: Callable = default_f,
                 g: Callable = default_g,
                 fix_coordination_on_second_half: bool = True,
                 num_particles: int = 10000,
                 seed: Optional[int] = None,
                 show_progress_bar: bool = False):
        super().__init__()

        self.mean_prior_latent_vocalics = mean_prior_latent_vocalics
        self.std_prior_latent_vocalics = std_prior_latent_vocalics
        self.std_coordinated_latent_vocalics = std_coordinated_latent_vocalics
        self.std_observed_vocalics = std_observed_vocalics
        self.f = f
        self.g = g
        self.fix_coordination_on_second_half = fix_coordination_on_second_half
        self.num_particles = num_particles
        self.seed = seed
        self.show_progress_bar = show_progress_bar

    def fit(self, input_features: EvidenceDataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        return self

    def predict(self, input_features: EvidenceDataset, *args, **kwargs) -> List[np.ndarray]:
        if input_features.num_trials > 0:
            assert len(self.mean_prior_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self.std_prior_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self.std_coordinated_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self.std_observed_vocalics) == input_features.series[0].vocalics.num_features

        particle_filter = ParticleFilter(
            num_particles=self.num_particles,
            resample_at_fn=self._resample_at,
            sample_from_prior_fn=self._sample_from_prior,
            sample_from_transition_fn=self._sample_from_transition_to,
            calculate_log_likelihood_fn=self._calculate_log_likelihood_at,
            seed=self.seed
        )

        pbar_outer = None
        if self.show_progress_bar:
            pbar_outer = tqdm(total=input_features.num_trials, desc="Trial", position=0)

        result = []
        for d in range(input_features.num_trials):
            particle_filter.reset_state()
            series = input_features.series[d]

            M = int(series.num_time_steps / 2)
            num_time_steps = M + 1 if self.fix_coordination_on_second_half else series.num_time_steps

            pbar_inner = None
            if self.show_progress_bar:
                pbar_inner = tqdm(total=input_features.series[0].num_time_steps, desc="Time Step", position=1,
                                  leave=False)

            params = np.zeros((2, num_time_steps))
            for t in range(0, series.num_time_steps):
                particle_filter.next(series)

                # We keep generating latent vocalics after M but not coordination. The fixed coordination is given by
                # the set of particles after the last latent vocalics was generated
                real_time = min(t, M) if self.fix_coordination_on_second_half else t
                mean = particle_filter.states[-1].mean()
                variance = particle_filter.states[-1].var()
                params[:, real_time] = [mean, variance]

                if self.show_progress_bar:
                    pbar_inner.update()

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(f"inference/coordination-{series.uuid}", mean, t)

            result.append(params)

            if self.show_progress_bar:
                pbar_outer.update()

            if self.tb_writer is not None:
                self.log_coordination_inference_plot(series, params, series.uuid)

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
            mean = np.ones((self.num_particles, series.vocalics.num_series)) * self.mean_prior_latent_vocalics
            new_particles.latent_vocalics[speaker] = norm(loc=mean, scale=self.std_prior_latent_vocalics).rvs()

    def _sample_from_transition_to(self, time_step: int, states: List[LatentVocalicsParticles],
                                   series: SeriesData) -> LatentVocalicsParticles:
        M = int(series.num_time_steps / 2)
        if not self.fix_coordination_on_second_half or time_step <= M:
            new_particles = self._create_new_particles()
            self._sample_coordination_from_transition(states[time_step - 1], new_particles)
        else:
            # Coordination if fixed
            new_particles = copy.deepcopy(states[time_step - 1])

        self._sample_vocalics_from_transition_to(time_step, states[time_step - 1], new_particles, series)

        return new_particles

    def _sample_vocalics_from_transition_to(self, time_step: int, previous_particles: LatentVocalicsParticles,
                                            new_particles: LatentVocalicsParticles, series: SeriesData):
        new_particles.latent_vocalics = previous_particles.latent_vocalics.copy()
        if series.vocalics.mask[time_step] == 1:
            speaker = series.vocalics.utterances[time_step].subject_id
            A_prev = previous_particles.latent_vocalics[speaker]

            A_prev = self.f(A_prev, 0) if A_prev is not None else np.ones(
                (self.num_particles, series.vocalics.num_series)) * self.mean_prior_latent_vocalics
            if series.vocalics.previous_from_other[time_step] is None:
                new_particles.latent_vocalics[speaker] = norm(loc=A_prev, scale=self.std_prior_latent_vocalics).rvs()
            else:
                other_speaker = series.vocalics.utterances[
                    series.vocalics.previous_from_other[time_step]].subject_id
                B_prev = self.f(previous_particles.latent_vocalics[other_speaker], 1)
                D = B_prev - A_prev
                mean = D * new_particles.coordination[:, np.newaxis] + A_prev
                new_particles.latent_vocalics[speaker] = norm(loc=mean,
                                                              scale=self.std_coordinated_latent_vocalics).rvs()

    def _calculate_log_likelihood_at(self, time_step: int, states: List[LatentVocalicsParticles], series: SeriesData):
        if series.vocalics.mask[time_step] == 1:
            speaker = series.vocalics.utterances[time_step].subject_id
            A_t = states[time_step].latent_vocalics[speaker]
            O_t = series.vocalics.values[:, time_step]
            log_likelihoods = norm(loc=self.g(A_t), scale=self.std_observed_vocalics).logpdf(O_t).sum(axis=1)
        else:
            log_likelihoods = 0

        return log_likelihoods

    @staticmethod
    def _resample_at(time_step: int, series: SeriesData):
        return series.vocalics.mask[time_step] == 1 and series.vocalics.previous_from_other[
            time_step] is not None

    def _sample_coordination_from_prior(self, new_particles: LatentVocalicsParticles):
        raise NotImplementedError

    def _sample_coordination_from_transition(self, previous_particles: LatentVocalicsParticles,
                                             new_particles: LatentVocalicsParticles) -> Particles:
        raise NotImplementedError

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return LatentVocalicsParticles()
