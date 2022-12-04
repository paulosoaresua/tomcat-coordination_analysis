from __future__ import annotations
from typing import Dict, List, Optional, TypeVar

from datetime import datetime
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm, invgamma
from tqdm import tqdm

from coordination.common.log import BaseLogger
from coordination.common.parallelism import display_inner_progress_bar
from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries
from coordination.model.particle_filter import Particles
from coordination.model.pgm import PGM
from coordination.model.utils.coordination_blending_latent_vocalics import LatentVocalicsParticles, \
    LatentVocalicsParticlesSummary, LatentVocalicsSamples, LatentVocalicsDataSeries, LatentVocalicsDataset, BaseF, \
    BaseG, clip_coordination, LatentVocalicsTrainingHyperParameters, LatentVocalicsModelParameters

SP = TypeVar('SP')
S = TypeVar('S')


class CoordinationBlendingLatentVocalics(PGM[SP, S]):

    def __init__(self,
                 initial_coordination: float,
                 num_vocalic_features: int,
                 num_speakers: int,
                 f: BaseF,
                 g: BaseG,
                 disable_self_dependency):
        """

        @param initial_coordination: Initial value of coordination
        @param num_vocalic_features: Number of vocalic features
        @param num_speakers: Number of speakers
        @param f: Latent vocalics transition transformation
        @param g: Latent vocalics emission transformation
        @param disable_self_dependency: Whether coordination mediates dependency on self and other or only other.
        """
        super().__init__()

        self.initial_coordination = initial_coordination
        self.num_vocalic_features = num_vocalic_features
        self.num_speakers = num_speakers
        self.f = f
        self.g = g
        self.disable_self_dependency = disable_self_dependency

        self._hyper_params = {
            "c0": initial_coordination,
            "#features": num_vocalic_features,
            "#speakers": num_speakers,
            "f": f.__repr__(),
            "g": g.__repr__(),
            "disable_self_dependency": disable_self_dependency
        }

        # Trainable parameters of the model
        self.parameters = LatentVocalicsModelParameters()

        # Last sampled value of coordination and latent vocalics during training
        self.last_coordination_samples_ = np.array([])
        self.last_latent_vocalics_samples_ = np.array([])

        # The variable below is initialized during training to create blocks of time stamps for which variables can be
        # updated in parallel. Latent vocalics will be sampled in sequence (single process) to avoid a fancier logic to
        # retrieve dependencies. This is okay for now because latent vocalics posterior has a closed form and sampling
        # is faster compared to coordination that is sampled via MCMC.
        self._latent_vocalics_time_step_block = np.array([])

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------

    def sample(self, num_samples: int, num_time_steps: int, seed: Optional[int], time_scale_density: float = 1, *args,
               **kwargs) -> SP:
        """
        Regular ancestral sampling procedure.
        """
        super().sample(num_samples, num_time_steps, seed)

        samples = LatentVocalicsSamples()
        self._generate_coordination_samples(num_samples, num_time_steps, samples)
        samples.latent_vocalics = []
        samples.observed_vocalics = []

        for i in tqdm(range(num_samples), desc="Sampling Trial", position=0, leave=False):
            # Subjects A and B
            previous_self = [None] * num_time_steps
            previous_other = [None] * num_time_steps
            previous_time_per_speaker: Dict[str, int] = {}
            latent_vocalics_values = np.zeros((self.num_vocalic_features, num_time_steps))
            observed_vocalics_values = np.zeros((self.num_vocalic_features, num_time_steps))
            utterances: List[Optional[SegmentedUtterance]] = [None] * num_time_steps

            speakers = self._generate_random_speakers(num_time_steps, time_scale_density)
            mask = np.zeros(num_time_steps)

            for t in tqdm(range(num_time_steps), desc="Sampling Time Step", position=1, leave=False):
                current_coordination = samples.coordination[i, t]

                if speakers[t] is not None:
                    mask[t] = 1

                    previous_time_self = previous_time_per_speaker.get(speakers[t], None)
                    previous_time_other = None
                    for speaker, time in previous_time_per_speaker.items():
                        if speaker == speakers[t]:
                            continue

                        # Most recent vocalics from a different speaker
                        previous_time_other = time if previous_time_other is None else max(previous_time_other, time)

                    previous_value_self = None if previous_time_self is None else latent_vocalics_values[:,
                                                                                  previous_time_self]
                    previous_value_other = None if previous_time_other is None else latent_vocalics_values[:,
                                                                                    previous_time_other]

                    latent_vocalics_values[:, t] = self._sample_latent_vocalics(previous_value_self,
                                                                                previous_value_other,
                                                                                current_coordination)
                    observed_vocalics_values[:, t] = self._sample_observed_vocalics(latent_vocalics_values[:, t])

                    previous_self[t] = previous_time_self
                    previous_other[t] = previous_time_other

                    # Dummy utterance
                    utterances[t] = SegmentedUtterance(f"Speaker {speakers[t]}", datetime.now(), datetime.now(), "")
                    previous_time_per_speaker[speakers[t]] = t

            samples.latent_vocalics.append(VocalicsSparseSeries(utterances=utterances, previous_from_self=previous_self,
                                                                previous_from_other=previous_other,
                                                                values=latent_vocalics_values, mask=mask))
            samples.observed_vocalics.append(
                VocalicsSparseSeries(utterances=utterances, previous_from_self=previous_self,
                                     previous_from_other=previous_other,
                                     values=observed_vocalics_values, mask=mask))

        return samples

    def _generate_coordination_samples(self, num_samples: int, num_time_steps: int, samples: LatentVocalicsSamples):
        raise NotImplementedError

    def _generate_random_speakers(self, num_time_steps: int, time_scale_density: float) -> List[Optional[str]]:
        # We always change speakers between time steps when generating vocalics
        transition_matrix = 1 - np.eye(self.num_speakers + 1)

        transition_matrix *= time_scale_density / (self.num_speakers - 1)
        transition_matrix[:, -1] = 1 - time_scale_density

        prior = np.ones(self.num_speakers + 1) * time_scale_density / self.num_speakers
        prior[-1] = 1 - time_scale_density
        transition_matrix[-1] = prior

        initial_speaker = np.random.choice(self.num_speakers + 1, 1, p=prior)[0]
        initial_speaker = None if initial_speaker == self.num_speakers else initial_speaker
        speakers = [initial_speaker]

        for t in range(1, num_time_steps):
            probabilities = transition_matrix[self.num_speakers] if speakers[t - 1] is None else transition_matrix[
                speakers[t - 1]]
            speaker = np.random.choice(self.num_speakers + 1, 1, p=probabilities)[0]
            speaker = None if speaker == self.num_speakers else speaker
            speakers.append(speaker)

        return speakers

    def _sample_latent_vocalics(self, previous_self: Optional[float], previous_other: Optional[float],
                                coordination: float) -> np.ndarray:
        if previous_other is None:
            if previous_self is None or self.disable_self_dependency:
                distribution = norm(loc=np.zeros(self.num_vocalic_features), scale=np.sqrt(self.parameters.var_a))
            else:
                distribution = norm(loc=self.f(previous_self, 0), scale=np.sqrt(self.parameters.var_aa))
        else:
            if previous_self is None or self.disable_self_dependency:
                D = self.f(previous_other, 1)
                distribution = norm(loc=D * clip_coordination(coordination), scale=np.sqrt(self.parameters.var_aa))
            else:
                D = self.f(previous_other, 1) - self.f(previous_self, 0)
                distribution = norm(loc=D * clip_coordination(coordination) + previous_self,
                                    scale=np.sqrt(self.parameters.var_aa))

        return distribution.rvs()

    def _sample_observed_vocalics(self, latent_vocalics: np.array) -> np.ndarray:
        return norm(loc=self.g(latent_vocalics), scale=np.sqrt(self.parameters.var_o)).rvs()

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------

    def _fit_init(self, evidence: LatentVocalicsDataset, train_hyper_parameters: LatentVocalicsTrainingHyperParameters,
                  burn_in: int,
                  seed: Optional[int], num_jobs: int = 1, logger: BaseLogger = BaseLogger()):
        # Self-dependency can be toggled by just modifying some variables in the dataset. We do this here and change the
        # dataset to its original configuration in the end of the fit method.
        if self.disable_self_dependency:
            evidence.disable_self_dependency()

    def _fit_end(self, evidence: LatentVocalicsDataset, train_hyper_parameters: LatentVocalicsTrainingHyperParameters,
                 burn_in: int,
                 seed: Optional[int], num_jobs: int = 1, logger: BaseLogger = BaseLogger()):
        if self.disable_self_dependency:
            evidence.enable_self_dependency()

    def _initialize_gibbs(self, evidence: LatentVocalicsDataset,
                          train_hyper_parameters: LatentVocalicsTrainingHyperParameters, burn_in: int, seed: int,
                          num_jobs: int):

        self._create_parallel_time_step_blocks(evidence, num_jobs)
        self._initialize_parameters(evidence, train_hyper_parameters, burn_in, seed)
        self._initialize_latent_variables(evidence, train_hyper_parameters, burn_in, seed)

    def _create_parallel_time_step_blocks(self, evidence: LatentVocalicsDataset, num_jobs: int):
        # Latent vocalics will be updated in a single process
        self._latent_vocalics_time_step_block = np.arange(evidence.num_time_steps)

    def _initialize_parameters(self, evidence: LatentVocalicsDataset,
                               train_hyper_parameters: LatentVocalicsTrainingHyperParameters, burn_in: int, seed: int):

        # Dependent on the subclass implementation
        self._initialize_coordination_parameters_for_fit(evidence, train_hyper_parameters, burn_in, seed)

        if not self.parameters.var_a_frozen:
            self.parameters.set_var_a(train_hyper_parameters.va0, freeze=False)

        if not self.parameters.var_aa_frozen:
            self.parameters.set_var_aa(train_hyper_parameters.vaa0, freeze=False)

        if not self.parameters.var_o_frozen:
            self.parameters.set_var_o(train_hyper_parameters.vo0, freeze=False)

    def _initialize_coordination_parameters_for_fit(self, evidence: LatentVocalicsDataset,
                                                    train_hyper_parameters: LatentVocalicsTrainingHyperParameters,
                                                    burn_in: int,
                                                    seed: int):
        raise NotImplementedError

    def _initialize_latent_variables(self, evidence: LatentVocalicsDataset,
                                     train_hyper_parameters: LatentVocalicsTrainingHyperParameters, burn_in: int,
                                     seed: int):

        # Dependent on the subclass implementation
        self._initialize_coordination_for_fit(evidence, train_hyper_parameters, burn_in, seed)

        if evidence.latent_vocalics is None:
            # Initialize it from a standard normal
            self.last_latent_vocalics_samples_ = norm(
                loc=np.zeros((evidence.num_trials, self.num_vocalic_features, evidence.num_time_steps)), scale=1).rvs()
        else:
            self.last_latent_vocalics_samples_ = evidence.latent_vocalics.copy()

    def _initialize_coordination_for_fit(self, evidence: LatentVocalicsDataset,
                                         train_hyper_parameters: LatentVocalicsTrainingHyperParameters,
                                         burn_in: int, seed: int):
        raise NotImplementedError

    def _get_max_num_jobs(self) -> int:
        raise NotImplementedError

    def _update_latent_variables(self, evidence: LatentVocalicsDataset,
                                 train_hyper_parameters: LatentVocalicsTrainingHyperParameters,
                                 pool: Pool):
        self._update_coordination(evidence, train_hyper_parameters, pool)
        self._update_vocalics(evidence)

    def _update_coordination(self, evidence: LatentVocalicsDataset,
                             train_hyper_parameters: LatentVocalicsTrainingHyperParameters, pool: Pool):
        raise NotImplementedError

    def _update_vocalics(self, evidence: LatentVocalicsDataset):
        """
        Sample vocalics in a single thread. A fancier logic may be applied to sample vocalics in different threads
        by dealing with the jumps in dependencies such that parallel blocks do not have dependent vocalics, but
        we don't do this by now since sampling from vocalics is fast.
        """

        if evidence.latent_vocalics is None:
            self.last_latent_vocalics_samples_ = self._sample_latent_vocalics_on_fit(evidence,
                                                                                     self._latent_vocalics_time_step_block,
                                                                                     1, 0)

    def _sample_latent_vocalics_on_fit(self, evidence: LatentVocalicsDataset,
                                       time_steps: np.ndarray, job_num: int, group_order: int) -> np.ndarray:

        va = self.parameters.var_a
        vaa = self.parameters.var_aa
        vo = self.parameters.var_o
        coordination = self.last_coordination_samples_

        latent_vocalics = self.last_latent_vocalics_samples_.copy()

        pbar = None
        if display_inner_progress_bar():
            pbar = tqdm(time_steps, desc=f"Sampling Latent Vocalics (Group {group_order + 1})", position=job_num,
                        leave=False)

        for t in time_steps:
            C1 = clip_coordination(coordination[:, t][:, np.newaxis])
            M1 = evidence.vocalics_mask[:, t][:, np.newaxis]

            previous_times_from_self = evidence.previous_vocalics_from_self[:, t]
            A1 = np.take_along_axis(latent_vocalics, previous_times_from_self[:, np.newaxis, np.newaxis], axis=-1)[:, :,
                 0]
            Ma1 = np.where(previous_times_from_self >= 0, 1, 0)[:, np.newaxis]

            previous_times_from_other = evidence.previous_vocalics_from_other[:, t]
            B1 = np.take_along_axis(latent_vocalics, previous_times_from_other[:, np.newaxis, np.newaxis], axis=-1)[
                ..., 0]
            Mb1 = np.where(previous_times_from_other >= 0, 1, 0)[:, np.newaxis]

            # Time steps in which the next speaker is the same
            t2a = evidence.next_vocalics_from_self[:, t]
            C2a = clip_coordination(np.take_along_axis(coordination, t2a[:, np.newaxis], axis=-1))
            V2a = np.take_along_axis(latent_vocalics, t2a[:, np.newaxis, np.newaxis], axis=-1)[..., 0]
            M2a = np.take_along_axis(evidence.vocalics_mask, t2a[:, np.newaxis], axis=-1)
            Ma2a = np.where(t2a >= 0, 1, 0)[:, np.newaxis]
            previous_times_from_other = np.take_along_axis(evidence.previous_vocalics_from_other, t2a[:, np.newaxis],
                                                           axis=-1)
            B2a = np.take_along_axis(latent_vocalics, previous_times_from_other[..., np.newaxis], axis=-1)[..., 0]
            Mb2a = np.where(previous_times_from_other >= 0, 1, 0)

            # Time steps in which the next speaker is different
            t2b = evidence.next_vocalics_from_other[:, t]
            C2b = clip_coordination(np.take_along_axis(coordination, t2b[:, np.newaxis], axis=-1))
            V2b = np.take_along_axis(latent_vocalics, t2b[:, np.newaxis, np.newaxis], axis=-1)[..., 0]
            M2b = np.take_along_axis(evidence.vocalics_mask, t2b[:, np.newaxis], axis=-1)
            previous_times_from_self = np.take_along_axis(evidence.previous_vocalics_from_self, t2b[:, np.newaxis],
                                                          axis=-1)
            A2b = np.take_along_axis(latent_vocalics, previous_times_from_self[..., np.newaxis], axis=-1)[..., 0]
            Ma2b = np.where(previous_times_from_self >= 0, 1, 0)
            Mb2b = np.where(t2b >= 0, 1, 0)[:, np.newaxis]

            m1 = (B1 - A1 * Ma1) * C1 * Mb1 + A1 * Ma1
            v1 = np.where((1 - Ma1) * (1 - Mb1) == 1, va, vaa)

            h2a = C2a * Mb2a
            m2a = V2a - h2a * B2a
            u2a = M2a * (1 - h2a)
            v2a = np.where((1 - Ma2a) * (1 - Mb2a) == 1, va, vaa)

            h2b = C2b * Mb2b
            m2b = V2b - A2b * M2b * (1 - h2b)
            u2b = h2b
            v2b = np.where((1 - Ma2b) * (1 - Mb2b) == 1, va, vaa)

            Obs = evidence.observed_vocalics[:, :, t]
            m3 = ((m1 / v1) + ((m2a * u2a * M2a) / v2a) + ((m2b * u2b * M2b) / v2b) + (Obs / vo)) * M1

            v_inv = ((1 / v1) + (((u2a ** 2) * M2a) / v2a) + (((u2b ** 2) * M2b) / v2b) + (1 / vo)) * M1

            # For numerical stability
            v_inv = np.maximum(v_inv, 1E-16)

            v = 1 / v_inv
            m = m3 * v

            sampled_latent_vocalics = norm(loc=m, scale=np.sqrt(v)).rvs()

            if t == 0:
                # Latent vocalic values start with -1 until someone starts to talk
                sampled_latent_vocalics = np.where(M1 == 1, sampled_latent_vocalics, -1)
            else:
                # Repeat previous latent vocalics if no one is talking at time t
                sampled_latent_vocalics = np.where(M1 == 1, sampled_latent_vocalics, latent_vocalics[:, :, t - 1])

            latent_vocalics[:, :, t] = sampled_latent_vocalics

            if display_inner_progress_bar():
                pbar.update()

        return latent_vocalics

    def _compute_joint_loglikelihood_at(self, evidence: LatentVocalicsDataset,
                                        train_hyper_parameters: LatentVocalicsTrainingHyperParameters) -> float:
        sa = np.sqrt(self.parameters.var_a)
        saa = np.sqrt(self.parameters.var_aa)
        so = np.sqrt(self.parameters.var_o)

        coordination = self.last_coordination_samples_
        latent_vocalics = self.last_latent_vocalics_samples_

        ll = self._compute_coordination_loglikelihood(evidence)
        for t in range(evidence.num_time_steps):
            # Latent vocalics
            C = coordination[:, t][:, np.newaxis]

            # n x k
            V = latent_vocalics[:, :, t]

            # n x 1
            M = evidence.vocalics_mask[:, t][:, np.newaxis]

            # Vs (n x k) will have the values of vocalics from the same speaker per trial
            A = latent_vocalics[range(evidence.num_trials), :, evidence.previous_vocalics_from_self[:, t]]

            # Mask with 1 in the cells in which there are previous vocalics from the same speaker and 0 otherwise
            Ma = evidence.previous_vocalics_from_self_mask[:, t][:,
                 np.newaxis]  # np.where(evidence.previous_vocalics_from_self[:, t] >= 0, 1, 0)[:, np.newaxis]

            # Vo (n x k) will have the values of vocalics from other speaker per trial
            B = latent_vocalics[range(evidence.num_trials), :, evidence.previous_vocalics_from_other[:, t]]

            # Mask with 1 in the cells in which there are previous vocalics from another speaker and 0 otherwise
            Mb = evidence.previous_vocalics_from_other_mask[:, t][:,
                 np.newaxis]  # np.where(evidence.previous_vocalics_from_other[:, t] >= 0, 1, 0)[:, np.newaxis]

            # Use variance from prior if there are no previous vocalics from any speaker
            v = np.where((1 - Ma) * (1 - Mb) == 1, sa, saa)

            # Clipping has no effect in models that do not sample coordination outside the range [0, 1].
            means = (B - A * Ma) * Mb * clip_coordination(C) + A * Ma

            # Do not count LL if no speaker is talking at time t (mask M tells us that)
            ll += (norm(loc=means, scale=np.sqrt(v)).logpdf(V) * M).sum()

            # LL from latent to observed vocalics
            ll += (norm(loc=V, scale=so).logpdf(evidence.observed_vocalics[:, :, t]) * M).sum()

        return ll

    def _compute_coordination_loglikelihood(self, evidence: LatentVocalicsDataset) -> float:
        raise NotImplementedError

    @staticmethod
    def _get_latent_vocalics_term_for_coordination_posterior_unormalized_logprob(
            proposed_coordination_sample: np.ndarray,
            saa: float,
            evidence: LatentVocalicsDataset,
            latent_vocalics: np.ndarray,
            time_step: int) -> np.ndarray:

        V = latent_vocalics[..., time_step]

        previous_self_time_steps = evidence.previous_vocalics_from_self[:, time_step]
        previous_other_time_steps = evidence.previous_vocalics_from_other[:, time_step]
        A = np.take_along_axis(latent_vocalics, previous_self_time_steps[:, np.newaxis, np.newaxis], axis=-1)[..., 0]
        B = np.take_along_axis(latent_vocalics, previous_other_time_steps[:, np.newaxis, np.newaxis], axis=-1)[..., 0]

        M = evidence.vocalics_mask[:, time_step][:, np.newaxis]
        Ma = evidence.previous_vocalics_from_self_mask[:, time_step][:, np.newaxis]
        Mb = evidence.previous_vocalics_from_other_mask[:, time_step][:, np.newaxis]

        mean = ((B - A * Ma) * clip_coordination(proposed_coordination_sample) * Mb + A * Ma) * M * Mb

        # Coordination only affects vocalics in times in which there's a dependency on a previous other speaker
        log_posterior = (norm(loc=mean, scale=saa).logpdf(V) * M * Mb).sum(axis=1)

        return log_posterior

    def _update_parameters(self, evidence: LatentVocalicsDataset,
                           train_hyper_parameters: LatentVocalicsTrainingHyperParameters,
                           pool: Pool):

        self._update_coordination_parameters(evidence, train_hyper_parameters)

        # Variance of the latent vocalics prior
        if not self.parameters.var_a_frozen:
            V = self.last_latent_vocalics_samples_
            M = evidence.vocalics_mask

            first_time_steps = np.argmax(M, axis=1)
            first_latent_vocalics = np.take_along_axis(V, first_time_steps[:, np.newaxis, np.newaxis], axis=-1)
            M_first_time_steps = np.take_along_axis(M, first_time_steps[:, np.newaxis], axis=-1)

            a = train_hyper_parameters.a_va + M_first_time_steps.sum() * self.num_vocalic_features / 2
            b = train_hyper_parameters.b_va + np.square(first_latent_vocalics).sum() / 2

            new_var_a = invgamma(a=a, scale=b).mean()
            self.parameters.set_var_a(new_var_a, freeze=False)

        # Variance of the latent vocalics transition
        if not self.parameters.var_aa_frozen:
            coordination = np.expand_dims(self.last_coordination_samples_, axis=1)
            V = self.last_latent_vocalics_samples_
            M = np.expand_dims(evidence.vocalics_mask, axis=1)

            Ma = np.expand_dims(np.where(evidence.previous_vocalics_from_self == -1, 0, 1), axis=1)
            Mb = np.expand_dims(np.where(evidence.previous_vocalics_from_other == -1, 0, 1), axis=1)

            # All times in which there were vocalics and either previous vocalics from the same speaker or a different
            # one
            Ml = (M * (1 - (1 - Ma) * (1 - Mb)))
            A = np.take_along_axis(V, np.expand_dims(evidence.previous_vocalics_from_self, axis=1), axis=-1)
            B = np.take_along_axis(V, np.expand_dims(evidence.previous_vocalics_from_other, axis=1), axis=-1)

            m = (B - A * Ma) * clip_coordination(coordination) * Mb + A * Ma
            x = (V - m) * Ml

            a = train_hyper_parameters.a_vaa + Ml.sum() * self.num_vocalic_features / 2
            b = train_hyper_parameters.b_vaa + np.square(x).sum() / 2

            new_var_aa = invgamma(a=a, scale=b).mean()
            self.parameters.set_var_aa(new_var_aa, freeze=False)

        # Variance of the vocalics emission
        if not self.parameters.var_o_frozen:
            V = self.last_latent_vocalics_samples_
            M = np.expand_dims(evidence.vocalics_mask, axis=1)

            x = (V - evidence.observed_vocalics) * M
            a = train_hyper_parameters.a_vo + M.sum() * self.num_vocalic_features / 2
            b = train_hyper_parameters.b_vo + np.square(x).sum() / 2

            new_var_o = invgamma(a=a, scale=b).mean()
            self.parameters.set_var_o(new_var_o, freeze=False)

    def _update_coordination_parameters(self, evidence: LatentVocalicsDataset,
                                        train_hyper_parameters: LatentVocalicsTrainingHyperParameters):
        raise NotImplementedError

    def _log_parameters(self, gibbs_step: int, logger: BaseLogger):
        logger.add_scalar("train/var_c", self.parameters.var_c, gibbs_step)
        logger.add_scalar("train/var_a", self.parameters.var_a, gibbs_step)
        logger.add_scalar("train/var_aa", self.parameters.var_aa, gibbs_step)
        logger.add_scalar("train/var_o", self.parameters.var_o, gibbs_step)

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
    def _predict_init(self, evidence: LatentVocalicsDataset, num_particles: int, seed: Optional[int],
                      num_jobs: int = 1):
        # Self-dependency can be toggled by just modifying some variables in the dataset. We do this here and change the
        # dataset to its original configuration in the end of the fit method.
        if self.disable_self_dependency:
            evidence.disable_self_dependency()

    def _predict_end(self, evidence: LatentVocalicsDataset, num_particles: int, seed: Optional[int], num_jobs: int = 1):
        if self.disable_self_dependency:
            evidence.enable_self_dependency()

    def _sample_from_prior(self, num_particles: int, series: LatentVocalicsDataSeries) -> Particles:
        new_particles = self._create_new_particles()

        self._sample_coordination_from_prior(num_particles, new_particles, series)

        if series.latent_vocalics is None:
            self._sample_vocalics_from_prior(num_particles, series, new_particles)
        else:
            new_particles.latent_vocalics = {subject: None for subject in series.observed_vocalics.subjects}
            if series.observed_vocalics.mask[0] == 1:
                speaker = series.observed_vocalics.utterances[0].subject_id
                new_particles.latent_vocalics[speaker] = np.ones((num_particles, 1)) * series.latent_vocalics.values[:,
                                                                                       0]
        return new_particles

    def _sample_coordination_from_prior(self, num_particles: int, new_particles: LatentVocalicsParticles,
                                        series: LatentVocalicsDataSeries):
        raise NotImplementedError

    def _sample_vocalics_from_prior(self, num_particles: int, series: LatentVocalicsDataSeries,
                                    new_particles: LatentVocalicsParticles):
        new_particles.latent_vocalics = {subject: None for subject in series.observed_vocalics.subjects}
        if series.observed_vocalics.mask[0] == 1:
            speaker = series.observed_vocalics.utterances[0].subject_id
            mean = np.zeros((num_particles, series.observed_vocalics.num_features))
            new_particles.latent_vocalics[speaker] = norm(loc=mean, scale=np.sqrt(self.parameters.var_a)).rvs()

    def _sample_from_transition_to(self, time_step: int, num_particles: int, states: List[LatentVocalicsParticles],
                                   series: LatentVocalicsDataSeries) -> LatentVocalicsParticles:

        new_particles = self._create_new_particles()

        self._sample_coordination_from_transition_to(time_step, states, new_particles, series)

        if series.latent_vocalics is None:
            self._sample_vocalics_from_transition_to(time_step, states, new_particles, series)
        else:
            new_particles.latent_vocalics = states[time_step - 1].latent_vocalics.copy()
            if series.observed_vocalics.mask[time_step] == 1:
                speaker = series.observed_vocalics.utterances[time_step].subject_id
                new_particles.latent_vocalics[speaker] = np.ones((num_particles, 1)) * series.latent_vocalics.values[:,
                                                                                       time_step]

        return new_particles

    def _sample_coordination_from_transition_to(self, time_step: int, states: List[LatentVocalicsParticles],
                                                new_particles: LatentVocalicsParticles,
                                                series: LatentVocalicsDataSeries):
        raise NotImplementedError

    def _sample_vocalics_from_transition_to(self, time_step: int, states: List[LatentVocalicsParticles],
                                            new_particles: LatentVocalicsParticles, series: LatentVocalicsDataSeries):
        previous_particles = states[time_step - 1]
        new_particles.latent_vocalics = previous_particles.latent_vocalics.copy()

        if series.observed_vocalics.mask[time_step] == 1:
            speaker = series.observed_vocalics.utterances[time_step].subject_id

            if series.observed_vocalics.previous_from_self[time_step] is None or self.disable_self_dependency:
                # Mean of the prior distribution
                num_particles = len(new_particles.coordination)
                A = np.zeros((num_particles, self.num_vocalic_features))
            else:
                # Sample from dependency on previous vocalics from the same speaker
                A = self.f(previous_particles.latent_vocalics[speaker], 0)

            if series.observed_vocalics.previous_from_other[time_step] is None:
                if series.observed_vocalics.previous_from_self[time_step] is None or self.disable_self_dependency:
                    # Sample from prior
                    new_particles.latent_vocalics[speaker] = norm(loc=A, scale=np.sqrt(self.parameters.var_a)).rvs()
                else:
                    # Sample from dependency on previous vocalics from the same speaker
                    new_particles.latent_vocalics[speaker] = norm(loc=A, scale=np.sqrt(self.parameters.var_aa)).rvs()
            else:
                C = new_particles.coordination[:, np.newaxis]
                other_speaker = series.observed_vocalics.utterances[
                    series.observed_vocalics.previous_from_other[time_step]].subject_id
                B = self.f(previous_particles.latent_vocalics[other_speaker], 1)
                mean = (B - A) * clip_coordination(C) + A
                new_particles.latent_vocalics[speaker] = norm(loc=mean, scale=np.sqrt(self.parameters.var_aa)).rvs()

    def _calculate_evidence_log_likelihood_at(self, time_step: int, states: List[LatentVocalicsParticles],
                                              series: LatentVocalicsDataSeries):
        if series.coordination is not None and series.latent_vocalics is not None:
            num_particles = len(states[time_step].coordination)
            return np.zeros(num_particles)

        if series.latent_vocalics is None:
            # Observed vocalics as evidence for latent vocalics
            if series.observed_vocalics.mask[time_step] == 1:
                speaker = series.observed_vocalics.utterances[time_step].subject_id
                Vt = states[time_step].latent_vocalics[speaker]
                Ot = series.observed_vocalics.values[:, time_step]

                if np.ndim(Vt) > 1:
                    # particles x features
                    log_likelihoods = norm(loc=self.g(Vt), scale=np.sqrt(self.parameters.var_o)).logpdf(Ot).sum(axis=1)
                else:
                    # 1 particle only when ground truth for the latent vocalics is provided
                    log_likelihoods = norm(loc=self.g(Vt), scale=np.sqrt(self.parameters.var_o)).logpdf(Ot).sum()
            else:
                log_likelihoods = np.zeros(len(states[time_step].coordination))
        else:
            # Latent vocalics is evidence for coordination
            previous_time_step_from_other = series.observed_vocalics.previous_from_other[time_step]
            if series.observed_vocalics.mask[time_step] == 1 and previous_time_step_from_other is not None:

                # # Coordination only plays a whole in latent vocalics when there's previous vocalics from a different
                # # speaker.
                if series.observed_vocalics.previous_from_self[time_step] is None:
                    # Mean of the prior distribution
                    A = 0
                else:
                    # Sample from dependency on previous vocalics from the same speaker
                    A = series.latent_vocalics.values[:, series.observed_vocalics.previous_from_self[time_step]]

                B = series.latent_vocalics.values[:, previous_time_step_from_other]
                mean = (B - A) * clip_coordination(states[time_step].coordination[:, np.newaxis]) + A

                Vt = series.latent_vocalics.values[:, time_step]
                log_likelihoods = norm(loc=mean, scale=np.sqrt(self.parameters.var_aa)).logpdf(Vt).sum(axis=1)
            else:
                log_likelihoods = np.zeros(len(states[time_step].coordination))

        return log_likelihoods

    def _resample_at(self, time_step: int, series: LatentVocalicsDataSeries):
        if series.is_complete:
            return False

        if series.latent_vocalics is None:
            return series.observed_vocalics.mask[time_step] == 1
        else:
            # Only coordination is latent, but we only need to sample it if there's a link between the
            # current vocalics and previous vocalics from a different speaker.
            return series.observed_vocalics.mask[time_step] == 1 and series.observed_vocalics.previous_from_other[
                time_step] is not None

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return LatentVocalicsParticles()

    def _summarize_particles(self, series: LatentVocalicsDataSeries,
                             particles: List[LatentVocalicsParticles]) -> LatentVocalicsParticlesSummary:

        summary = LatentVocalicsParticlesSummary()
        summary.coordination_mean = np.zeros(series.num_time_steps)
        summary.coordination_var = np.zeros(series.num_time_steps)
        summary.latent_vocalics_mean = np.zeros((series.num_vocalic_features, series.num_time_steps))
        summary.latent_vocalics_var = np.zeros((series.num_vocalic_features, series.num_time_steps))

        for t, particles_in_time in enumerate(particles):
            summary.coordination_mean[t] = particles_in_time.coordination.mean()
            summary.coordination_var[t] = particles_in_time.coordination.var()

            if series.observed_vocalics.mask[t] == 1:
                speaker = series.observed_vocalics.utterances[t].subject_id
                summary.latent_vocalics_mean[:, t] = particles_in_time.latent_vocalics[speaker].mean(axis=0)
                summary.latent_vocalics_var[:, t] = particles_in_time.latent_vocalics[speaker].var(axis=0)
            else:
                if t > 0:
                    summary.latent_vocalics_mean[:, t] = summary.latent_vocalics_mean[:, t - 1]
                    summary.latent_vocalics_var[:, t] = summary.latent_vocalics_var[:, t - 1]

        return summary

    # TODO: Consider using to parallelize latent vocalics training.
    # @staticmethod
    # def _get_time_step_blocks_for_parallel_fitting(evidence: LatentVocalicsDataset, num_jobs: int) -> Tuple[
    #     List[np.ndarray], List[np.ndarray]]:
    #     """
    #     Analyze the dependencies in the model to create 2 groups with lists of time steps that can be processed in
    #     parallel in each group.
    #     """
    #
    #     num_effective_jobs = int(min(evidence.num_time_steps / 2, num_jobs))
    #     if num_effective_jobs == 1:
    #         # No parallel jobs. Variables from all time steps are assigned to the first group.
    #         return [np.arange(evidence.num_time_steps)], []
    #
    #     # Vocalics in a time step depends on two previous vocalics from different time steps: one from the same
    #     # speaker, and one from a different speaker. Therefore, we cannot simply add the border of the blocks to
    #     # the list of time steps in the first group.
    #     # The strategy here will be to first split the time steps in as many blocks as the number of jobs. These,
    #     # blocks will hold the list of time steps to process in the second group. For each block, we take the last
    #     # time step in which there was observation across the trials, and from there we take the next timestep from
    #     # the same speaker and other speaker. We then take the minimum and maximum among all these time
    #     # steps. That yields the earliest (a) and latest (b) time steps in the future the current block depends on.
    #     # We than adjust the block to be [start:a] + [b+1:end] and reserve time steps between a and b to the 1st
    #     # group. In the end, we will split the first group into multiple blocks that are not dependent on each other
    #     # which can be determined by jumps in time step larger than 1, meaning these blocks depend on blocks in the
    #     # second group, and can, therefore, run in parallel.
    #     time_chunks = np.array_split(np.arange(evidence.num_time_steps), num_effective_jobs)
    #     masks = np.array_split(evidence.vocalics_mask, num_effective_jobs, axis=-1)
    #
    #     # There's always a dependency between the last time step of a block and the first time step of the next
    #     # block because coordination at time t+1 depends on coordination at time t. So, we start by moving
    #     # the first element of each block (starting from the second) to the list of time steps in the second group.
    #     second_group_time_steps = []
    #     time_steps_reserved_for_1st_group = []
    #     for i in range(1, num_effective_jobs):
    #         time_steps_reserved_for_1st_group.append(time_chunks[i][0])
    #         time_chunks[i] = time_chunks[i][1:]
    #         masks[i] = masks[i][:, 1:]
    #
    #     # For indexes in which the next speaker does not exist, there's no dependency with the future. We keep two
    #     # matrices, one with minimum value (-1) and one with maximum value (T). We will use this to determine the
    #     # smallest and largest indices in the future, the current block depends on.
    #     next_time_steps_from_self_min = evidence.next_vocalics_from_self
    #     next_time_steps_from_self_max = np.where(evidence.next_vocalics_from_self == -1, evidence.num_time_steps,
    #                                              evidence.next_vocalics_from_self)
    #     next_time_steps_from_other_min = evidence.next_vocalics_from_other
    #     next_time_steps_from_other_max = np.where(evidence.next_vocalics_from_other == -1, evidence.num_time_steps,
    #                                               evidence.next_vocalics_from_other)
    #     for i in range(len(time_chunks) - 1):
    #         block_size = len(time_chunks[i])
    #
    #         if block_size > 0:
    #             second_group_time_steps.append(time_chunks[i])
    #
    #             # Last indexes where M[j] = 1 per column. We use this to identify last indices in the block where
    #             # someone talked
    #             last_indices_with_speaker = block_size - np.argmax(np.flip(masks[i], axis=1), axis=1) - 1
    #             last_times_with_speaker = time_chunks[i][last_indices_with_speaker]
    #
    #             next_block_time_step_self_min = np.take_along_axis(next_time_steps_from_self_min,
    #                                                                last_times_with_speaker[:, np.newaxis], axis=-1)
    #             next_block_time_step_other_min = np.take_along_axis(next_time_steps_from_other_min,
    #                                                                 last_times_with_speaker[:, np.newaxis], axis=-1)
    #             next_block_time_step_self_max = np.take_along_axis(next_time_steps_from_self_max,
    #                                                                last_times_with_speaker[:, np.newaxis], axis=-1)
    #             next_block_time_step_other_max = np.take_along_axis(next_time_steps_from_other_max,
    #                                                                 last_times_with_speaker[:, np.newaxis], axis=-1)
    #
    #             # Because the _min variables contain -1 when no one is talking, the maximum is always going to
    #             # return the maximum time step among all trials in the block. Or -1 if no one is talking in that
    #             # block in all trials. Similar logic is applied for the _max case.
    #             first_next_dependent_time_step = np.minimum(np.min(next_block_time_step_self_max),
    #                                                         np.min(next_block_time_step_other_max))
    #             last_next_dependent_time_step = np.maximum(np.max(next_block_time_step_self_min),
    #                                                        np.max(next_block_time_step_other_min))
    #
    #             # Find next non-zero block
    #             j = i + 1
    #             while j < len(time_chunks):
    #                 if len(time_chunks[j]) > 0:
    #                     break
    #                 j += 1
    #
    #             if last_next_dependent_time_step < time_chunks[j][0]:
    #                 # There is no dependency with the future or there's a dependency with time steps already
    #                 # added to the single thread list. In that case, the current block can just be added to the
    #                 # parallel processing list.
    #                 continue
    #
    #             # There is a dependency with the future. It might be the case that some elements in the range
    #             # below are already in the 1st group list if the dependencies are from one time step to another
    #             # since we already added the first time steps of every block to the list in the beginning of this
    #             # function. We will remove duplicates and sort the array in the end to address this case.
    #             independent_range = np.arange(first_next_dependent_time_step, last_next_dependent_time_step + 1)
    #             time_steps_reserved_for_1st_group.extend(independent_range)
    #
    #             # We adjust the next blocks to remove the time steps that we added to the 1st group.
    #             for j in range(i + 1, len(time_chunks)):
    #                 if len(time_chunks[j]) == 0 or time_chunks[j][-1] < first_next_dependent_time_step:
    #                     # The dependency is a block further in the future. So, we do nothing.
    #                     continue
    #
    #                 if first_next_dependent_time_step < time_chunks[j][0]:
    #                     first_next_dependent_time_step = time_chunks[j][0]
    #
    #                 stop = False
    #                 if last_next_dependent_time_step <= time_chunks[j][-1]:
    #                     stop = True
    #
    #                 # We remove time steps in the 1st group from the 2nd group block.
    #                 block1 = np.arange(time_chunks[j][0], first_next_dependent_time_step)
    #                 block2 = np.arange(np.minimum(time_chunks[j][-1], last_next_dependent_time_step) + 1,
    #                                    time_chunks[j][-1] + 1)
    #
    #                 time_chunks[j] = np.concatenate([block1, block2])
    #                 masks[j] = np.take_along_axis(evidence.vocalics_mask,
    #                                               indices=time_chunks[j][np.newaxis, :],
    #                                               axis=-1)
    #
    #                 if stop:
    #                     break
    #
    #     if len(time_chunks[-1]) > 0:
    #         # Include the last block if it is not empty
    #         second_group_time_steps.append(time_chunks[-1])
    #
    #     if len(second_group_time_steps) == 1:
    #         # If there's only one block in the 2nd group, we just run everything in a single group.
    #         time_steps_reserved_for_1st_group.extend(second_group_time_steps[0])
    #         time_steps_reserved_for_1st_group = np.array(time_steps_reserved_for_1st_group)
    #         second_group_time_steps = []
    #
    #     # Remove duplicates and sort time steps by ascending order
    #     time_steps_reserved_for_1st_group = sorted(list(set(time_steps_reserved_for_1st_group)))
    #
    #     # We can split the time steps in the first group into multiple blocks that don't depend on each other.
    #     # The boundaries are defined by subsequent time-steps with jumps larger than 1. This is because if the
    #     # difference is larger than one, it means there is another block that was added to the 2nd group already
    #     # and the dependencies of the blocks to be separated are with this 2nd-group block.
    #     # Also, notice that we only reserve one block to the 1st group per 2nd-group block processed, meaning that
    #     # we don't have to worry about the number of independent blocks in the 1st group, as it will never be more than
    #     # the desired number of jobs.
    #     first_group_blocks = []
    #     block = []
    #     for i, t in enumerate(time_steps_reserved_for_1st_group):
    #         if i == 0 or t - block[-1] == 1:
    #             block.append(t)
    #         else:
    #             first_group_blocks.append(np.array(block))
    #             block = [t]
    #
    #     first_group_blocks.append(np.array(block))
    #
    #     return first_group_blocks, second_group_time_steps
