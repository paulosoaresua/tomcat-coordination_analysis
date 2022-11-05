from __future__ import annotations
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from datetime import datetime

import numpy as np
from scipy.stats import norm, invgamma
from tqdm import tqdm

from coordination.common.log import BaseLogger
from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.inference.mcmc import MCMC
from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries
from coordination.model.particle_filter import Particles
from coordination.model.pgm import PGM, ParticlesSummary, Samples


class LatentVocalicsParticles(Particles):
    coordination: np.ndarray
    latent_vocalics: Dict[str, np.ndarray]

    def _keep_particles_at(self, indices: np.ndarray):
        if isinstance(self.coordination, np.ndarray):
            # otherwise, coordination is given and it will be a single number
            self.coordination = self.coordination[indices]

        for speaker, latent_vocalics in self.latent_vocalics.items():
            if latent_vocalics is not None:
                if np.ndim(latent_vocalics) > 1:
                    # otherwise, coordination is given and it will be a single number
                    self.latent_vocalics[speaker] = latent_vocalics[indices, :]


class LatentVocalicsParticlesSummary(ParticlesSummary):
    coordination_mean: np.ndarray
    coordination_var: np.ndarray
    latent_vocalics_mean: np.ndarray
    latent_vocalics_var: np.ndarray


class LatentVocalicsSamples(Samples):
    coordination: np.ndarray
    latent_vocalics: List[VocalicsSparseSeries]
    observed_vocalics: List[VocalicsSparseSeries]

    @property
    def size(self):
        return len(self.observed_vocalics)


class LatentVocalicsDataSeries(EvidenceDataSeries):

    def __init__(self, uuid: str, observed_vocalics: VocalicsSparseSeries, coordination: Optional[np.ndarray] = None,
                 latent_vocalics: VocalicsSparseSeries = None):
        super().__init__(uuid)
        self.coordination = coordination
        self.latent_vocalics = latent_vocalics
        self.observed_vocalics = observed_vocalics

    @property
    def is_complete(self) -> bool:
        return self.coordination is not None and self.latent_vocalics is not None

    @property
    def num_time_steps(self):
        return self.observed_vocalics.num_time_steps

    @property
    def num_vocalic_features(self):
        return self.observed_vocalics.num_features


class LatentVocalicsDataset(EvidenceDataset):

    def __init__(self, series: List[LatentVocalicsDataSeries]):
        super().__init__(series)

        # Keep a matrix representation of the data for fast processing during training
        self.coordination = None if series[0].coordination is None else np.zeros(
            (len(series), series[0].num_time_steps))

        # n (num samples) x k (num features) x T (num time steps)
        self.latent_vocalics = None if series[0].latent_vocalics is None else np.zeros(
            (len(series), series[0].num_vocalic_features, series[0].num_time_steps))

        self.observed_vocalics = np.zeros((len(series), series[0].num_vocalic_features, series[0].num_time_steps))
        self.vocalics_mask = np.zeros((len(series), series[0].num_time_steps))
        self.previous_vocalics_from_self = np.zeros((len(series), series[0].num_time_steps)).astype(np.int)
        self.previous_vocalics_from_other = np.zeros((len(series), series[0].num_time_steps)).astype(np.int)
        self.next_vocalics_from_self = np.ones((len(series), series[0].num_time_steps)).astype(np.int) * -1
        self.next_vocalics_from_other = np.ones((len(series), series[0].num_time_steps)).astype(np.int) * -1

        for i, series in enumerate(series):
            if series.coordination is not None:
                self.coordination[i] = series.coordination

            if series.latent_vocalics is not None:
                self.latent_vocalics[i] = series.latent_vocalics.values

            self.observed_vocalics[i] = series.observed_vocalics.values
            self.vocalics_mask[i] = series.observed_vocalics.mask
            self.previous_vocalics_from_self[i] = np.array(
                [-1 if t is None else t for t in series.observed_vocalics.previous_from_self])
            self.previous_vocalics_from_other[i] = np.array(
                [-1 if t is None else t for t in series.observed_vocalics.previous_from_other])
            self.previous_vocalics_from_self_mask = np.where(self.previous_vocalics_from_self >= 0, 1, 0)
            self.previous_vocalics_from_other_mask = np.where(self.previous_vocalics_from_other >= 0, 1, 0)

            for t in range(series.num_time_steps):
                if self.previous_vocalics_from_self[i, t] >= 0:
                    self.next_vocalics_from_self[i, self.previous_vocalics_from_self[i, t]] = t

                if self.previous_vocalics_from_other[i, t] >= 0:
                    self.next_vocalics_from_other[i, self.previous_vocalics_from_other[i, t]] = t


def default_f(latent_vocalics: np.ndarray, speaker_mask: int) -> np.ndarray:
    return latent_vocalics


def default_g(latent_vocalics: np.ndarray) -> np.ndarray:
    return latent_vocalics


def clip_coordination(coordination: np.ndarray) -> np.ndarray:
    return np.clip(coordination, a_min=0, a_max=1)


SP = TypeVar('SP')
S = TypeVar('S')


class CoordinationBlendingLatentVocalics(PGM[SP, S]):

    def __init__(self,
                 initial_coordination: float,
                 num_vocalic_features: int,
                 num_speakers: int,
                 a_vcc: float,
                 b_vcc: float,
                 a_va: float,
                 b_va: float,
                 a_vaa: float,
                 b_vaa: float,
                 a_vo: float,
                 b_vo: float,
                 f: Callable = default_f,
                 g: Callable = default_g):
        super().__init__()

        self.initial_coordination = initial_coordination
        self.num_vocalic_features = num_vocalic_features
        self.num_speakers = num_speakers
        self.f = f
        self.g = g

        # Parameters of the model
        self.var_cc: Optional[float] = None
        self.var_a: Optional[np.ndarray] = None
        self.var_aa: Optional[np.ndarray] = None
        self.var_o: Optional[np.ndarray] = None

        # Parameters of the prior distributions (Inverse-Gamma) of the variances
        self.a_vcc = a_vcc
        self.b_vcc = b_vcc
        self.a_va = a_va
        self.b_va = b_va
        self.a_vaa = a_vaa
        self.b_vaa = b_vaa
        self.a_vo = a_vo
        self.b_vo = b_vo

        self._hyper_params = {
            "C0": initial_coordination,
            "#features": num_vocalic_features,
            "#speakers": num_speakers,
            "a_vcc": a_vcc,
            "b_vcc": b_vcc,
            "a_va": a_va,
            "b_va": b_va,
            "a_vaa": a_vaa,
            "b_vaa": b_vaa,
            "a_vo": a_vo,
            "b_vo": b_vo,
        }

        # Samples collected during training
        self.vcc_samples_ = np.array([])
        self.va_samples_ = np.array([])
        self.vaa_samples_ = np.array([])
        self.vo_samples_ = np.array([])
        self.coordination_samples_ = np.array([])
        self.latent_vocalics_samples_ = np.array([])

    def reset_parameters(self):
        self.var_cc = None
        self.var_a = None
        self.var_aa = None
        self.var_o = None

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
            if previous_self is None:
                distribution = norm(loc=np.zeros(self.num_vocalic_features), scale=np.sqrt(self.var_a))
            else:
                distribution = norm(loc=previous_self, scale=np.sqrt(self.var_aa))
        else:
            if previous_self is None:
                D = previous_other
                distribution = norm(loc=D * clip_coordination(coordination), scale=np.sqrt(self.var_aa))
            else:
                D = previous_other - previous_self
                distribution = norm(loc=D * clip_coordination(coordination) + previous_self, scale=np.sqrt(self.var_aa))

        return distribution.rvs()

    def _sample_observed_vocalics(self, latent_vocalics: np.array) -> np.ndarray:
        return norm(loc=latent_vocalics, scale=np.sqrt(self.var_o)).rvs()

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------

    def _initialize_gibbs(self, burn_in: int, evidence: LatentVocalicsDataset):
        super()._initialize_gibbs(burn_in, evidence)

        # History of samples in each Gibbs step
        self.vcc_samples_ = np.zeros(burn_in + 1)
        self.va_samples_ = np.zeros(burn_in + 1)
        self.vaa_samples_ = np.zeros(burn_in + 1)
        self.vo_samples_ = np.zeros(burn_in + 1)
        self.coordination_samples_ = np.zeros((burn_in + 1, evidence.num_trials, evidence.num_time_steps))
        self.latent_vocalics_samples_ = np.zeros(
            (burn_in + 1, evidence.num_trials, self.num_vocalic_features, evidence.num_time_steps))

        # 1. Latent variables and parameter initialization
        self._initialize_coordination_for_gibbs(evidence)

        if evidence.latent_vocalics is None:
            self.latent_vocalics_samples_[0] = norm(
                loc=np.zeros((evidence.num_trials, self.num_vocalic_features, evidence.num_time_steps)),
                scale=1).rvs()
            if burn_in > 0:
                self.latent_vocalics_samples_[1] = self.latent_vocalics_samples_[0]
        else:
            self.latent_vocalics_samples_[0] = evidence.latent_vocalics

        if self.var_cc is None:
            self.vcc_samples_[0] = invgamma(a=self.a_vcc, scale=self.b_vcc).rvs()
            if burn_in > 0:
                self.vcc_samples_[1] = self.vcc_samples_[0]
        else:
            self.vcc_samples_[:] = self.var_cc

        if self.var_a is None:
            self.va_samples_[0] = invgamma(a=self.a_va, scale=self.b_va).rvs()
            if burn_in > 0:
                self.va_samples_[1] = self.va_samples_[0]
        else:
            self.va_samples_[:] = self.var_a

        if self.var_aa is None:
            self.vaa_samples_[0] = invgamma(a=self.a_vaa, scale=self.b_vaa).rvs()
            if burn_in > 0:
                self.vaa_samples_[1] = self.vaa_samples_[0]
        else:
            self.vaa_samples_[:] = self.var_aa

        if self.var_o is None:
            self.vo_samples_[0] = invgamma(a=self.a_vo, scale=self.b_vo).rvs()
            if burn_in > 0:
                self.vo_samples_[1] = self.vo_samples_[0]
        else:
            self.vo_samples_[:] = self.var_o

    def _initialize_coordination_for_gibbs(self, evidence: LatentVocalicsDataset):
        raise NotImplementedError

    def _compute_joint_loglikelihood_at(self, gibbs_step: int, evidence: LatentVocalicsDataset) -> float:
        sa = np.sqrt(self.va_samples_[gibbs_step])
        saa = np.sqrt(self.vaa_samples_[gibbs_step])
        so = np.sqrt(self.vo_samples_[gibbs_step])

        coordination = self.coordination_samples_[gibbs_step]
        latent_vocalics = self.latent_vocalics_samples_[gibbs_step]

        ll = self._compute_coordination_likelihood(gibbs_step, evidence)
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
            Ma = np.where(evidence.previous_vocalics_from_self[:, t] >= 0, 1, 0)[:, np.newaxis]

            # Vo (n x k) will have the values of vocalics from other speaker per trial
            B = latent_vocalics[range(evidence.num_trials), :, evidence.previous_vocalics_from_other[:, t]]

            # Mask with 1 in the cells in which there are previous vocalics from another speaker and 0 otherwise
            Mb = np.where(evidence.previous_vocalics_from_other[:, t] >= 0, 1, 0)[:, np.newaxis]

            # Use variance from prior if there are no previous vocalics from any speaker
            v = np.where((1 - Ma) * (1 - Mb) == 1, sa, saa)

            # Clipping has no effect in models that do not sample coordination outside the range [0, 1].
            means = (B - A * Ma) * Mb * clip_coordination(C) + A * Ma

            # Do not count LL if no speaker is talking at time t (mask M tells us that)
            ll += (norm(loc=means, scale=np.sqrt(v)).logpdf(V) * M).sum()

            # LL from latent to observed vocalics
            ll += (norm(loc=V, scale=so).logpdf(evidence.observed_vocalics[:, :, t]) * M).sum()

        return ll

    def _compute_coordination_likelihood(self, gibbs_step: int, evidence: LatentVocalicsDataset) -> float:
        raise NotImplementedError

    def _gibbs_step(self, gibbs_step: int, evidence: LatentVocalicsDataset, time_steps: np.ndarray, job_num: int):
        coordination, extra_variables = self._sample_coordination_on_fit(gibbs_step, evidence, time_steps, job_num)

        if evidence.latent_vocalics is None:
            latent_vocalics = self._sample_latent_vocalics_on_fit(coordination, gibbs_step, evidence, time_steps,
                                                                  job_num)
        else:
            latent_vocalics = self.latent_vocalics_samples_[gibbs_step - 1].copy()

        return coordination, latent_vocalics, extra_variables

    def _sample_coordination_on_fit(self, gibbs_step: int, evidence: LatentVocalicsDataset, time_steps: np.ndarray,
                                    job_num: int) -> Tuple[np.ndarray, ...]:
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

        mean = ((B - A * Ma) * clip_coordination(proposed_coordination_sample) * Mb + A * Ma) * M

        log_posterior = (norm(loc=mean, scale=saa).logpdf(V) * M).sum(axis=1)

        return log_posterior

    def _sample_latent_vocalics_on_fit(self, coordination: np.ndarray, gibbs_step: int, evidence: LatentVocalicsDataset,
                                       time_steps: np.ndarray, job_num: int) -> np.ndarray:

        va = self.va_samples_[gibbs_step]
        vaa = self.vaa_samples_[gibbs_step]
        vo = self.vo_samples_[gibbs_step]

        latent_vocalics = self.latent_vocalics_samples_[gibbs_step].copy()
        for t in tqdm(time_steps, desc="Sampling Latent Vocalics", position=job_num, leave=False):
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

        return latent_vocalics

    def _retain_samples_from_latent(self, gibbs_step: int, latents: Any, time_steps: np.ndarray):
        """
        The only latent variable return by _gibbs_step is a state variable
        """
        self.coordination_samples_[gibbs_step][:, time_steps] = latents[0][:, time_steps]
        self.latent_vocalics_samples_[gibbs_step][:, :, time_steps] = latents[1][:, :, time_steps]

        if gibbs_step < self.latent_vocalics_samples_.shape[0] - 1:
            # Copy the current estimation to the next step. This is necessary for the parallelization to work properly.
            # This allows the blocks in the next step to use the values of the previous step (gibbs_step) as a start
            # point but without indexing the previous step. This is necessary because the execution block that runs in a
            # single thread, will update the latent values for the current step. If the parallel block execution
            # indexes the past,they won't have access to the most up-to-date values.
            self.coordination_samples_[gibbs_step + 1] = self.coordination_samples_[gibbs_step]
            self.latent_vocalics_samples_[gibbs_step + 1] = self.latent_vocalics_samples_[gibbs_step]

    def _update_latent_parameters(self, gibbs_step: int, evidence: LatentVocalicsDataset, logger: BaseLogger):
        self._update_latent_parameters_coordination(gibbs_step, evidence, logger)

        if gibbs_step < len(self.vcc_samples_) - 1:
            self.vcc_samples_[gibbs_step + 1] = self.vcc_samples_[gibbs_step]

        # Variance of the latent vocalics prior
        if self.var_a is None:
            V = self.latent_vocalics_samples_[gibbs_step]
            M = evidence.vocalics_mask

            first_time_steps = np.argmax(M, axis=1)
            first_latent_vocalics = np.take_along_axis(V, first_time_steps[:, np.newaxis, np.newaxis], axis=-1)
            M_first_time_steps = np.take_along_axis(M, first_time_steps[:, np.newaxis], axis=-1)

            a = self.a_va + M_first_time_steps.sum() * self.num_vocalic_features / 2
            b = self.b_va + np.square(first_latent_vocalics).sum() / 2
            self.va_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()

            if self.va_samples_[gibbs_step] == np.nan:
                self.va_samples_[gibbs_step] = np.inf

            if gibbs_step < len(self.va_samples_) - 1:
                self.va_samples_[gibbs_step + 1] = self.va_samples_[gibbs_step]

        # Variance of the latent vocalics transition
        if self.var_aa is None:
            coordination = np.expand_dims(self.coordination_samples_[gibbs_step], axis=1)
            V = self.latent_vocalics_samples_[gibbs_step]
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

            a = self.a_vaa + Ml.sum() * self.num_vocalic_features / 2
            b = self.b_vaa + np.square(x).sum() / 2
            self.vaa_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()

            if self.vaa_samples_[gibbs_step] == np.nan:
                self.vaa_samples_[gibbs_step] = np.inf

            if gibbs_step < len(self.vaa_samples_) - 1:
                self.vaa_samples_[gibbs_step + 1] = self.vaa_samples_[gibbs_step]

        # Variance of the vocalics emission
        if self.var_o is None:
            V = self.latent_vocalics_samples_[gibbs_step]
            M = np.expand_dims(evidence.vocalics_mask, axis=1)

            x = (V - evidence.observed_vocalics) * M
            a = self.a_vo + M.sum() * self.num_vocalic_features / 2
            b = self.b_vo + np.square(x).sum() / 2
            self.vo_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()
            if self.vo_samples_[gibbs_step] == np.nan:
                self.vo_samples_[gibbs_step] = np.inf

            if gibbs_step < len(self.vo_samples_) - 1:
                self.vo_samples_[gibbs_step + 1] = self.vo_samples_[gibbs_step]

        logger.add_scalar("train/var_cc", self.vcc_samples_[gibbs_step], gibbs_step)
        logger.add_scalar("train/var_a", self.va_samples_[gibbs_step], gibbs_step)
        logger.add_scalar("train/var_aa", self.vaa_samples_[gibbs_step], gibbs_step)
        logger.add_scalar("train/var_o", self.vo_samples_[gibbs_step], gibbs_step)

    def _update_latent_parameters_coordination(self, gibbs_step: int, evidence: EvidenceDataset, logger: BaseLogger):
        raise NotImplementedError

    def _retain_parameters(self):
        self.var_cc = self.vcc_samples_[-1]
        self.var_a = self.va_samples_[-1]
        self.var_aa = self.vaa_samples_[-1]
        self.var_o = self.vo_samples_[-1]

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
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
            new_particles.latent_vocalics[speaker] = norm(loc=mean, scale=np.sqrt(self.var_a)).rvs()

    def _sample_from_transition_to(self, time_step: int, num_particles: int, states: List[LatentVocalicsParticles],
                                   series: LatentVocalicsDataSeries) -> LatentVocalicsParticles:

        new_particles = self._create_new_particles()

        self._sample_coordination_from_transition_to(time_step, states, new_particles, series)

        if series.latent_vocalics is None:
            self._sample_vocalics_from_transition_to(time_step, states, new_particles, series)
        else:
            # Ground truth. Just use 1 particle.
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

            if series.observed_vocalics.previous_from_self[time_step] is None:
                # Mean of the prior distribution
                num_particles = len(new_particles.coordination)
                A = np.zeros((num_particles, self.num_vocalic_features))
            else:
                # Sample from dependency on previous vocalics from the same speaker
                A = self.f(previous_particles.latent_vocalics[speaker], 0)

            if series.observed_vocalics.previous_from_other[time_step] is None:
                if series.observed_vocalics.previous_from_self[time_step] is None:
                    # Sample from prior
                    new_particles.latent_vocalics[speaker] = norm(loc=A, scale=np.sqrt(self.var_a)).rvs()
                else:
                    # Sample from dependency on previous vocalics from the same speaker
                    new_particles.latent_vocalics[speaker] = norm(loc=A, scale=np.sqrt(self.var_aa)).rvs()
            else:
                C = new_particles.coordination[:, np.newaxis]
                other_speaker = series.observed_vocalics.utterances[
                    series.observed_vocalics.previous_from_other[time_step]].subject_id
                B = self.f(previous_particles.latent_vocalics[other_speaker], 1)
                mean = (B - A) * clip_coordination(C) + A
                new_particles.latent_vocalics[speaker] = norm(loc=mean, scale=np.sqrt(self.var_aa)).rvs()

    def _calculate_evidence_log_likelihood_at(self, time_step: int, states: List[LatentVocalicsParticles],
                                              series: LatentVocalicsDataSeries):
        if series.is_complete:
            return 0

        if series.latent_vocalics is None:
            # Observed vocalics as evidence for latent vocalics
            if series.observed_vocalics.mask[time_step] == 1:
                speaker = series.observed_vocalics.utterances[time_step].subject_id
                Vt = states[time_step].latent_vocalics[speaker]
                Ot = series.observed_vocalics.values[:, time_step]

                if np.ndim(Vt) > 1:
                    # particles x features
                    log_likelihoods = norm(loc=self.g(Vt), scale=np.sqrt(self.var_o)).logpdf(Ot).sum(axis=1)
                else:
                    # 1 particle only when ground truth for the latent vocalics is provided
                    log_likelihoods = norm(loc=self.g(Vt), scale=np.sqrt(self.var_o)).logpdf(Ot).sum()
            else:
                log_likelihoods = np.zeros(len(states[time_step].coordination))
        else:
            # Latent vocalics is evidence for coordination
            previous_time_step_from_other = series.observed_vocalics.previous_from_other[time_step]
            if series.observed_vocalics.mask[time_step] == 1 and previous_time_step_from_other is not None:

                # Coordination only plays a whole in latent vocalics when there's previous vocalics from a different
                # speaker.
                if series.observed_vocalics.previous_from_self[time_step] is None:
                    # Mean of the prior distribution
                    A = 0
                else:
                    # Sample from dependency on previous vocalics from the same speaker
                    A = series.latent_vocalics.values[:, series.observed_vocalics.previous_from_self[time_step]]

                B = series.latent_vocalics.values[:, previous_time_step_from_other]
                mean = (B - A) * clip_coordination(states[time_step].coordination[:, np.newaxis]) + A

                Vt = series.latent_vocalics.values[:, time_step]
                log_likelihoods = norm(loc=mean, scale=np.sqrt(self.var_aa)).logpdf(Vt).sum(axis=1)
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

    def _get_time_step_blocks_for_parallel_fitting(self, evidence: LatentVocalicsDataset, num_jobs: int):
        parallel_time_step_blocks = []
        single_thread_time_steps = []

        num_effective_jobs = min(evidence.num_time_steps / 2, num_jobs)
        if num_effective_jobs == 1:
            # No parallel jobs
            single_thread_time_steps = np.arange(evidence.num_time_steps)
        else:
            # A vocalics in a time step depends on two previous vocalics from different time steps: one from the same
            # speaker, and one from a different speaker. Therefore, we cannot simply add the border of the blocks in the
            # list of single-threaded time steps.
            # The strategy here will be to first split the time steps in as many blocks as the number of jobs. For each
            # block, we take the last time step in which there was observation across the trials, and from there we take
            # the next timestep from the same speaker and other speaker. We then take the maximum among all these time
            # steps. That yields the earliest time step in the next block, the current block depends on. We than take
            # the portion between the beginning of the next block and that computed time step to be part of the list of
            # single threaded time steps.
            time_chunks = np.array_split(np.arange(evidence.num_time_steps), num_effective_jobs)
            masks = np.array_split(evidence.vocalics_mask, num_effective_jobs, axis=-1)

            all_time_steps = np.arange(evidence.num_time_steps)[np.newaxis, :].repeat(evidence.num_trials, axis=0)

            # For indexes in which the next speaker does not exist, we replace with the current index. That is, there's
            # no dependency in the future
            next_time_steps_from_self = np.where(evidence.next_vocalics_from_self == -1, all_time_steps,
                                                 evidence.next_vocalics_from_self)
            next_time_steps_from_other = np.where(evidence.next_vocalics_from_other == -1, all_time_steps,
                                                  evidence.next_vocalics_from_other)
            j = 0
            while j < len(time_chunks) - 1:
                block_size = len(time_chunks[j])

                if block_size > 0:
                    # Last indexes where M[j] = 1 per column
                    last_indices_with_speaker = block_size - np.argmax(np.flip(masks[j], axis=1), axis=1) - 1
                    last_times_with_speaker = time_chunks[j][last_indices_with_speaker]
                    next_block_time_step_self = np.take_along_axis(next_time_steps_from_self,
                                                                   last_times_with_speaker[:, np.newaxis], axis=-1)
                    next_block_time_step_other = np.take_along_axis(next_time_steps_from_other,
                                                                    last_times_with_speaker[:, np.newaxis], axis=-1)
                    last_time_step_independent_block = np.maximum(np.max(next_block_time_step_self),
                                                                  np.max(next_block_time_step_other))

                    if last_time_step_independent_block > time_chunks[j][-1]:
                        # There is a dependency with the next block
                        independent_range = np.arange(time_chunks[j][-1] + 1, last_time_step_independent_block + 1)
                        single_thread_time_steps.extend(independent_range)

                    parallel_time_step_blocks.append(time_chunks[j])

                    # Sometimes, the dependency might be with time steps in a block that is not the subsequent one.
                    # The loop below will skip intermediary blocks until we find the one in which the dependent time
                    # step is.
                    while last_time_step_independent_block > time_chunks[j + 1][-1]:
                        j += 1

                    next_parallel_range = np.arange(last_time_step_independent_block + 1, time_chunks[j + 1][-1] + 1)

                    time_chunks[j + 1] = next_parallel_range

                j += 1

            if len(time_chunks[-1]) > 0:
                parallel_time_step_blocks.append(time_chunks[-1])

            if len(parallel_time_step_blocks) == 1:
                single_thread_time_steps.extend(parallel_time_step_blocks[0])
                single_thread_time_steps = np.array(single_thread_time_steps)
                parallel_time_step_blocks = []

        return parallel_time_step_blocks, single_thread_time_steps

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
