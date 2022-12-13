from __future__ import annotations
from typing import List

from multiprocessing import Pool
import pickle

import numpy as np
from scipy.stats import invgamma, norm
from tqdm import tqdm

from coordination.common.log import BaseLogger
from coordination.common.parallelism import display_inner_progress_bar

from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationBlendingLatentVocalics
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsParticles, \
    BetaCoordinationLatentVocalicsDataSeries, BetaCoordinationLatentVocalicsDataset
from coordination.model.utils.gendered_beta_coordination_blending_latent_vocalics import \
    GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters, GenderedBetaCoordinationLatentVocalicsModelParameters
from coordination.model.utils.coordination_blending_latent_vocalics import clip_coordination
from coordination.model.utils.coordination_blending_latent_vocalics import BaseF, BaseG

# For numerical stability
EPSILON = 1e-6
MIN_COORDINATION = 2 * EPSILON
MAX_COORDINATION = 1 - MIN_COORDINATION


class GenderedBetaCoordinationBlendingLatentVocalics(BetaCoordinationBlendingLatentVocalics):

    def __init__(self,
                 initial_coordination: float,
                 num_vocalic_features: int,
                 num_speakers: int,
                 f: BaseF = BaseF(),
                 g: BaseG = BaseG(),
                 disable_self_dependency: bool = False):
        super().__init__(initial_coordination, num_vocalic_features, num_speakers, f, g, disable_self_dependency)

        # Trainable parameters of the model
        self.parameters = GenderedBetaCoordinationLatentVocalicsModelParameters()

    @classmethod
    def from_pickled_file(cls, filepath: str) -> GenderedBetaCoordinationLatentVocalicsModelParameters:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------
    def _sample_observed_vocalics(self, latent_vocalics: np.array, gender: int) -> np.ndarray:
        # Simply treat even speaker as male and odd as female for synthetic data generation.
        if gender == 0:
            mean = self.parameters.mean_o_male
            var = self.parameters.var_o_male
        else:
            mean = self.parameters.mean_o_female
            var = self.parameters.var_o_female

        return norm(loc=self.g(latent_vocalics * mean), scale=np.sqrt(var)).rvs()

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------
    def _initialize_parameters(self, evidence: BetaCoordinationLatentVocalicsDataset,
                               train_hyper_parameters: GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters,
                               burn_in: int, seed: int):
        super()._initialize_parameters(evidence, train_hyper_parameters, burn_in, seed)

        # Unbounded coordination
        if not self.parameters.mean_var_o_male_frozen:
            self.parameters.set_mean_var_male(train_hyper_parameters.mo0_male, train_hyper_parameters.vo0_male,
                                              freeze=False)

        # Coordination
        if not self.parameters.mean_var_o_female_frozen:
            self.parameters.set_mean_var_female(train_hyper_parameters.mo0_female, train_hyper_parameters.vo0_female,
                                                freeze=False)

    def _sample_latent_vocalics_on_fit(self, evidence: BetaCoordinationLatentVocalicsDataset,
                                       time_steps: np.ndarray, job_num: int, group_order: int) -> np.ndarray:

        # Similar to the implementation in CoordinationBlendingLatentVocalics. Only the final part is different.

        va = self.parameters.var_a
        vaa = self.parameters.var_aa
        mo_male = self.parameters.mean_o_male
        vo_male = self.parameters.var_o_male
        mo_female = self.parameters.mean_o_female
        vo_female = self.parameters.var_o_female
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

            # ---------- Snippet modified to add gender info ---
            G = evidence.genders[:, t]

            mo = np.where(G == 0, mo_male, mo_female)
            vo = np.where(G == 0, vo_male, vo_female)

            m3 = ((m1 / v1) + ((m2a * u2a * M2a) / v2a) + ((m2b * u2b * M2b) / v2b) + (Obs * mo / vo)) * M1
            v_inv = ((1 / v1) + (((u2a ** 2) * M2a) / v2a) + (((u2b ** 2) * M2b) / v2b) + (mo / vo)) * M1

            # ---------- End of gender snippet -----------------

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

    def _compute_joint_loglikelihood_at(self, evidence: BetaCoordinationLatentVocalicsDataset,
                                        train_hyper_parameters: GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters) -> float:
        sa = np.sqrt(self.parameters.var_a)
        saa = np.sqrt(self.parameters.var_aa)
        mo_male = self.parameters.mean_o_male
        so_male = np.sqrt(self.parameters.var_o_male)
        mo_female = self.parameters.mean_o_female
        so_female = np.sqrt(self.parameters.var_o_female)

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
            G = evidence.genders[:, t][:, np.newaxis]
            mo = np.where(G == 0, mo_male, mo_female)
            so = np.where(G == 0, so_male, so_female)

            ll += (norm(loc=V * mo, scale=so).logpdf(evidence.observed_vocalics[:, :, t]) * M).sum()

        return ll

    def _update_parameters(self, evidence: BetaCoordinationLatentVocalicsDataset,
                           train_hyper_parameters: GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters,
                           pool: Pool):

        super()._update_parameters(evidence, train_hyper_parameters, pool)

        # Mean and variance of observed vocalics
        if not self.parameters.mean_var_o_male_frozen:
            V = self.last_latent_vocalics_samples_
            Obs = evidence.observed_vocalics

            # Mask will have 1 when there's a speaker and that speaker is a male
            M = (evidence.vocalics_mask * (1 - evidence.genders))[:, np.newaxis, :]

            n = sum(M.flatten())
            sum_vv = np.sum(V * V * M, axis=(0, 2))
            sum_ov = np.sum(Obs * V * M, axis=(0, 2))
            sum_oo = np.sum(Obs * Obs * M, axis=(0, 2))
            mu = train_hyper_parameters.mu_mo_male
            nu = train_hyper_parameters.nu_mo_male
            a = train_hyper_parameters.a_vo_male
            b = train_hyper_parameters.b_vo_male

            m_star = (nu * mu + sum_ov) / (nu + n)
            a_star = a + n / 2
            b_star = b + (1 / 2) * (
                    (sum_oo - (sum_ov ** 2) / sum_vv) + (nu * sum_vv / (nu + sum_vv)) * (mu - sum_ov / sum_vv) ** 2)

            # In a Gaussian-Inverse-Gamma, the mean of the expected value of the mean is the mean of the posterior
            # and the expected variance is b*/(a*-1).
            self.parameters.set_mean_var_male(m_star, b_star / (a_star - 1), False)

        if not self.parameters.mean_var_o_female_frozen:
            V = self.last_latent_vocalics_samples_
            Obs = evidence.observed_vocalics

            # Mask will have 1 when there's a speaker and that speaker is a female
            M = (evidence.vocalics_mask * evidence.genders)[:, np.newaxis, :]

            n = sum(M.flatten())
            sum_vv = np.sum(V * V * M, axis=(0, 2))
            sum_ov = np.sum(Obs * V * M, axis=(0, 2))
            sum_oo = np.sum(Obs * Obs * M, axis=(0, 2))
            mu = train_hyper_parameters.mu_mo_female
            nu = train_hyper_parameters.nu_mo_female
            a = train_hyper_parameters.a_vo_female
            b = train_hyper_parameters.b_vo_female

            m_star = (nu * mu + sum_ov) / (nu + n)
            a_star = a + n / 2
            b_star = b + (1 / 2) * (
                    (sum_oo - (sum_ov ** 2) / sum_vv) + (nu * sum_vv / (nu + sum_vv)) * (mu - sum_ov / sum_vv) ** 2)

            # In a Gaussian-Inverse-Gamma, the mean of the expected value of the mean is the mean of the posterior
            # and the expected variance is b*/(a*-1).
            self.parameters.set_mean_var_female(m_star, b_star / (a_star - 1), False)

    def _log_parameters(self, gibbs_step: int, logger: BaseLogger):
        super()._log_parameters(gibbs_step, logger)
        logger.add_scalar("train/mean_o_male", self.parameters.mean_o_male, gibbs_step)
        logger.add_scalar("train/var_o_male", self.parameters.var_o_female, gibbs_step)
        logger.add_scalar("train/mean_o_female", self.parameters.mean_o_male, gibbs_step)
        logger.add_scalar("train/var_o_female", self.parameters.var_o_female, gibbs_step)

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
    def _calculate_evidence_log_likelihood_at(self, time_step: int,
                                              states: List[BetaCoordinationLatentVocalicsParticles],
                                              series: BetaCoordinationLatentVocalicsDataSeries):

        if series.coordination is None:
            if series.observed_vocalics.mask[time_step] == 1:
                speaker = series.observed_vocalics.utterances[time_step].subject_id
                Vt = states[time_step].latent_vocalics[speaker]
                Ot = series.observed_vocalics.values[:, time_step]
                gender = series.get_speaker_gender(speaker)

                mo = self.parameters.mean_o_male if gender == 0 else self.parameters.mean_o_female
                vo = self.parameters.var_o_male if gender == 0 else self.parameters.var_o_female

                if np.ndim(Vt) > 1:
                    # particles x features
                    log_likelihoods = norm(loc=self.g(Vt * mo), scale=np.sqrt(vo)).logpdf(Ot).sum(axis=1)
                else:
                    # 1 particle only when ground truth for the latent vocalics is provided
                    log_likelihoods = norm(loc=self.g(Vt * mo), scale=np.sqrt(vo)).logpdf(Ot).sum()
            else:
                log_likelihoods = np.zeros(len(states[time_step].coordination))
        else:
            log_likelihoods = super()._calculate_evidence_log_likelihood_at(time_step, states, series)

        return log_likelihoods
