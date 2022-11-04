from __future__ import annotations

import random
from typing import Any, Callable, List, Tuple, Union, Optional

import numpy as np
from scipy.stats import norm, invgamma
from tqdm import tqdm

from coordination.common.distribution import truncnorm
from coordination.inference.mcmc import MCMC
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingLatentVocalics, \
    LatentVocalicsDataset, LatentVocalicsDataSeries, LatentVocalicsParticles
from coordination.model.coordination_blending_latent_vocalics import clip_coordination, default_f, default_g


# def coordination_proposal(previous_coordination_sample: np.ndarray):
#     # Since coordination is constrained to 0 and 1, we don't expect a high variance.
#     # We set variance to be smaller than 0.01 such that MCMC don't do big jumps and ends up
#     # overestimating coordination.
#     std = 0.005
#     new_coordination_sample = truncnorm(previous_coordination_sample, std).rvs()
#
#     if previous_coordination_sample.shape[0] == 1:
#         # The norm.rvs function does not preserve the dimensions of a unidimensional array.
#         # We need to correct that if we are working with a single trial sample.
#         new_coordination_sample = np.array([[new_coordination_sample]])
#
#     # Hastings factor
#     nominator = truncnorm(new_coordination_sample, std).logpdf(previous_coordination_sample)
#     denominator = truncnorm(previous_coordination_sample, std).logpdf(new_coordination_sample)
#     factor = np.exp(nominator - denominator).sum(axis=1)
#
#     return new_coordination_sample, factor
#
#
# def coordination_posterior_unormalized_logprob(proposed_coordination_sample: np.ndarray,
#                                                previous_coordination_sample: np.ndarray,
#                                                next_coordination_sample: Optional[np.ndarray],
#                                                scc: float,
#                                                saa: float,
#                                                evidence: LatentVocalicsDataset,
#                                                latent_vocalics: np.ndarray,
#                                                time_step: int):
#     log_posterior = truncnorm(previous_coordination_sample, scc).logpdf(proposed_coordination_sample)
#     if next_coordination_sample is not None:
#         log_posterior += truncnorm(proposed_coordination_sample, scc).logpdf(next_coordination_sample)
#
#     V = latent_vocalics[..., time_step]
#
#     previous_self_time_steps = evidence.previous_vocalics_from_self[:, time_step]
#     previous_other_time_steps = evidence.previous_vocalics_from_other[:, time_step]
#     A = np.take_along_axis(latent_vocalics, previous_self_time_steps[:, np.newaxis, np.newaxis], axis=-1)[..., 0]
#     B = np.take_along_axis(latent_vocalics, previous_other_time_steps[:, np.newaxis, np.newaxis], axis=-1)[..., 0]
#
#     M = evidence.vocalics_mask[:, time_step][:, np.newaxis]
#     Ma = evidence.previous_vocalics_from_self_mask[:, time_step][:, np.newaxis]
#     Mb = evidence.previous_vocalics_from_other_mask[:, time_step][:, np.newaxis]
#
#     mean = ((B - A * Ma) * clip_coordination(proposed_coordination_sample) * Mb + A * Ma) * M
#
#     log_posterior = log_posterior.flatten()
#     log_posterior += (norm(loc=mean, scale=saa).logpdf(V) * M).sum(axis=1)
#
#     return log_posterior


class GaussianCoordinationBlendingLatentVocalics(CoordinationBlendingLatentVocalics):

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
        super().__init__(initial_coordination, num_vocalic_features, num_speakers, a_vcc, b_vcc, a_va, b_va, a_vaa,
                         b_vaa, a_vo, b_vo, f, g)

    def _get_coordination_distribution(self, mean: np.ndarray, std: np.ndarray) -> Any:
        return truncnorm(mean, std)

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------

    # def _generate_coordination_samples(self, num_samples: int, num_time_steps: int) -> np.ndarray:
    #     # scc = np.sqrt(self.var_cc)
    #     samples = np.zeros((num_samples, num_time_steps))
    #
    #     for t in tqdm(range(num_time_steps), desc="Coordination", position=0, leave=False):
    #         if t == 0:
    #             samples[:, 0] = self.initial_coordination
    #         else:
    #             # The mean of a truncated Gaussian distribution is given by mu + an offset. We remove the offset here,
    #             # such that the previous sample is indeed the mean of the truncated Gaussian.
    #             mean = samples[:, t - 1]
    #             # samples[:, t] = truncnorm(mean, scc).rvs()
    #             samples[:, t] = self._get_coordination_distribution(mean).rvs()
    #
    #     return samples

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------
    def _get_initial_coordination_for_gibbs(self, evidence: LatentVocalicsDataset) -> np.ndarray:
        # samples = truncnorm(np.zeros((evidence.num_trials, evidence.num_time_steps)), 0.1).rvs()
        means = np.zeros((evidence.num_trials, evidence.num_time_steps))
        samples = truncnorm(means, 0.1).rvs()
        samples[0] = self.initial_coordination

        return samples

    # def _compute_coordination_transition_loglikelihood_at(self, gibbs_step: int, evidence: LatentVocalicsDataset):
    #     # scc = np.sqrt(self.vcc_samples_[gibbs_step])
    #
    #     ll = 0
    #     coordination = self.coordination_samples_[gibbs_step]
    #     for t in range(evidence.num_time_steps):
    #         if t > 0:
    #             mean = coordination[:, t - 1]
    #             # ll += truncnorm(mean, scc).logpdf(coordination[:, t]).sum()
    #             ll += self._get_coordination_distribution(mean, gibbs_step).logpdf(coordination[:, t]).sum()
    #
    #     return ll

    # def _sample_coordination_on_fit(self, gibbs_step: int, evidence: LatentVocalicsDataset, time_steps: np.ndarray,
    #                                 job_num: int) -> np.ndarray:
    #     coordination = self.coordination_samples_[gibbs_step - 1].copy()
    #     latent_vocalics = self.latent_vocalics_samples_[gibbs_step - 1]
    #
    #     if evidence.coordination is None:
    #         scc = np.sqrt(self.vcc_samples_[gibbs_step - 1])
    #         saa = np.sqrt(self.vaa_samples_[gibbs_step - 1])
    #
    #         for t in tqdm(time_steps, desc="Sampling Coordination", position=job_num, leave=False):
    #             if t > 0:
    #                 next_coordination = None if t == len(time_steps) - 1 else coordination[:, t + 1][:, np.newaxis]
    #                 log_prob_fn_params = {
    #                     "previous_coordination_sample": coordination[:, t - 1][:, np.newaxis],
    #                     "next_coordination_sample": next_coordination,
    #                     "scc": scc,
    #                     "saa": saa,
    #                     "evidence": evidence,
    #                     "latent_vocalics": latent_vocalics,
    #                     "time_step": t
    #                 }
    #
    #                 sampler = MCMC(proposal_fn=coordination_proposal,
    #                                proposal_fn_kwargs={},
    #                                log_prob_fn=coordination_posterior_unormalized_logprob,
    #                                log_prob_fn_kwargs=log_prob_fn_params)
    #                 initial_sample = coordination[:, t][:, np.newaxis]
    #                 inferred_coordination = sampler.generate_samples(initial_sample=initial_sample,
    #                                                                  num_samples=1,
    #                                                                  burn_in=50,
    #                                                                  retain_every=1)[0, :, 0]
    #                 coordination[:, t] = inferred_coordination
    #
    #     return coordination

    def _get_coordination_proposal(self, previous_coordination_sample: np.ndarray):

        # Since coordination is constrained to 0 and 1, we don't expect a high variance.
        # We set variance to be smaller than 0.01 such that MCMC don't do big jumps and ends up
        # overestimating coordination.
        std = 0.005
        new_coordination_sample = truncnorm(previous_coordination_sample, std).rvs()

        if previous_coordination_sample.shape[0] == 1:
            # The norm.rvs function does not preserve the dimensions of a unidimensional array.
            # We need to correct that if we are working with a single trial sample.
            new_coordination_sample = np.array([[new_coordination_sample]])

        # Hastings factor
        # nominator = truncnorm(new_coordination_sample, std).logpdf(previous_coordination_sample)
        # denominator = truncnorm(previous_coordination_sample, std).logpdf(new_coordination_sample)
        # factor = np.exp(nominator - denominator).sum(axis=1)
        factor = 1

        return new_coordination_sample, factor

    def _update_latent_parameters_coordination(self, gibbs_step: int, evidence: LatentVocalicsDataset):
        # Variance of the State Transition
        if self.var_cc is None:
            a = self.a_vcc + evidence.num_trials * (evidence.num_time_steps - 1) / 2
            x = self.coordination_samples_[gibbs_step, :, 1:]
            y = self.coordination_samples_[gibbs_step, :, :evidence.num_time_steps - 1]
            b = self.b_vcc + np.square(x - y).sum() / 2
            self.vcc_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()
            if self.vcc_samples_[gibbs_step] == np.nan:
                self.vcc_samples_[gibbs_step] = np.inf
        else:
            # Given
            self.vcc_samples_[gibbs_step] = self.var_cc

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------

    def _summarize_particles(self, series: LatentVocalicsDataSeries,
                             particles: List[LatentVocalicsParticles]) -> np.ndarray:
        # Mean and variance over time
        summary = np.zeros((4, len(particles)))

        for t, particles_in_time in enumerate(particles):
            if t == 0:
                summary[0, t] = self.initial_coordination
                summary[1, t] = 0
            else:
                summary[0, t] = particles_in_time.coordination.mean()
                summary[1, t] = particles_in_time.coordination.var()

            if series.observed_vocalics.mask[t] == 1:
                speaker = series.observed_vocalics.utterances[t].subject_id
                summary[2, t] = particles_in_time.latent_vocalics[speaker][:, 0].mean()
                summary[3, t] = particles_in_time.latent_vocalics[speaker][:, 0].var()
            else:
                summary[2, t] = -1
                summary[3, t] = 0

        return summary

    # def _sample_coordination_from_transition(self, previous_particles: LatentVocalicsParticles,
    #                                          new_particles: LatentVocalicsParticles):
    #     mean = previous_particles.coordination
    #     scc = np.sqrt(self.var_cc)
    #     new_particles.coordination = truncnorm(mean, scc).rvs()

    # def _create_new_particles(self) -> LatentVocalicsParticles:
    #     return LatentVocalicsParticles()


if __name__ == "__main__":
    from coordination.common.utils import set_seed
    import matplotlib.pyplot as plt
    from coordination.common.log import TensorBoardLogger

    TIME_STEPS = 50
    NUM_SAMPLES = 100
    NUM_FEATURES = 2
    model = GaussianCoordinationBlendingLatentVocalics(
        initial_coordination=0,
        num_vocalic_features=NUM_FEATURES,
        num_speakers=3,
        a_vcc=1,
        b_vcc=1,
        a_va=1,
        b_va=1,
        a_vaa=1,
        b_vaa=1,
        a_vo=1,
        b_vo=1
    )

    VAR_CC = 0.01
    VAR_A = 1
    VAR_AA = 0.5
    VAR_O = 1

    model.var_cc = VAR_CC
    model.var_a = VAR_A
    model.var_aa = VAR_AA
    model.var_o = VAR_O

    samples = model.sample(NUM_SAMPLES, TIME_STEPS, seed=0, time_scale_density=0.5)

    ts = np.arange(TIME_STEPS)

    # plt.figure()
    # plt.title("Coordination")
    # plt.plot(ts, samples.coordination[0], color="tab:red", marker="o")
    # plt.plot(ts, samples.coordination[1], color="tab:green", marker="o")
    # plt.plot(ts, samples.coordination[2], color="tab:blue", marker="o")
    # plt.show()
    #
    # plt.figure()
    # plt.title("Latent Vocalics")
    # plt.plot(ts, samples.latent_vocalics[0].values[0], color="tab:red", marker="o")
    # plt.plot(ts, samples.latent_vocalics[0].values[1], color="tab:red", marker="+")
    # plt.plot(ts, samples.latent_vocalics[1].values[0], color="tab:green", marker="o")
    # plt.plot(ts, samples.latent_vocalics[1].values[1], color="tab:green", marker="+")
    # plt.plot(ts, samples.latent_vocalics[2].values[0], color="tab:blue", marker="o")
    # plt.plot(ts, samples.latent_vocalics[2].values[1], color="tab:blue", marker="+")
    # plt.show()
    #
    # plt.figure()
    # plt.title("Observed Vocalics")
    # plt.plot(ts, samples.observed_vocalics[0].values[0], color="tab:red", marker="o")
    # plt.plot(ts, samples.observed_vocalics[0].values[1], color="tab:red", marker="+")
    # plt.plot(ts, samples.observed_vocalics[1].values[0], color="tab:green", marker="o")
    # plt.plot(ts, samples.observed_vocalics[1].values[1], color="tab:green", marker="+")
    # plt.plot(ts, samples.observed_vocalics[2].values[0], color="tab:blue", marker="o")
    # plt.plot(ts, samples.observed_vocalics[2].values[1], color="tab:blue", marker="+")
    # plt.show()

    from coordination.model.coordination_blending_latent_vocalics import LatentVocalicsDataSeries

    full_evidence = LatentVocalicsDataset(
        [LatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i], samples.coordination[i],
                                  samples.latent_vocalics[i]) for i in
         range(samples.size)])

    evidence_with_coordination = LatentVocalicsDataset(
        [LatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i], samples.coordination[i]) for i in
         range(samples.size)])

    evidence_with_latent_vocalics = LatentVocalicsDataset(
        [LatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i], latent_vocalics=samples.latent_vocalics[i]) for
         i in range(samples.size)])

    partial_evidence = LatentVocalicsDataset(
        [LatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i]) for i in range(samples.size)])

    model.fit(full_evidence, burn_in=1, seed=0, num_jobs=1)
    true_nll = model.nll_[-1]

    print(f"True NLL = {true_nll}")

    # model.var_cc = None
    # model.var_a = None
    # model.var_aa = None
    # model.var_o = None
    # model.fit(full_evidence, burn_in=1, seed=0, num_jobs=1)
    #
    # print()
    # print("Parameter estimation with full evidence")
    # print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    # print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    # print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    # print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")
    #
    model.var_cc = None
    model.var_a = None
    model.var_aa = None
    model.var_o = None
    tb_logger = TensorBoardLogger("/Users/paulosoares/code/tomcat-coordination/boards/evidence_with_coordination")
    model.fit(evidence_with_coordination, burn_in=100, seed=0, num_jobs=4, logger=tb_logger)

    print()
    print("Parameter estimation with coordination")
    print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")

    model.var_cc = None
    model.var_a = None
    model.var_aa = None
    model.var_o = None
    tb_logger = TensorBoardLogger("/Users/paulosoares/code/tomcat-coordination/boards/evidence_with_latent_vocalics")
    model.fit(evidence_with_latent_vocalics, burn_in=100, seed=0, num_jobs=4, logger=tb_logger)

    print()
    print("Parameter estimation with latent vocalics")
    print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")

    # model.var_cc = None
    # model.var_a = None
    # model.var_aa = None
    # model.var_o = None
    # tb_logger = TensorBoardLogger("/Users/paulosoares/code/tomcat-coordination/boards/partial_evidence")
    # model.fit(partial_evidence, burn_in=100, seed=0, num_jobs=4, logger=tb_logger)
    #
    # print()
    # print("Parameter estimation with partial evidence")
    # print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    # print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    # print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    # print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")

    # estimates = model.predict(evidence=evidence_with_latent_vocalics, num_particles=10000, seed=0, num_jobs=1)
    #
    # plt.figure(figsize=(15, 8))
    # means = estimates[0][0]
    # stds = np.sqrt(estimates[0][1])
    # plt.plot(ts, means, color="tab:orange", marker="o")
    # plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    # plt.plot(ts, samples.coordination[0], color="tab:blue", marker="o", alpha=0.5)
    # plt.show()
    #
    # plt.figure()
    # means = estimates[0][2]
    # stds = np.sqrt(estimates[0][3])
    # plt.plot(ts, means, color="tab:orange", marker="o")
    # plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    # plt.plot(ts, samples.latent_vocalics[0].values[0], color="tab:blue", marker="o", alpha=0.5)
    # plt.show()
    #
    # means = []
    # means_offset = []
    #
    # ini = 1
    # start = np.ones(50000) * ini
    # means.append(start.mean())
    # # means_offset.append(np.clip((start - offset).mean(), a_min=0, a_max=1))
    # # start -= offset
    # for t in range(1, 100):
    #     offset, a, b = truncate_norm_mean_offset(start, 0.1)
    #     # offset = (norm().pdf(a) - norm().pdf(b)) * 0.1 / (norm().cdf(b) - norm().cdf(a))
    #     start = truncnorm(loc=start - offset, scale=0.1, a=a, b=b).rvs()
    #     # start = norm(loc=start, scale=0.1).rvs()
    #     means.append(start.mean())
    #     # means_offset.append(np.clip((start - offset).mean(), a_min=0, a_max=1))
    #     # start -= offset
    #
    # plt.figure()
    # plt.plot(range(100), means, marker="o", label="raw")
    # # plt.plot(range(100), means_offset, marker="o", label="offset")
    # plt.legend()
    # plt.ylim([0, 1])
    # plt.show()
