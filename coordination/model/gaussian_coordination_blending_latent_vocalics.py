from __future__ import annotations

from typing import Callable, List

import numpy as np
from scipy.stats import norm, invgamma
from tqdm import tqdm

from coordination.inference.mcmc import MCMC
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingLatentVocalics, \
    LatentVocalicsDataset, LatentVocalicsParticles
from coordination.model.coordination_blending_latent_vocalics import default_f, default_g


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

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------

    def _generate_coordination_samples(self, num_samples: int, num_time_steps: int) -> np.ndarray:
        scc = np.sqrt(self.var_cc)
        samples = np.zeros((num_samples, num_time_steps))

        for t in tqdm(range(num_time_steps), desc="Coordination", position=0, leave=False):
            if t == 0:
                samples[:, 0] = self.initial_coordination
            else:
                transition_distribution = norm(loc=samples[:, t - 1], scale=scc)
                samples[:, t] = transition_distribution.rvs()

        return samples

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------

    def _compute_coordination_transition_loglikelihood_at(self, gibbs_step: int, evidence: LatentVocalicsDataset):
        scc = np.sqrt(self.vcc_samples_[gibbs_step])

        ll = 0
        coordination = self.coordination_samples_[gibbs_step]
        for t in range(evidence.num_time_steps):
            if t > 0:
                ll += norm(coordination[:, t - 1], scale=scc).logpdf(coordination[:, t]).sum()

        return ll

    def _sample_coordination_on_fit(self, gibbs_step: int, evidence: LatentVocalicsDataset, time_steps: np.ndarray,
                                    job_num: int) -> np.ndarray:
        coordination = self.coordination_samples_[gibbs_step - 1].copy()
        latent_vocalics = self.latent_vocalics_samples_[gibbs_step - 1]

        if evidence.coordination is None:
            scc = np.sqrt(self.vcc_samples_[gibbs_step - 1])
            saa = np.sqrt(self.vaa_samples_[gibbs_step - 1])

            def coordination_proposal(previous_coordination_sample: np.ndarray):
                new_coordination_sample = norm(loc=previous_coordination_sample, scale=0.1).rvs()

                if previous_coordination_sample.shape[0] == 1:
                    # The norm.rvs function does not preserve the dimensions of a unidimensional array.
                    # We need to correct that if we are working with a single trial sample.
                    new_coordination_sample = np.array([new_coordination_sample])

                return new_coordination_sample

            trial_indices = np.arange(evidence.num_trials)
            for t in tqdm(time_steps, desc="Sampling Coordination", position=job_num, leave=False):
                M = evidence.vocalics_mask[:, t]
                Mb = evidence.previous_vocalics_from_other_mask[:, t]

                # If coordination does not affect latent vocalics in a given time step, we can find a closed-form
                # solution for its posterior and avoid using MCMC
                trials_steps_mcmc = trial_indices[M * Mb == 1]
                trials_steps_closed_form = trial_indices[M * Mb == 0]

                def unormalized_log_posterior(sample: np.ndarray):
                    log_posterior = norm(loc=coordination[trials_steps_mcmc, t - 1], scale=scc).logpdf(sample)
                    if t < evidence.num_time_steps - 1:
                        log_posterior += norm(loc=sample, scale=scc).logpdf(coordination[trials_steps_mcmc, t + 1])

                    # Latent vocalics entrainment
                    V = latent_vocalics[trials_steps_mcmc, :, t]
                    A = latent_vocalics[trials_steps_mcmc, :,
                        evidence.previous_vocalics_from_self[trials_steps_mcmc, t]]
                    B = latent_vocalics[trials_steps_mcmc, :,
                        evidence.previous_vocalics_from_other[trials_steps_mcmc, t]]
                    Ma = evidence.previous_vocalics_from_self_mask[trials_steps_mcmc, t][:, np.newaxis]

                    log_posterior += norm(loc=(B - A * Ma) * np.clip(sample[:, np.newaxis], a_min=0, a_max=1) + A * Ma,
                                          scale=saa).logpdf(V).sum()

                    return log_posterior

                def acceptance_criterion(previous_sample: np.ndarray, new_sample: np.ndarray):
                    p1 = unormalized_log_posterior(previous_sample)
                    p2 = unormalized_log_posterior(new_sample)

                    return np.minimum(1, np.exp(p2 - p1))

                if t > 0:
                    # MCMC
                    if len(trials_steps_mcmc) > 0:
                        sampler = MCMC(1, 100, 1, coordination_proposal, acceptance_criterion)
                        coordination[trials_steps_mcmc, t] = \
                            sampler.generate_samples(coordination[trials_steps_mcmc, t])[0]

                    # Closed-form
                    if len(trials_steps_closed_form) > 0:
                        if t < evidence.num_time_steps - 1:
                            mean = (coordination[trials_steps_closed_form, t - 1] + coordination[
                                trials_steps_closed_form, t + 1]) / 2
                            std = scc / np.sqrt(2)
                        else:
                            mean = coordination[trials_steps_closed_form, t - 1]
                            std = scc

                        coordination[trials_steps_closed_form, t] = norm(loc=mean, scale=std).rvs()

        return coordination

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

    def _summarize_particles(self, particles: List[LatentVocalicsParticles]) -> np.ndarray:
        # Mean and variance over time
        summary = np.zeros((2, len(particles)))

        for t, particles_in_time in enumerate(particles):
            summary[0, t] = particles_in_time.coordination.mean()
            summary[1, t] = particles_in_time.coordination.var()

        return summary

    def _sample_coordination_from_transition(self, previous_particles: LatentVocalicsParticles,
                                             new_particles: LatentVocalicsParticles):
        new_particles.coordination = norm(loc=previous_particles.coordination, scale=np.sqrt(self.var_cc)).rvs()

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return LatentVocalicsParticles()


if __name__ == "__main__":
    # mask = np.array([
    #     # [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    #     # [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    #     [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    # ])
    #
    # Ns = np.array([
    #     [-1, 4, 3, 8, 10, -1, -1, -1, -1, -1, 13, 12, 14, 15, 17, 16, -1, -1],
    #     [-1, 4, 3, 8, 10, -1, -1, -1, -1, -1, 13, 12, 14, 15, 17, 16, -1, -1]
    # ])
    #
    # No = np.array([
    #     [-1, 2, 4, 4, 11, -1, -1, -1, -1, -1, 12, 13, 13, 14, 15, 17, 17, -1],
    #     [-1, 2, 4, 4, 11, -1, -1, -1, -1, -1, 12, 13, 13, 14, 15, 17, 17, -1]
    # ])
    #
    # B = np.array_split(np.arange(18), 3)
    # M = np.array_split(mask, 3, axis=-1)
    #
    # time_steps = Ns.shape[1]
    # num_trials = Ns.shape[0]
    # index_mask = np.arange(time_steps)[np.newaxis, :].repeat(num_trials, axis=0)
    #
    # # For indexes in which the next speaker does not exist, we replace with the current index. That is, there's
    # # no dependency in the future
    # Ns = np.where(Ns == -1, index_mask, Ns)
    # No = np.where(No == -1, index_mask, No)
    #
    # parallel_time_steps = []
    # independent_time_steps = []
    # j = 0
    # while j < len(B) - 1:
    #     block_size = len(B[j])
    #
    #     if block_size > 0:
    #         # Last indexes where M[j] = 1 per column
    #         last_indices_with_speaker = block_size - np.argmax(np.flip(M[j], axis=1), axis=1) - 1
    #         last_times_with_speaker = B[j][last_indices_with_speaker]
    #         next_block_time_step_self = np.take_along_axis(Ns, last_times_with_speaker[:, np.newaxis], axis=-1)
    #         next_block_time_step_other = np.take_along_axis(No, last_times_with_speaker[:, np.newaxis], axis=-1)
    #         last_time_step_independent_block = np.maximum(np.max(next_block_time_step_self), np.max(next_block_time_step_other))
    #
    #         if last_time_step_independent_block > B[j][-1]:
    #             # There is a dependency with the next block
    #             independent_range = np.arange(B[j][-1] + 1, last_time_step_independent_block + 1)
    #             independent_time_steps.extend(independent_range)
    #
    #         parallel_time_steps.append(B[j])
    #
    #         while last_time_step_independent_block > B[j + 1][-1]:
    #             j += 1
    #         next_parallel_range = np.arange(last_time_step_independent_block + 1, B[j + 1][-1] + 1)
    #
    #         B[j + 1] = next_parallel_range
    #
    #     j += 1

    # if len(B[-1]) > 0:
    #     parallel_time_steps.append(B[-1])
    #
    # print(independent_time_steps)
    # print(parallel_time_steps)

    TIME_STEPS = 20
    NUM_SAMPLES = 100
    NUM_FEATURES = 2
    model = GaussianCoordinationBlendingLatentVocalics(
        initial_coordination=0,
        num_vocalic_features=NUM_FEATURES,
        num_speakers=2,
        a_vcc=1,
        b_vcc=1,
        a_va=1,
        b_va=1,
        a_vaa=1,
        b_vaa=1,
        a_vo=1,
        b_vo=1
    )

    VAR_CC = 0.1
    VAR_A = 1
    VAR_AA = 0.5
    VAR_O = 1

    model.var_cc = VAR_CC
    model.var_a = VAR_A
    model.var_aa = VAR_AA
    model.var_o = VAR_O

    samples = model.sample(NUM_SAMPLES, TIME_STEPS, seed=0, time_scale_density=0.5)

    import matplotlib.pyplot as plt

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

    from coordination.common.log import TensorBoardLogger

    # model.var_cc = None
    # model.var_a = None
    # model.var_aa = None
    # model.var_o = None
    # model.fit(full_evidence, burn_in=1, seed=0, num_jobs=1)
    #
    # print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    # print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    # print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    # print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")
    #

    # model.var_cc = None
    # model.var_a = None
    # model.var_aa = None
    # model.var_o = None
    # tb_logger = TensorBoardLogger("/Users/paulosoares/code/tomcat-coordination/boards/evidence_with_latent_vocalics")
    # model.fit(evidence_with_latent_vocalics, burn_in=20, seed=0, num_jobs=1, logger=tb_logger)
    # print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    # print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    # print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    # print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")
    #
    # cs = model.coordination_samples_[-1].mean(axis=0)
    # std = model.coordination_samples_[-1].std(axis=0)
    #
    # plt.figure()
    # plt.plot(ts, cs, color="tab:orange", marker="o")
    # plt.fill_between(ts, cs - std, cs + std, color="tab:orange",
    #                  alpha=0.5)
    # plt.plot(ts, samples.coordination.mean(axis=0), color="tab:blue", marker="o")
    # plt.show()

    # model.var_cc = None
    # model.var_a = None
    # model.var_aa = None
    # model.var_o = None
    # tb_logger = TensorBoardLogger("/Users/paulosoares/code/tomcat-coordination/boards/evidence_with_coordination")
    # model.fit(evidence_with_coordination, burn_in=100, seed=0, num_jobs=1, logger=tb_logger)
    # print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    # print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    # print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    # print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")
    #
    # cs = model.latent_vocalics_samples_[-1, :, 0].mean(axis=0)
    # std = model.latent_vocalics_samples_[-1, :, 0].std(axis=0)
    #
    # plt.figure()
    # plt.plot(ts, cs, color="tab:orange", marker="o")
    # plt.fill_between(ts, cs - std, cs + std, color="tab:orange",
    #                  alpha=0.5)
    # plt.plot(ts, evidence_with_latent_vocalics.latent_vocalics[:, 0].mean(axis=0), color="tab:blue", marker="o")
    # plt.show()

    # ---------------------

    model.var_cc = None
    model.var_a = None
    model.var_aa = None
    model.var_o = None
    tb_logger = TensorBoardLogger("/Users/paulosoares/code/tomcat-coordination/boards/partial_evidence_small_density")
    model.fit(partial_evidence, burn_in=100, seed=0, num_jobs=2, logger=tb_logger)
    print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")
