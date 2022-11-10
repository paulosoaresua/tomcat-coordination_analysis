from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy.stats import invgamma, norm
from tqdm import tqdm

from coordination.common.log import BaseLogger
from coordination.common.distribution import beta
from coordination.common.utils import logit, sigmoid
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingLatentVocalics, \
    LatentVocalicsDataset, LatentVocalicsDataSeries, LatentVocalicsParticles, LatentVocalicsSamples, \
    LatentVocalicsParticlesSummary
from coordination.model.coordination_blending_latent_vocalics import default_f, default_g
from coordination.inference.mcmc import MCMC
from coordination.component.speech.common import VocalicsSparseSeries

import matplotlib.pyplot as plt

# For numerical stability
EPSILON = 1e-6
MIN_COORDINATION = 2 * EPSILON
MAX_COORDINATION = 1 - MIN_COORDINATION


class BetaCoordinationLatentVocalicsParticles(LatentVocalicsParticles):
    unbounded_coordination: np.ndarray

    def _keep_particles_at(self, indices: np.ndarray):
        super()._keep_particles_at(indices)

        if isinstance(self.unbounded_coordination, np.ndarray):
            self.unbounded_coordination = self.unbounded_coordination[indices]


class BetaCoordinationLatentVocalicsParticlesSummary(LatentVocalicsParticlesSummary):
    unbounded_coordination_mean: np.ndarray
    unbounded_coordination_var: np.ndarray

    @classmethod
    def from_latent_vocalics_particles_summary(cls, summary: LatentVocalicsParticlesSummary):
        new_summary = cls()
        new_summary.coordination_mean = summary.coordination_mean
        new_summary.coordination_var = summary.coordination_var
        new_summary.latent_vocalics_mean = summary.latent_vocalics_mean
        new_summary.latent_vocalics_var = summary.latent_vocalics_var

        return new_summary


class BetaCoordinationLatentVocalicsSamples(LatentVocalicsSamples):
    unbounded_coordination: np.ndarray


class BetaCoordinationLatentVocalicsDataSeries(LatentVocalicsDataSeries):

    def __init__(self, uuid: str, observed_vocalics: VocalicsSparseSeries,
                 unbounded_coordination: Optional[np.ndarray] = None,
                 coordination: Optional[np.ndarray] = None, latent_vocalics: VocalicsSparseSeries = None):
        super().__init__(uuid, observed_vocalics, coordination, latent_vocalics)
        self.unbounded_coordination = unbounded_coordination

    @property
    def is_complete(self) -> bool:
        return super().is_complete and self.unbounded_coordination is not None


class BetaCoordinationLatentVocalicsDataset(LatentVocalicsDataset):

    def __init__(self, series: List[BetaCoordinationLatentVocalicsDataSeries]):
        super().__init__(series)

        self.unbounded_coordination = None if series[0].unbounded_coordination is None else np.array(
            [s.unbounded_coordination for s in series])


class BetaCoordinationBlendingLatentVocalics(
    CoordinationBlendingLatentVocalics[
        BetaCoordinationLatentVocalicsSamples, BetaCoordinationLatentVocalicsParticlesSummary]):

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
                 a_vuc: float,
                 b_vuc: float,
                 initial_var_unbounded_coordination: float,
                 initial_var_coordination: float,
                 var_unbounded_coordination_proposal: float,
                 var_coordination_proposal: float,
                 unbounded_coordination_num_mcmc_iterations: int = 50,
                 coordination_num_mcmc_iterations: int = 50,
                 f: Callable = default_f,
                 g: Callable = default_g):
        super().__init__(initial_coordination, num_vocalic_features, num_speakers, a_vcc, b_vcc, a_va, b_va, a_vaa,
                         b_vaa, a_vo, b_vo, f, g)

        self.var_uc: Optional[float] = None

        self.a_vuc = a_vuc
        self.b_vuc = b_vuc
        self.initial_var_unbounded_coordination = initial_var_unbounded_coordination
        self.initial_var_coordination = initial_var_coordination
        self.var_unbounded_coordination_proposal = var_unbounded_coordination_proposal
        self.var_coordination_proposal = var_coordination_proposal
        self.unbounded_coordination_num_mcmc_iterations = unbounded_coordination_num_mcmc_iterations
        self.coordination_num_mcmc_iterations = coordination_num_mcmc_iterations

        self._hyper_params["a_vuc"] = a_vuc
        self._hyper_params["b_vuc"] = b_vuc
        self._hyper_params["vuc_0"] = initial_var_unbounded_coordination
        self._hyper_params["vcc_0"] = initial_var_coordination
        self._hyper_params["vuc_prop"] = var_unbounded_coordination_proposal
        self._hyper_params["vcc_prop"] = var_coordination_proposal
        self._hyper_params["uc_mcmc_iter"] = unbounded_coordination_num_mcmc_iterations
        self._hyper_params["c_mcmc_iter"] = coordination_num_mcmc_iterations

        self.vuc_samples_ = np.array([])

        # Acceptance rate in the samples of coordination over time in the last Gibbs step.
        # We log this to keep track of the MCMC sampler's health.
        self.unbounded_coordination_acceptance_rates_ = np.array([])
        self.coordination_acceptance_rates_ = np.array([])

    def reset_parameters(self):
        super().reset_parameters()
        self.var_uc = None

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------
    def _generate_coordination_samples(self, num_samples: int, num_time_steps: int,
                                       samples: BetaCoordinationLatentVocalicsSamples):
        samples.unbounded_coordination = np.zeros((num_samples, num_time_steps))
        samples.coordination = np.zeros((num_samples, num_time_steps))

        suc = np.sqrt(self.var_uc)

        for t in tqdm(range(num_time_steps), desc="Coordination", position=0, leave=False):
            if t == 0:
                samples.unbounded_coordination[:, 0] = logit(self.initial_coordination)
                samples.coordination[:, 0] = self.initial_coordination
            else:
                # The variance of a beta distribution, cannot be bigger than m * (1 - m). Therefore, we
                # constrain the sampled from the unbounded distribution such that we cannot generate
                # beta distributions with impossible means when we compute coordination.
                min_value = logit((1 - np.sqrt(1 - 4 * (self.var_cc + EPSILON))) / 2)
                max_value = logit((1 + np.sqrt(1 - 4 * (self.var_cc + EPSILON))) / 2)
                mean = samples.unbounded_coordination[:, t - 1]
                samples.unbounded_coordination[:, t] = np.clip(norm(mean, suc).rvs(), a_min=min_value, a_max=max_value)

                m = sigmoid(samples.unbounded_coordination[:, t])
                samples.coordination[:, t] = np.clip(beta(m, self.var_cc).rvs(), a_min=MIN_COORDINATION,
                                                     a_max=MAX_COORDINATION)

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------
    def _initialize_coordination_for_gibbs(self, evidence: BetaCoordinationLatentVocalicsDataset):
        burn_in = self.coordination_samples_.shape[0] - 1

        self.vuc_samples_ = np.zeros(burn_in + 1)
        self.unbounded_coordination_samples_ = np.zeros((burn_in + 1, evidence.num_trials, evidence.num_time_steps))
        self.unbounded_coordination_acceptance_rates_ = np.ones((evidence.num_trials, evidence.num_time_steps))
        self.coordination_acceptance_rates_ = np.ones((evidence.num_trials, evidence.num_time_steps))

        # Initialize variables
        if self.var_uc is None:
            self.vuc_samples_[0] = self.initial_var_unbounded_coordination
            if burn_in > 0:
                self.vuc_samples_[1] = self.vuc_samples_[0]
        else:
            self.vuc_samples_[:] = self.var_uc

        if self.var_cc is None:
            self.vcc_samples_[0] = self.initial_var_coordination

            if self.vcc_samples_.shape[0] > 0:
                self.vcc_samples_[1] = self.vcc_samples_[0]
        else:
            self.vcc_samples_[:] = self.var_cc

        if evidence.unbounded_coordination is None:
            suc = np.sqrt(self.vuc_samples_[0])
            vcc = self.vcc_samples_[0]
            min_value = logit((1 - np.sqrt(1 - 4 * (vcc + EPSILON))) / 2)
            max_value = logit((1 + np.sqrt(1 - 4 * (vcc + EPSILON))) / 2)
            self.unbounded_coordination_samples_[0] = logit(self.initial_coordination)
            # self.unbounded_coordination_samples_[0, :, 0] = logit(self.initial_coordination)
            # for t in range(1, evidence.num_time_steps):
            #     self.unbounded_coordination_samples_[0, :, t] = np.clip(
            #         norm(self.unbounded_coordination_samples_[0, :, t - 1],
            #              suc).rvs(), a_min=min_value, a_max=max_value)
            #
            # if self.unbounded_coordination_samples_.shape[0] > 0:
            #     self.unbounded_coordination_samples_[1] = self.unbounded_coordination_samples_[0]
        else:
            self.unbounded_coordination_samples_[:] = evidence.unbounded_coordination[np.newaxis, :]

        if evidence.coordination is None:
            m = sigmoid(self.unbounded_coordination_samples_[0])
            vcc = self.vcc_samples_[0]

            # The variance of a beta distribution cannot be bigger than mean * (1 - mean)
            # TODO - vcc = np.where(m * (1 - m) < self.vcc_samples_[0], m * (1 - m) - EPSILON, self.vcc_samples_[0])

            # Don't let coordination samples be 0 or 1 for numerical stability.
            self.coordination_samples_[0] = np.clip(beta(m, vcc).rvs(), a_min=MIN_COORDINATION, a_max=MAX_COORDINATION)
            self.coordination_samples_[0, :, 0] = self.initial_coordination

            if self.coordination_samples_.shape[0] > 0:
                self.coordination_samples_[1] = self.coordination_samples_[0]
        else:
            self.coordination_samples_[:] = evidence.coordination[np.newaxis, :]

    def _compute_coordination_loglikelihood(self, gibbs_step: int,
                                            evidence: BetaCoordinationLatentVocalicsDataset) -> float:
        unbounded_coordination = self.unbounded_coordination_samples_[gibbs_step]
        coordination = self.coordination_samples_[gibbs_step]

        suc = np.sqrt(self.vuc_samples_[gibbs_step])
        vcc = self.vcc_samples_[gibbs_step]

        ll = 0
        for t in range(1, evidence.num_time_steps):
            # Initial coordination is given
            ll += norm(unbounded_coordination[:, t - 1], suc).logpdf(unbounded_coordination[:, t]).sum()

            m = sigmoid(unbounded_coordination[:, t])
            # TODO - var = np.minimum(m * (1 - m) - EPSILON, vcc)
            # ll += beta(m, var).logpdf(coordination[:, t]).sum()
            ll += beta(m, vcc).logpdf(coordination[:, t]).sum()

        return ll

    def _sample_coordination_on_fit(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                                    time_steps: np.ndarray,
                                    job_num: int) -> Tuple[np.ndarray, ...]:

        unbounded_coordination, uc_acceptance_rates = self._sample_unbounded_coordination_on_fit(
            gibbs_step, evidence, time_steps, job_num)

        coordination = self.coordination_samples_[gibbs_step].copy()
        coordination_acceptance_rates = self.coordination_acceptance_rates_.copy()

        if evidence.coordination is not None:
            return coordination, unbounded_coordination, coordination_acceptance_rates, uc_acceptance_rates

        # The retain method copies the estimate in one gibbs step to the next one. Therefore, accessing the values in
        # the current gibbs step will give us the latest values of the estimates.
        latent_vocalics = self.latent_vocalics_samples_[gibbs_step]

        vcc = self.vcc_samples_[gibbs_step]
        saa = np.sqrt(self.vaa_samples_[gibbs_step])

        for t in tqdm(time_steps, desc="Sampling Coordination", position=job_num, leave=False):
            if t > 0:
                # Initial coordination is given

                proposal_fn_params = {
                    "vcc": vcc
                }

                log_prob_fn_params = {
                    "current_unbounded_coordination_sample": unbounded_coordination[:, t][:, np.newaxis],
                    "vcc": vcc,
                    "saa": saa,
                    "evidence": evidence,
                    "latent_vocalics": latent_vocalics,
                    "time_step": t
                }

                sampler = MCMC(proposal_fn=self._get_coordination_proposal,
                               proposal_fn_kwargs=proposal_fn_params,
                               log_prob_fn=self._get_coordination_posterior_unormalized_logprob,
                               log_prob_fn_kwargs=log_prob_fn_params)
                initial_sample = coordination[:, t][:, np.newaxis]
                inferred_coordination = sampler.generate_samples(initial_sample=initial_sample,
                                                                 num_samples=1,
                                                                 burn_in=self.coordination_num_mcmc_iterations,
                                                                 retain_every=1)[0, :, 0]
                coordination[:, t] = inferred_coordination
                coordination_acceptance_rates[:, t] = sampler.acceptance_rates_[-1]

        return coordination, unbounded_coordination, coordination_acceptance_rates, uc_acceptance_rates

    def _sample_unbounded_coordination_on_fit(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                                              time_steps: np.ndarray, job_num: int) -> Tuple[np.ndarray, np.ndarray]:

        unbounded_coordination = self.unbounded_coordination_samples_[gibbs_step].copy()
        acceptance_rates = self.unbounded_coordination_acceptance_rates_.copy()

        if evidence.unbounded_coordination is not None:
            return unbounded_coordination, acceptance_rates

        # The retain method copies the estimate in one gibbs step to the next one. Therefore, accessing the values in
        # the current gibbs step will give us the latest values of the estimates.
        coordination = self.coordination_samples_[gibbs_step]

        suc = np.sqrt(self.vuc_samples_[gibbs_step])
        vcc = self.vcc_samples_[gibbs_step]

        for t in tqdm(time_steps, desc="Sampling Unbounded Coordination", position=job_num, leave=False):
            if t > 0:
                next_unbounded_coordination = None if t == unbounded_coordination.shape[
                    1] - 1 else unbounded_coordination[:, t + 1][:, np.newaxis]

                proposal_fn_params = {
                    "vcc": vcc
                }

                log_prob_fn_params = {
                    "previous_unbounded_coordination_sample": unbounded_coordination[:, t - 1][:, np.newaxis],
                    "next_unbounded_coordination_sample": next_unbounded_coordination,
                    "suc": suc,
                    "vcc": vcc,
                    "coordination": coordination[:, t][:, np.newaxis]
                }

                sampler = MCMC(proposal_fn=self._get_unbounded_coordination_proposal,
                               proposal_fn_kwargs=proposal_fn_params,
                               log_prob_fn=self._get_unbounded_coordination_posterior_unormalized_logprob,
                               log_prob_fn_kwargs=log_prob_fn_params)
                initial_sample = unbounded_coordination[:, t][:, np.newaxis]
                inferred_unbounded_coordination = sampler.generate_samples(initial_sample=initial_sample,
                                                                           num_samples=1,
                                                                           burn_in=self.unbounded_coordination_num_mcmc_iterations,
                                                                           retain_every=1)[0, :, 0]

                unbounded_coordination[:, t] = inferred_unbounded_coordination
                acceptance_rates[:, t] = sampler.acceptance_rates_[-1]

        return unbounded_coordination, acceptance_rates

    def _get_unbounded_coordination_proposal(self, previous_unbounded_coordination_sample: np.ndarray, vcc: float):
        std = np.sqrt(self.var_unbounded_coordination_proposal)
        min_value = logit((1 - np.sqrt(1 - 4 * (vcc + EPSILON))) / 2)
        max_value = logit((1 + np.sqrt(1 - 4 * (vcc + EPSILON))) / 2)

        # Never propose a sample incompatible with the variance of the coordination distribution
        new_unbounded_coordination_sample = np.clip(norm(previous_unbounded_coordination_sample, std).rvs(),
                                                    a_min=min_value, a_max=max_value)

        if previous_unbounded_coordination_sample.shape[0] == 1:
            # The norm.rvs function does not preserve the dimensions of a unidimensional array.
            # We need to correct that if we are working with a single trial sample.
            new_unbounded_coordination_sample = np.array([[new_unbounded_coordination_sample]])

        # Hastings factor
        log_factor = 0

        return new_unbounded_coordination_sample, log_factor

    @staticmethod
    def _get_unbounded_coordination_posterior_unormalized_logprob(proposed_unbounded_coordination_sample: np.ndarray,
                                                                  previous_unbounded_coordination_sample: np.ndarray,
                                                                  next_unbounded_coordination_sample: Optional[
                                                                      np.ndarray],
                                                                  suc: float,
                                                                  vcc: float,
                                                                  coordination: np.ndarray):

        log_posterior = norm(previous_unbounded_coordination_sample, suc).logpdf(proposed_unbounded_coordination_sample)
        if next_unbounded_coordination_sample is not None:
            log_posterior += norm(proposed_unbounded_coordination_sample, suc).logpdf(
                next_unbounded_coordination_sample)

        m = sigmoid(proposed_unbounded_coordination_sample)
        # TODO var = np.minimum(m * (1 - m) - EPSILON, vcc)
        # log_posterior += beta(m, var).logpdf(coordination)
        log_posterior += beta(m, vcc).logpdf(coordination)

        return log_posterior.flatten()

    def _get_coordination_proposal(self, previous_coordination_sample: np.ndarray, vcc: float):

        # Since coordination is constrained to 0 and 1, we don't expect a high variance.
        # We set variance to be smaller than 0.01 such that MCMC don't do big jumps and ends up
        # overestimating coordination.

        # The variance in a beta distribution cannot be bigger than mean * (1 - mean). We never generate unbounded
        # coordination samples incompatible with the current vcc, but if the variance of the proposal is different,
        # we have to make sure we adjust it according to the magnitude of the unbounded coordination sample to avoid
        # ill-defined scenarios.
        m = previous_coordination_sample
        var = np.minimum(m * (1 - m) - EPSILON, self.var_coordination_proposal)
        min_value = np.maximum((1 - np.sqrt(1 - 4 * (vcc + EPSILON))) / 2, MIN_COORDINATION)
        max_value = np.minimum((1 + np.sqrt(1 - 4 * (vcc + EPSILON))) / 2, MAX_COORDINATION)
        new_coordination_sample = np.clip(beta(m, var).rvs(), a_min=min_value, a_max=max_value)

        if previous_coordination_sample.shape[0] == 1:
            # The norm.rvs function does not preserve the dimensions of a unidimensional array.
            # We need to correct that if we are working with a single trial sample.
            new_coordination_sample = np.array([[new_coordination_sample]])

        # Hastings factor.
        nominator = beta(new_coordination_sample, var).logpdf(previous_coordination_sample)
        denominator = beta(previous_coordination_sample, var).logpdf(new_coordination_sample)
        log_factor = nominator - denominator

        return new_coordination_sample, log_factor.flatten()

    def _get_coordination_posterior_unormalized_logprob(self,
                                                        proposed_coordination_sample: np.ndarray,
                                                        current_unbounded_coordination_sample: np.ndarray,
                                                        vcc: float,
                                                        saa: float,
                                                        evidence: LatentVocalicsDataset,
                                                        latent_vocalics: np.ndarray,
                                                        time_step: int):

        m = sigmoid(current_unbounded_coordination_sample)
        # TODO - var = np.minimum(m * (1 - m), vcc)
        # log_posterior = beta(m, var).logpdf(proposed_coordination_sample)
        log_posterior = beta(m, vcc).logpdf(proposed_coordination_sample)

        log_posterior = log_posterior.flatten()
        log_posterior += super()._get_latent_vocalics_term_for_coordination_posterior_unormalized_logprob(
            proposed_coordination_sample, saa, evidence, latent_vocalics, time_step)

        # For numerical stability, we never accept samples outside the boundaries
        # TODO - log_posterior = np.where(
        #     (proposed_coordination_sample.flatten() < MIN_COORDINATION) | (
        #             proposed_coordination_sample.flatten() > MAX_COORDINATION), -np.inf, log_posterior)

        return log_posterior

    def _retain_samples_from_latent(self, gibbs_step: int, latents: Any, time_steps: np.ndarray):
        super()._retain_samples_from_latent(gibbs_step, latents, time_steps)

        self.unbounded_coordination_samples_[gibbs_step][:, time_steps] = latents[2][:, time_steps]

        self.coordination_acceptance_rates_[:, time_steps] = latents[3][:, time_steps]
        self.unbounded_coordination_acceptance_rates_[:, time_steps] = latents[4][:, time_steps]

        if gibbs_step < self.unbounded_coordination_samples_.shape[0] - 1:
            self.unbounded_coordination_samples_[gibbs_step + 1] = self.unbounded_coordination_samples_[gibbs_step]

    def _update_latent_parameters_coordination(self, gibbs_step: int, evidence: LatentVocalicsDataset,
                                               logger: BaseLogger):
        if self.var_uc is None:
            a = self.a_vuc + evidence.num_trials * (evidence.num_time_steps - 1) / 2
            x = self.unbounded_coordination_samples_[gibbs_step, :, 1:]
            y = self.unbounded_coordination_samples_[gibbs_step, :, :evidence.num_time_steps - 1]
            b = self.b_vuc + np.square(x - y).sum() / 2
            self.vuc_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()

            if gibbs_step < len(self.vuc_samples_) - 1:
                self.vuc_samples_[gibbs_step + 1] = self.vuc_samples_[gibbs_step]

        if self.var_cc is None:
            # TODO: I am using a uniform prior for now (no prior). Reassess that later.
            # The variance is computed from the data directly. Do not use the first time step as variance is 0 in this
            # time since initial coordination is given.
            m = sigmoid(self.unbounded_coordination_samples_[gibbs_step])[:, 1:]
            # Max variance to keep compatibility with the samples.
            max_var = np.min(m * (1 - m) - EPSILON)
            self.vcc_samples_[gibbs_step] = np.clip(np.square(self.coordination_samples_[gibbs_step][:, 1:] - m).mean(),
                                                    a_min=EPSILON, a_max=max_var)

            if gibbs_step < len(self.vcc_samples_) - 1:
                self.vcc_samples_[gibbs_step + 1] = self.vcc_samples_[gibbs_step]

        logger.add_scalar("train/var_uc", self.vuc_samples_[gibbs_step], gibbs_step)
        logger.add_scalar("train/avg_ar_c", self.coordination_acceptance_rates_.mean(), gibbs_step)
        logger.add_scalar("train/avg_ar_uc", self.unbounded_coordination_acceptance_rates_.mean(), gibbs_step)

    def _retain_parameters(self):
        super()._retain_parameters()
        self.var_uc = self.vuc_samples_[-1]

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
    def _sample_coordination_from_prior(self, num_particles: int,
                                        new_particles: BetaCoordinationLatentVocalicsParticles,
                                        series: BetaCoordinationLatentVocalicsDataSeries):
        if series.unbounded_coordination is None:
            new_particles.unbounded_coordination = np.ones(num_particles) * logit(self.initial_coordination)
        else:
            new_particles.unbounded_coordination = np.ones(num_particles) * series.unbounded_coordination[0]

        if series.coordination is None:
            new_particles.coordination = np.ones(num_particles) * self.initial_coordination

            # # Any sample outside the boundaries of variance will have -inf log prob in the adjustment.
            # # For now, we adjust the variance such that we can sample from the beta distribution without any issues.
            # m = sigmoid(new_particles.unbounded_coordination)
            # # TODO - var = np.minimum(m * (1 - m) - EPSILON, self.var_cc)
            # # new_particles.coordination = np.clip(beta(m, var).rvs(), a_min=MIN_COORDINATION, a_max=MAX_COORDINATION)
            # new_particles.coordination = np.clip(beta(m, self.var_cc).rvs(), a_min=MIN_COORDINATION,
            #                                      a_max=MAX_COORDINATION)
        else:
            new_particles.coordination = np.ones(num_particles) * series.coordination[0]

    def _sample_coordination_from_transition_to(self, time_step: int,
                                                states: List[BetaCoordinationLatentVocalicsParticles],
                                                new_particles: BetaCoordinationLatentVocalicsParticles,
                                                series: BetaCoordinationLatentVocalicsDataSeries):
        previous_particles = states[time_step - 1]
        if series.unbounded_coordination is None:
            min_value = logit((1 - np.sqrt(1 - 4 * (self.var_cc + EPSILON))) / 2)
            max_value = logit((1 + np.sqrt(1 - 4 * (self.var_cc + EPSILON))) / 2)
            # new_particles.unbounded_coordination = np.clip(norm(previous_particles.unbounded_coordination,
            #                                                     np.sqrt(self.var_uc)).rvs(),
            #                                                a_min=logit(MIN_COORDINATION), a_max=logit(MAX_COORDINATION))
            new_particles.unbounded_coordination = np.clip(norm(previous_particles.unbounded_coordination,
                                                                np.sqrt(self.var_uc)).rvs(), a_min=min_value,
                                                           a_max=max_value)
        else:
            num_particles = len(previous_particles.unbounded_coordination)
            new_particles.unbounded_coordination = np.ones(num_particles) * series.unbounded_coordination[time_step]

        if series.coordination is None:
            # Any sample outside the boundaries of variance will have -inf log prob in the adjustment.
            # For now, we adjust the variance such that we can sample from the beta distribution without any issues.
            m = sigmoid(new_particles.unbounded_coordination)
            # TODO - var = np.minimum(m * (1 - m) - EPSILON, self.var_cc)
            # new_particles.coordination = np.clip(beta(m, var).rvs(), a_min=MIN_COORDINATION, a_max=MAX_COORDINATION)
            new_particles.coordination = np.clip(beta(m, self.var_cc).rvs(), a_min=MIN_COORDINATION,
                                                 a_max=MAX_COORDINATION)
        else:
            num_particles = len(previous_particles.coordination)
            new_particles.coordination = np.ones(num_particles) * series.coordination[time_step]

    def _create_new_particles(self) -> BetaCoordinationLatentVocalicsParticles:
        return BetaCoordinationLatentVocalicsParticles()

    def _calculate_evidence_log_likelihood_at(self, time_step: int,
                                              states: List[BetaCoordinationLatentVocalicsParticles],
                                              series: LatentVocalicsDataSeries):
        ll = super()._calculate_evidence_log_likelihood_at(time_step, states, series)

        m = sigmoid(states[time_step].unbounded_coordination)
        if series.coordination is not None:
            # TODO - var = np.minimum(m * (1 - m) - EPSILON, self.var_cc)
            # ll += beta(m, var).logpdf(series.coordination[time_step])
            ll += beta(m, self.var_cc).logpdf(series.coordination[time_step])

        # Samples that are incompatible with the variance of the beta distribution, should be rejected.
        # TODO - ll[m * (1 - m) <= self.var_cc] = np.log(EPSILON)

        return ll

    def _resample_at(self, time_step: int, series: BetaCoordinationLatentVocalicsDataSeries):
        if series.is_complete:
            return False

        if series.coordination is not None and series.unbounded_coordination is None:
            return True
        elif series.latent_vocalics is None:
            return series.observed_vocalics.mask[time_step] == 1
        else:
            # Only coordination is latent, but we only need to sample it if there's a link between the
            # current vocalics and previous vocalics from a different speaker.
            return series.observed_vocalics.mask[time_step] == 1 and series.observed_vocalics.previous_from_other[
                time_step] is not None

    def _summarize_particles(self, series: BetaCoordinationLatentVocalicsDataSeries,
                             particles: List[
                                 BetaCoordinationLatentVocalicsParticles]) -> BetaCoordinationLatentVocalicsParticlesSummary:

        summary = super()._summarize_particles(series, particles)
        summary = BetaCoordinationLatentVocalicsParticlesSummary.from_latent_vocalics_particles_summary(summary)
        summary.unbounded_coordination_mean = np.zeros_like(summary.coordination_mean)
        summary.unbounded_coordination_var = np.zeros_like(summary.coordination_var)

        for t, particles_in_time in enumerate(particles):
            summary.unbounded_coordination_mean[t] = particles_in_time.unbounded_coordination.mean()
            summary.unbounded_coordination_var[t] = particles_in_time.unbounded_coordination.var()

        return summary
