from __future__ import annotations
from typing import List, Optional, Tuple

from multiprocessing import Pool

import numpy as np
from scipy.stats import invgamma, norm
from tqdm import tqdm

from coordination.common.log import BaseLogger
from coordination.common.distribution import beta
from coordination.common.utils import logit, sigmoid
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingLatentVocalics
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsParticles, \
    BetaCoordinationLatentVocalicsParticlesSummary, BetaCoordinationLatentVocalicsSamples, \
    BetaCoordinationLatentVocalicsDataSeries, BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationLatentVocalicsTrainingHyperParameters, BetaCoordinationLatentVocalicsModelParameters
from coordination.model.utils.coordination_blending_latent_vocalics import BaseF, BaseG
from coordination.inference.mcmc import MCMC

# For numerical stability
EPSILON = 1e-6
MIN_COORDINATION = 2 * EPSILON
MAX_COORDINATION = 1 - MIN_COORDINATION


class BetaCoordinationBlendingLatentVocalics(
    CoordinationBlendingLatentVocalics[
        BetaCoordinationLatentVocalicsSamples, BetaCoordinationLatentVocalicsParticlesSummary]):

    def __init__(self,
                 initial_coordination: float,
                 num_vocalic_features: int,
                 num_speakers: int,
                 f: BaseF = BaseF(),
                 g: BaseG = BaseG()):
        super().__init__(initial_coordination, num_vocalic_features, num_speakers, f, g)

        # Trainable parameters of the model
        self.parameters = BetaCoordinationLatentVocalicsModelParameters()

        # Samples collected during training
        self.vu_samples_ = np.array([])

        # Acceptance rate in the samples of coordination over time in the last Gibbs Step.
        # We log this to keep track of the MCMC sampler's health.
        self.unbounded_coordination_acceptance_rates_ = np.array([])
        self.coordination_acceptance_rates_ = np.array([])

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------
    def _generate_coordination_samples(self, num_samples: int, num_time_steps: int,
                                       samples: BetaCoordinationLatentVocalicsSamples):
        samples.unbounded_coordination = np.zeros((num_samples, num_time_steps))
        samples.coordination = np.zeros((num_samples, num_time_steps))

        std_u = np.sqrt(self.parameters.var_u)

        for t in tqdm(range(num_time_steps), desc="Coordination", position=0, leave=False):
            if t == 0:
                samples.unbounded_coordination[:, 0] = logit(self.initial_coordination)
                samples.coordination[:, 0] = self.initial_coordination
            else:
                # The variance of a beta distribution, cannot be bigger than m * (1 - m). Therefore, we
                # constrain the sampled from the unbounded distribution such that we cannot generate
                # beta distributions with impossible means when we compute coordination.
                min_value = logit((1 - np.sqrt(1 - 4 * (self.parameters.var_c + EPSILON))) / 2)
                max_value = logit((1 + np.sqrt(1 - 4 * (self.parameters.var_c + EPSILON))) / 2)
                mean = samples.unbounded_coordination[:, t - 1]
                samples.unbounded_coordination[:, t] = np.clip(norm(mean, std_u).rvs(), a_min=min_value,
                                                               a_max=max_value)

                m = sigmoid(samples.unbounded_coordination[:, t])
                samples.coordination[:, t] = np.clip(beta(m, self.parameters.var_c).rvs(), a_min=MIN_COORDINATION,
                                                     a_max=MAX_COORDINATION)

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------
    def _initialize_coordination_parameters_for_fit(self, evidence: BetaCoordinationLatentVocalicsDataset,
                                                    train_hyper_parameters: BetaCoordinationLatentVocalicsTrainingHyperParameters,
                                                    burn_in: int, seed: int):
        # Unbounded coordination
        self.vu_samples_ = np.zeros(burn_in + 1)

        if self.parameters.var_u_frozen:
            self.vu_samples_[:] = self.parameters.var_u
        else:
            self.vu_samples_[0] = train_hyper_parameters.vu0
            self.parameters.set_var_u(train_hyper_parameters.vu0, freeze=False)

            if burn_in > 0:
                self.vu_samples_[1] = self.vu_samples_[0]

        # Coordination
        if self.parameters.var_c_frozen:
            self.vc_samples_[:] = self.parameters.var_c
        else:
            self.vc_samples_[0] = train_hyper_parameters.vc0
            self.parameters.set_var_c(train_hyper_parameters.vc0, freeze=False)

            if burn_in > 0:
                self.vc_samples_[1] = self.vc_samples_[0]

    def _initialize_coordination_for_fit(self, evidence: BetaCoordinationLatentVocalicsDataset,
                                         train_hyper_parameters: BetaCoordinationLatentVocalicsTrainingHyperParameters,
                                         burn_in: int, seed: int):
        burn_in = self.coordination_samples_.shape[0] - 1

        self.vu_samples_ = np.zeros(burn_in + 1)
        self.unbounded_coordination_samples_ = np.zeros((burn_in + 1, evidence.num_trials, evidence.num_time_steps))
        self.unbounded_coordination_acceptance_rates_ = np.ones((evidence.num_trials, evidence.num_time_steps))
        self.coordination_acceptance_rates_ = np.ones((evidence.num_trials, evidence.num_time_steps))

        if evidence.unbounded_coordination is None:
            # An alternative is to initialize with the transition distribution by calling
            # _initialize_unbounded_coordination_from_transition_distribution. This approach can lead to fast
            # or slow convergence depending on the scenario.
            self.unbounded_coordination_samples_[0, :, 0] = logit(self.initial_coordination)
            if self.unbounded_coordination_samples_.shape[0] > 0:
                self.unbounded_coordination_samples_[1] = self.unbounded_coordination_samples_[0]
        else:
            self.unbounded_coordination_samples_[:] = evidence.unbounded_coordination[np.newaxis, :]

        if evidence.coordination is None:
            m = sigmoid(self.unbounded_coordination_samples_[0])
            vc = self.parameters.var_c

            # We don't let coordination samples be 0 or 1 for numerical stability.
            self.coordination_samples_[0] = np.clip(beta(m, vc).rvs(), a_min=MIN_COORDINATION, a_max=MAX_COORDINATION)
            self.coordination_samples_[0, :, 0] = self.initial_coordination

            if self.coordination_samples_.shape[0] > 0:
                self.coordination_samples_[1] = self.coordination_samples_[0]
        else:
            self.coordination_samples_[:] = evidence.coordination[np.newaxis, :]

    def _initialize_unbounded_coordination_from_transition_distribution(self,
                                                                        evidence: BetaCoordinationLatentVocalicsDataset):
        su = np.sqrt(self.parameters.var_u)
        vc = self.parameters.var_c
        min_value = logit((1 - np.sqrt(1 - 4 * (vc + EPSILON))) / 2)
        max_value = logit((1 + np.sqrt(1 - 4 * (vc + EPSILON))) / 2)
        self.unbounded_coordination_samples_[0, :, 0] = logit(self.initial_coordination)
        for t in range(1, evidence.num_time_steps):
            self.unbounded_coordination_samples_[0, :, t] = np.clip(
                norm(self.unbounded_coordination_samples_[0, :, t - 1],
                     su).rvs(), a_min=min_value, a_max=max_value)

    def _compute_coordination_loglikelihood(self, gibbs_step: int,
                                            evidence: BetaCoordinationLatentVocalicsDataset) -> float:
        unbounded_coordination = self.unbounded_coordination_samples_[gibbs_step]
        coordination = self.coordination_samples_[gibbs_step]

        su = np.sqrt(self.parameters.var_u)
        vc = self.parameters.var_c

        ll = 0
        for t in range(1, evidence.num_time_steps):
            # Initial coordination is given
            ll += norm(unbounded_coordination[:, t - 1], su).logpdf(unbounded_coordination[:, t]).sum()

            m = sigmoid(unbounded_coordination[:, t])
            ll += beta(m, vc).logpdf(coordination[:, t]).sum()

        return ll

    def _update_coordination(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                             train_hyper_parameters: BetaCoordinationLatentVocalicsTrainingHyperParameters,
                             pool: Pool):
        if self._get_max_num_jobs() == 1:
            # Update coordination in the main process
            time_steps_in_job = np.arange(evidence.num_time_steps)

            unbounded_coordination, acceptance_rates = self._sample_unbounded_coordination_on_fit(gibbs_step, evidence,
                                                                                                  train_hyper_parameters,
                                                                                                  time_steps_in_job, 1,
                                                                                                  0)
            self._retain_samples_from_unbounded_coordination(gibbs_step, unbounded_coordination, acceptance_rates,
                                                             time_steps_in_job)

            coordination, acceptance_rates = self._sample_coordination_on_fit(gibbs_step, evidence,
                                                                              train_hyper_parameters, time_steps_in_job,
                                                                              1, 0)
            self._retain_samples_from_coordination(gibbs_step, coordination, acceptance_rates, time_steps_in_job)

        else:
            # Unbounded coordination is split in two groups because of the dependency over time.
            groups = [self._unbounded_coordination_time_step_blocks1, self._unbounded_coordination_time_steps_blocks2]
            for g, time_step_blocks in enumerate(groups):
                job_args = [(gibbs_step, evidence, train_hyper_parameters, time_step_blocks[j], j + 1, g) for j in
                            range(len(time_step_blocks))]

                for block_idx, (unbounded_coordination, acceptance_rates) in enumerate(
                        pool.starmap(self._sample_unbounded_coordination_on_fit, job_args)):
                    self._retain_samples_from_unbounded_coordination(gibbs_step, unbounded_coordination,
                                                                     acceptance_rates,
                                                                     time_step_blocks[block_idx])

            # Coordination does not depend on other coordination at different time steps, thus we can parallelize a
            # single group of time steps blocks.
            job_args = [(gibbs_step, evidence, train_hyper_parameters, self._coordination_time_step_blocks[j], j + 1, 0)
                        for j in
                        range(len(self._coordination_time_step_blocks))]
            for block_idx, (coordination, acceptance_rates) in enumerate(
                    pool.starmap(self._sample_coordination_on_fit, job_args)):
                self._retain_samples_from_coordination(gibbs_step, coordination, acceptance_rates,
                                                       self._coordination_time_step_blocks[block_idx])

    def _sample_unbounded_coordination_on_fit(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                                              train_hyper_parameters: BetaCoordinationLatentVocalicsTrainingHyperParameters,
                                              time_steps: np.ndarray, job_num: int, group_order: int) -> Tuple[
        np.ndarray, np.ndarray]:

        unbounded_coordination = self.unbounded_coordination_samples_[gibbs_step].copy()
        acceptance_rates = self.unbounded_coordination_acceptance_rates_.copy()

        if evidence.unbounded_coordination is not None:
            return unbounded_coordination, acceptance_rates

        # The retain method copies the estimate in one gibbs step to the next one. Therefore, accessing the values in
        # the current gibbs step will give us the latest values of the estimates.
        coordination = self.coordination_samples_[gibbs_step]

        su = np.sqrt(self.parameters.var_u)
        vc = self.parameters.var_c

        for t in tqdm(time_steps, desc=f"Sampling Unbounded Coordination (Group {group_order + 1})", position=job_num,
                      leave=False):
            if t > 0:
                next_unbounded_coordination = None if t == unbounded_coordination.shape[
                    1] - 1 else unbounded_coordination[:, t + 1][:, np.newaxis]

                proposal_fn_params = {
                    "vc": vc,
                    "su_prop": np.sqrt(train_hyper_parameters.vu_mcmc_prop)
                }

                log_prob_fn_params = {
                    "previous_unbounded_coordination_sample": unbounded_coordination[:, t - 1][:, np.newaxis],
                    "next_unbounded_coordination_sample": next_unbounded_coordination,
                    "su": su,
                    "vc": vc,
                    "coordination": coordination[:, t][:, np.newaxis]
                }

                sampler = MCMC(proposal_fn=self._get_unbounded_coordination_proposal,
                               proposal_fn_kwargs=proposal_fn_params,
                               log_prob_fn=self._get_unbounded_coordination_posterior_unormalized_logprob,
                               log_prob_fn_kwargs=log_prob_fn_params)
                initial_sample = unbounded_coordination[:, t][:, np.newaxis]
                inferred_unbounded_coordination = sampler.generate_samples(initial_sample=initial_sample,
                                                                           num_samples=1,
                                                                           burn_in=train_hyper_parameters.u_mcmc_iter,
                                                                           retain_every=1)[0, :, 0]

                unbounded_coordination[:, t] = inferred_unbounded_coordination
                acceptance_rates[:, t] = sampler.acceptance_rates_[-1]

        return unbounded_coordination, acceptance_rates

    @staticmethod
    def _get_unbounded_coordination_proposal(previous_unbounded_coordination_sample: np.ndarray, vc: float,
                                             su_prop: float):
        min_value = logit((1 - np.sqrt(1 - 4 * (vc + EPSILON))) / 2)
        max_value = logit((1 + np.sqrt(1 - 4 * (vc + EPSILON))) / 2)

        # Never propose a sample incompatible with the variance of the coordination distribution
        new_unbounded_coordination_sample = np.clip(norm(previous_unbounded_coordination_sample, su_prop).rvs(),
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
                                                                  su: float,
                                                                  vc: float,
                                                                  coordination: np.ndarray):

        log_posterior = norm(previous_unbounded_coordination_sample, su).logpdf(proposed_unbounded_coordination_sample)
        if next_unbounded_coordination_sample is not None:
            log_posterior += norm(proposed_unbounded_coordination_sample, su).logpdf(
                next_unbounded_coordination_sample)

        m = sigmoid(proposed_unbounded_coordination_sample)
        log_posterior += beta(m, vc).logpdf(coordination)

        return log_posterior.flatten()

    def _sample_coordination_on_fit(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                                    train_hyper_parameters: BetaCoordinationLatentVocalicsTrainingHyperParameters,
                                    time_steps: np.ndarray, job_num: int, group_order: int) -> Tuple[
        np.ndarray, np.ndarray]:

        coordination = self.coordination_samples_[gibbs_step].copy()
        acceptance_rates = self.coordination_acceptance_rates_.copy()

        if evidence.coordination is not None:
            return coordination, acceptance_rates

        unbounded_coordination = self.unbounded_coordination_samples_[gibbs_step]
        # The retain method copies the estimate in one gibbs step to the next one. Therefore, accessing the values in
        # the current gibbs step will give us the latest values of the estimates.
        latent_vocalics = self.latent_vocalics_samples_[gibbs_step]

        vc = self.parameters.var_c
        saa = np.sqrt(self.parameters.var_aa)

        for t in tqdm(time_steps, desc=f"Sampling Coordination (Group {group_order + 1})", position=job_num,
                      leave=False):
            if t > 0:
                # Initial coordination is given
                proposal_fn_params = {
                    "vc": vc,
                    "vc_prop": train_hyper_parameters.vc_mcmc_prop
                }

                log_prob_fn_params = {
                    "current_unbounded_coordination_sample": unbounded_coordination[:, t][:, np.newaxis],
                    "vc": vc,
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
                                                                 burn_in=train_hyper_parameters.c_mcmc_iter,
                                                                 retain_every=1)[0, :, 0]
                coordination[:, t] = inferred_coordination
                acceptance_rates[:, t] = sampler.acceptance_rates_[-1]

        return coordination, acceptance_rates

    @staticmethod
    def _get_coordination_proposal(previous_coordination_sample: np.ndarray, vc: float, vc_prop: float):

        # Since coordination is constrained to 0 and 1, we don't expect a high variance.
        # We set variance to be smaller than 0.01 such that MCMC don't do big jumps and ends up
        # overestimating coordination.

        # The variance in a beta distribution cannot be bigger than mean * (1 - mean). We never generate unbounded
        # coordination samples incompatible with the current vcc, but if the variance of the proposal is different,
        # we have to make sure we adjust it according to the magnitude of the unbounded coordination sample to avoid
        # ill-defined scenarios.
        m = previous_coordination_sample
        var = np.minimum(m * (1 - m) - EPSILON, vc_prop)
        min_value = np.maximum((1 - np.sqrt(1 - 4 * (vc + EPSILON))) / 2, MIN_COORDINATION)
        max_value = np.minimum((1 + np.sqrt(1 - 4 * (vc + EPSILON))) / 2, MAX_COORDINATION)
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
                                                        vc: float,
                                                        saa: float,
                                                        evidence: BetaCoordinationLatentVocalicsDataset,
                                                        latent_vocalics: np.ndarray,
                                                        time_step: int):

        m = sigmoid(current_unbounded_coordination_sample)
        log_posterior = beta(m, vc).logpdf(proposed_coordination_sample)

        log_posterior = log_posterior.flatten()
        log_posterior += super()._get_latent_vocalics_term_for_coordination_posterior_unormalized_logprob(
            proposed_coordination_sample, saa, evidence, latent_vocalics, time_step)

        return log_posterior

    def _retain_samples_from_coordination(self, gibbs_step: int, coordination: np.ndarray, acceptance_rates: np.ndarray,
                                          time_steps: np.ndarray):
        self.coordination_samples_[gibbs_step][:, time_steps] = coordination[:, time_steps]
        self.coordination_acceptance_rates_[:, time_steps] = acceptance_rates[:, time_steps]

        if gibbs_step < self.coordination_samples_.shape[0] - 1:
            self.coordination_samples_[gibbs_step + 1] = self.coordination_samples_[gibbs_step]

    def _retain_samples_from_unbounded_coordination(self, gibbs_step: int, unbounded_coordination: np.ndarray,
                                                    acceptance_rates: np.ndarray, time_steps: np.ndarray):
        self.unbounded_coordination_samples_[gibbs_step][:, time_steps] = unbounded_coordination[:, time_steps]
        self.unbounded_coordination_acceptance_rates_[:, time_steps] = acceptance_rates[:, time_steps]

        if gibbs_step < self.unbounded_coordination_samples_.shape[0] - 1:
            self.unbounded_coordination_samples_[gibbs_step + 1] = self.unbounded_coordination_samples_[gibbs_step]

    def _update_parameters_coordination(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                                        train_hyper_parameters: BetaCoordinationLatentVocalicsTrainingHyperParameters):
        if not self.parameters.var_u_frozen:
            a = train_hyper_parameters.a_vu + evidence.num_trials * (evidence.num_time_steps - 1) / 2
            x = self.unbounded_coordination_samples_[gibbs_step, :, 1:]
            y = self.unbounded_coordination_samples_[gibbs_step, :, :evidence.num_time_steps - 1]
            b = train_hyper_parameters.b_vu + np.square(x - y).sum() / 2
            self.vu_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()

            self.parameters.set_var_u(self.vu_samples_[gibbs_step], freeze=False)

            if gibbs_step < len(self.vu_samples_) - 1:
                self.vu_samples_[gibbs_step + 1] = self.vu_samples_[gibbs_step]

        if not self.parameters.var_c_frozen:
            # The variance is computed from the data directly. Do not use the first time step as variance is 0 in this
            # time since initial coordination is given.
            m = sigmoid(self.unbounded_coordination_samples_[gibbs_step])[:, 1:]
            # Max variance to keep compatibility with the samples.
            max_var = np.min(m * (1 - m) - EPSILON)
            self.vc_samples_[gibbs_step] = np.clip(np.square(self.coordination_samples_[gibbs_step][:, 1:] - m).mean(),
                                                   a_min=EPSILON, a_max=max_var)

            self.parameters.set_var_c(self.vc_samples_[gibbs_step], freeze=False)

            if gibbs_step < len(self.vc_samples_) - 1:
                self.vc_samples_[gibbs_step + 1] = self.vc_samples_[gibbs_step]

    def _log_parameters(self, gibbs_step: int, logger: BaseLogger):
        super()._log_parameters(gibbs_step, logger)
        logger.add_scalar("train/var_u", self.vu_samples_[gibbs_step], gibbs_step)
        logger.add_scalar("train/avg_ar_c", self.coordination_acceptance_rates_.mean(), gibbs_step)
        logger.add_scalar("train/avg_ar_u", self.unbounded_coordination_acceptance_rates_.mean(), gibbs_step)

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
        else:
            new_particles.coordination = np.ones(num_particles) * series.coordination[0]

    def _sample_coordination_from_transition_to(self, time_step: int,
                                                states: List[BetaCoordinationLatentVocalicsParticles],
                                                new_particles: BetaCoordinationLatentVocalicsParticles,
                                                series: BetaCoordinationLatentVocalicsDataSeries):
        previous_particles = states[time_step - 1]
        if series.unbounded_coordination is None:
            min_value = logit((1 - np.sqrt(1 - 4 * (self.parameters.var_c + EPSILON))) / 2)
            max_value = logit((1 + np.sqrt(1 - 4 * (self.parameters.var_c + EPSILON))) / 2)
            new_particles.unbounded_coordination = np.clip(norm(previous_particles.unbounded_coordination,
                                                                np.sqrt(self.parameters.var_u)).rvs(), a_min=min_value,
                                                           a_max=max_value)
        else:
            num_particles = len(previous_particles.unbounded_coordination)
            new_particles.unbounded_coordination = np.ones(num_particles) * series.unbounded_coordination[time_step]

        if series.coordination is None:
            # Any sample outside the boundaries of variance will have -inf log prob in the adjustment.
            # For now, we adjust the variance such that we can sample from the beta distribution without any issues.
            m = sigmoid(new_particles.unbounded_coordination)
            new_particles.coordination = np.clip(beta(m, self.parameters.var_c).rvs(), a_min=MIN_COORDINATION,
                                                 a_max=MAX_COORDINATION)
        else:
            num_particles = len(previous_particles.coordination)
            new_particles.coordination = np.ones(num_particles) * series.coordination[time_step]

    def _create_new_particles(self) -> BetaCoordinationLatentVocalicsParticles:
        return BetaCoordinationLatentVocalicsParticles()

    def _calculate_evidence_log_likelihood_at(self, time_step: int,
                                              states: List[BetaCoordinationLatentVocalicsParticles],
                                              series: BetaCoordinationLatentVocalicsDataSeries):
        ll = super()._calculate_evidence_log_likelihood_at(time_step, states, series)

        m = sigmoid(states[time_step].unbounded_coordination)
        if series.coordination is not None:
            ll += beta(m, self.parameters.var_c).logpdf(series.coordination[time_step])

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
