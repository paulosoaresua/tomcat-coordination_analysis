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
                 f: Callable = default_f,
                 g: Callable = default_g):
        super().__init__(initial_coordination, num_vocalic_features, num_speakers, a_vcc, b_vcc, a_va, b_va, a_vaa,
                         b_vaa, a_vo, b_vo, f, g)

        # var_cc in this model is the dispersion of the reparameterized beta

        self.var_uc: Optional[float] = None

        self.a_vuc = a_vuc
        self.b_vuc = b_vuc

        self._hyper_params["a_vuc"] = a_vuc
        self._hyper_params["b_vuc"] = b_vuc

        self.vuc_samples_ = np.array([])

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
            else:
                # The mean of a truncated Gaussian distribution is given by mu + an offset. We remove the offset here,
                # such that the previous sample is indeed the mean of the truncated Gaussian.
                mean = samples.unbounded_coordination[:, t - 1]
                samples.unbounded_coordination[:, t] = norm(mean, suc).rvs()

            m = sigmoid(samples.unbounded_coordination[:, t])
            samples.coordination[:, t] = np.clip(beta(m, self.var_cc).rvs(), a_min=EPSILON, a_max=1 - EPSILON)

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------
    def _initialize_coordination_for_gibbs(self, evidence: BetaCoordinationLatentVocalicsDataset):
        burn_in = self.coordination_samples_.shape[0] - 1

        self.vuc_samples_ = np.zeros(burn_in + 1)
        self.unbounded_coordination_samples_ = np.zeros((burn_in + 1, evidence.num_trials, evidence.num_time_steps))

        if evidence.unbounded_coordination is None:
            means = np.zeros((evidence.num_trials, evidence.num_time_steps))
            self.unbounded_coordination_samples_[0] = norm(means, 0.1).rvs()
            self.unbounded_coordination_samples_[0, :, 0] = logit(self.initial_coordination)

            if self.unbounded_coordination_samples_.shape[0] > 0:
                self.unbounded_coordination_samples_[1] = self.unbounded_coordination_samples_[0]
        else:
            self.unbounded_coordination_samples_[:] = evidence.unbounded_coordination[np.newaxis, :]

        if evidence.coordination is None:
            self.coordination_samples_[0] = np.clip(beta(sigmoid(self.unbounded_coordination_samples_[0]), 0.01).rvs(),
                                                    a_min=EPSILON, a_max=1 - EPSILON)

            if self.coordination_samples_.shape[0] > 0:
                self.coordination_samples_[1] = self.coordination_samples_[0]
        else:
            self.coordination_samples_[:] = evidence.coordination[np.newaxis, :]

        # Parameters of the unbounded coordination distribution
        if self.var_uc is None:
            self.vuc_samples_[0] = invgamma(a=self.a_vuc, scale=self.b_vuc).rvs()
            if burn_in > 0:
                self.vuc_samples_[1] = self.vuc_samples_[0]
        else:
            self.vuc_samples_[:] = self.var_uc

        if self.var_cc is None:
            # TODO: uniform prior. It has to be between 0 and 1
            self.vcc_samples_[0] = np.random.uniform(EPSILON, 1 - EPSILON)

            if self.vcc_samples_.shape[0] > 0:
                self.vcc_samples_[1] = self.vcc_samples_[0]
        else:
            self.vcc_samples_[:] = self.var_cc

    def _compute_coordination_likelihood(self, gibbs_step: int,
                                         evidence: BetaCoordinationLatentVocalicsDataset) -> float:
        unbounded_coordination = self.unbounded_coordination_samples_[gibbs_step]
        coordination = self.coordination_samples_[gibbs_step]

        suc = np.sqrt(self.vuc_samples_[gibbs_step])
        vcc = self.vcc_samples_[gibbs_step]

        ll = 0
        for t in range(evidence.num_time_steps):
            # Coordination transition
            if t > 0:
                ll += norm(unbounded_coordination[:, t - 1], suc).logpdf(unbounded_coordination[:, t]).sum()

            ll += beta(sigmoid(unbounded_coordination[:, t]), vcc).logpdf(coordination[:, t]).sum()

        return ll

    def _sample_coordination_on_fit(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                                    time_steps: np.ndarray,
                                    job_num: int) -> Tuple[np.ndarray, ...]:

        unbounded_coordination = self._sample_unbounded_coordination_on_fit(gibbs_step, evidence, time_steps, job_num)

        if evidence.coordination is not None:
            return self.coordination_samples_[gibbs_step - 1].copy(), unbounded_coordination

        # The retain method copies the estimate in one gibbs step to the next one. Therefore, accessing the values in
        # the current gibbs step will give us the latest values of the estimates.
        coordination = self.coordination_samples_[gibbs_step].copy()
        latent_vocalics = self.latent_vocalics_samples_[gibbs_step]

        if evidence.coordination is None:
            vcc = self.vcc_samples_[gibbs_step]
            saa = np.sqrt(self.vaa_samples_[gibbs_step])

            for t in tqdm(time_steps, desc="Sampling Coordination", position=job_num, leave=False):
                if t > 0:
                    proposal_fn_params = {
                        "previous_unbounded_coordination_sample": unbounded_coordination[:, t - 1][:, np.newaxis]
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
                                                                     burn_in=50,
                                                                     retain_every=1)[0, :, 0]
                    coordination[:, t] = inferred_coordination

        return coordination, unbounded_coordination[:, time_steps]

    def _sample_unbounded_coordination_on_fit(self, gibbs_step: int, evidence: BetaCoordinationLatentVocalicsDataset,
                                              time_steps: np.ndarray, job_num: int) -> np.ndarray:

        if evidence.unbounded_coordination is not None:
            return self.unbounded_coordination_samples_[gibbs_step - 1].copy()

        # The retain method copies the estimate in one gibbs step to the next one. Therefore, accessing the values in
        # the current gibbs step will give us the latest values of the estimates.
        unbounded_coordination = self.coordination_samples_[gibbs_step].copy()
        coordination = self.coordination_samples_[gibbs_step]

        suc = np.sqrt(self.vuc_samples_[gibbs_step])
        vcc = self.vcc_samples_[gibbs_step]

        for t in tqdm(time_steps, desc="Sampling Unbounded Coordination", position=job_num, leave=False):
            if t > 0:
                next_unbounded_coordination = None if t == unbounded_coordination.shape[
                    1] - 1 else unbounded_coordination[:, t + 1][:, np.newaxis]

                log_prob_fn_params = {
                    "previous_unbounded_coordination_sample": unbounded_coordination[:, t - 1][:, np.newaxis],
                    "next_unbounded_coordination_sample": next_unbounded_coordination,
                    "suc": suc,
                    "vcc": vcc,
                    "coordination": coordination[:, t][:, np.newaxis]
                }

                sampler = MCMC(proposal_fn=self._get_unbounded_coordination_proposal,
                               proposal_fn_kwargs={},
                               log_prob_fn=self._get_unbounded_coordination_posterior_unormalized_logprob,
                               log_prob_fn_kwargs=log_prob_fn_params)
                initial_sample = unbounded_coordination[:, t][:, np.newaxis]
                inferred_unbounded_coordination = sampler.generate_samples(initial_sample=initial_sample,
                                                                           num_samples=1,
                                                                           burn_in=50,
                                                                           retain_every=1)[0, :, 0]
                unbounded_coordination[:, t] = inferred_unbounded_coordination

        return unbounded_coordination

    @staticmethod
    def _get_unbounded_coordination_proposal(previous_unbounded_coordination_sample: np.ndarray):
        std = 1
        new_unbounded_coordination_sample = norm(previous_unbounded_coordination_sample, std).rvs()

        if previous_unbounded_coordination_sample.shape[0] == 1:
            # The norm.rvs function does not preserve the dimensions of a unidimensional array.
            # We need to correct that if we are working with a single trial sample.
            new_unbounded_coordination_sample = np.array([[new_unbounded_coordination_sample]])

        # Hastings factor
        factor = 1

        return new_unbounded_coordination_sample, factor

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

        log_posterior += beta(sigmoid(proposed_unbounded_coordination_sample), vcc).logpdf(coordination)

        return log_posterior.flatten()

    @staticmethod
    def _get_coordination_proposal(previous_coordination_sample: np.ndarray,
                                   previous_unbounded_coordination_sample: np.ndarray):

        # Since coordination is constrained to 0 and 1, we don't expect a high variance.
        # We set variance to be smaller than 0.01 such that MCMC don't do big jumps and ends up
        # overestimating coordination.
        var = 0.05
        # new_coordination_sample = np.clip(beta(sigmoid(previous_unbounded_coordination_sample), var).rvs(),
        #                                   a_min=EPSILON, a_max=1 - EPSILON)
        new_coordination_sample = np.clip(beta(previous_coordination_sample, var).rvs(), a_min=EPSILON,
                                          a_max=1 - EPSILON)

        if previous_coordination_sample.shape[0] == 1:
            # The norm.rvs function does not preserve the dimensions of a unidimensional array.
            # We need to correct that if we are working with a single trial sample.
            new_coordination_sample = np.array([[new_coordination_sample]])

        # Hastings factor
        # nominator = beta(new_coordination_sample, var).logpdf(sigmoid(previous_unbounded_coordination_sample))
        # denominator = beta(sigmoid(previous_unbounded_coordination_sample), var).logpdf(new_coordination_sample)
        nominator = beta(new_coordination_sample, var).logpdf(previous_coordination_sample)
        denominator = beta(previous_coordination_sample, var).logpdf(new_coordination_sample)
        factor = np.exp(nominator - denominator)

        return new_coordination_sample, factor.flatten()

    def _get_coordination_posterior_unormalized_logprob(self,
                                                        proposed_coordination_sample: np.ndarray,
                                                        current_unbounded_coordination_sample: np.ndarray,
                                                        vcc: float,
                                                        saa: float,
                                                        evidence: LatentVocalicsDataset,
                                                        latent_vocalics: np.ndarray,
                                                        time_step: int):

        log_posterior = beta(sigmoid(current_unbounded_coordination_sample), vcc).logpdf(proposed_coordination_sample)

        log_posterior = log_posterior.flatten()
        log_posterior += super()._get_latent_vocalics_term_for_coordination_posterior_unormalized_logprob(
            proposed_coordination_sample, saa, evidence, latent_vocalics, time_step)

        return log_posterior

    def _retain_samples_from_latent(self, gibbs_step: int, latents: Any, time_steps: np.ndarray):
        super()._retain_samples_from_latent(gibbs_step, latents, time_steps)

        self.unbounded_coordination_samples_[gibbs_step][:, time_steps] = latents[2][:, time_steps]

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
            # The variance in this model is actually the dispersion of the reparameterized beta
            m = sigmoid(self.unbounded_coordination_samples_[gibbs_step])
            var = np.square(self.coordination_samples_[gibbs_step] - m)
            s = np.mean(var / (m * (1 - m)))
            self.vcc_samples_[gibbs_step] = s

            if gibbs_step < len(self.vcc_samples_) - 1:
                self.vcc_samples_[gibbs_step + 1] = self.vcc_samples_[gibbs_step]

        logger.add_scalar("train/var_uc", self.vuc_samples_[gibbs_step], gibbs_step)

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
            new_particles.coordination = beta(sigmoid(new_particles.unbounded_coordination), self.var_cc)
        else:
            new_particles.coordination = np.ones(num_particles) * series.coordination[0]

    def _sample_coordination_from_transition_to(self, time_step: int,
                                                states: List[BetaCoordinationLatentVocalicsParticles],
                                                new_particles: BetaCoordinationLatentVocalicsParticles,
                                                series: BetaCoordinationLatentVocalicsDataSeries):
        previous_particles = states[time_step - 1]
        if series.unbounded_coordination is None:
            new_particles.unbounded_coordination = norm(previous_particles.unbounded_coordination,
                                                        np.sqrt(self.var_uc)).rvs()
        else:
            num_particles = len(previous_particles.unbounded_coordination)
            new_particles.unbounded_coordination = np.ones(num_particles) * series.unbounded_coordination[time_step]

        if series.coordination is None:
            new_particles.coordination = beta(sigmoid(new_particles.unbounded_coordination), self.var_cc).rvs()
        else:
            num_particles = len(previous_particles.coordination)
            new_particles.coordination = np.ones(num_particles) * series.coordination[time_step]

    def _create_new_particles(self) -> BetaCoordinationLatentVocalicsParticles:
        return BetaCoordinationLatentVocalicsParticles()

    def _summarize_particles(self, series: BetaCoordinationLatentVocalicsDataSeries,
                             particles: List[
                                 BetaCoordinationLatentVocalicsParticles]) -> BetaCoordinationLatentVocalicsParticlesSummary:

        summary = super()._summarize_particles(series, particles)
        summary = BetaCoordinationLatentVocalicsParticlesSummary.from_latent_vocalics_particles_summary(summary)

        for t, particles_in_time in enumerate(particles):
            summary.unbounded_coordination_mean[t] = particles_in_time.unbounded_coordination.mean()
            summary.unbounded_coordination_var[t] = particles_in_time.unbounded_coordination.var()

        return summary
