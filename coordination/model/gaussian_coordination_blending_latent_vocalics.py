from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy.stats import invgamma, norm
from tqdm import tqdm

from coordination.common.log import BaseLogger
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingLatentVocalics, \
    LatentVocalicsDataset, LatentVocalicsDataSeries, LatentVocalicsParticles, LatentVocalicsSamples, \
    LatentVocalicsParticlesSummary
from coordination.model.coordination_blending_latent_vocalics import default_f, default_g
from coordination.inference.mcmc import MCMC


class GaussianCoordinationBlendingLatentVocalics(
    CoordinationBlendingLatentVocalics[
        LatentVocalicsSamples, LatentVocalicsParticlesSummary]):

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
    def _generate_coordination_samples(self, num_samples: int, num_time_steps: int, samples: LatentVocalicsSamples):
        samples.coordination = np.zeros((num_samples, num_time_steps))
        scc = np.sqrt(self.var_cc)

        for t in tqdm(range(num_time_steps), desc="Coordination", position=0, leave=False):
            if t == 0:
                samples.coordination[:, 0] = self.initial_coordination
            else:
                # The mean of a truncated Gaussian distribution is given by mu + an offset. We remove the offset here,
                # such that the previous sample is indeed the mean of the truncated Gaussian.
                mean = samples.coordination[:, t - 1]
                samples.coordination[:, t] = norm(mean, scc).rvs()

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------
    def _initialize_coordination_for_gibbs(self, evidence: LatentVocalicsDataset):
        if evidence.coordination is None:
            means = np.zeros((evidence.num_trials, evidence.num_time_steps))
            self.coordination_samples_[0] = norm(means, 0.1).rvs()
            self.coordination_samples_[0, :, 0] = self.initial_coordination

            if self.coordination_samples_.shape[0] > 0:
                self.coordination_samples_[1] = self.coordination_samples_[0]
        else:
            self.coordination_samples_[0] = evidence.coordination

    def _compute_coordination_likelihood(self, gibbs_step: int, evidence: LatentVocalicsDataset) -> float:
        coordination = self.coordination_samples_[gibbs_step]
        scc = np.sqrt(self.vcc_samples_[gibbs_step])

        ll = 0
        for t in range(evidence.num_time_steps):
            # Coordination transition
            if t > 0:
                ll += norm(coordination[:, t - 1], scc).logpdf(coordination[:, t]).sum()

        return ll

    def _sample_coordination_on_fit(self, gibbs_step: int, evidence: LatentVocalicsDataset, time_steps: np.ndarray,
                                    job_num: int) -> Tuple[np.ndarray, ...]:

        if evidence.coordination is not None:
            return self.coordination_samples_[gibbs_step - 1].copy(), ()

        # The retain method copies the estimate in one gibbs step to the next one. Therefore, accessing the values in
        # the current gibbs step will give us the latest values of the estimates.
        coordination = self.coordination_samples_[gibbs_step].copy()
        latent_vocalics = self.latent_vocalics_samples_[gibbs_step]

        scc = np.sqrt(self.vcc_samples_[gibbs_step])
        saa = np.sqrt(self.vaa_samples_[gibbs_step])

        for t in tqdm(time_steps, desc="Sampling Coordination", position=job_num, leave=False):
            if t > 0:
                next_coordination = None if t == coordination.shape[1] - 1 else coordination[:, t + 1][:,
                                                                                np.newaxis]
                log_prob_fn_params = {
                    "previous_coordination_sample": coordination[:, t - 1][:, np.newaxis],
                    "next_coordination_sample": next_coordination,
                    "scc": scc,
                    "saa": saa,
                    "evidence": evidence,
                    "latent_vocalics": latent_vocalics,
                    "time_step": t
                }

                sampler = MCMC(proposal_fn=self._get_coordination_proposal,
                               proposal_fn_kwargs={},
                               log_prob_fn=self._get_coordination_posterior_unormalized_logprob,
                               log_prob_fn_kwargs=log_prob_fn_params)
                initial_sample = coordination[:, t][:, np.newaxis]
                inferred_coordination = sampler.generate_samples(initial_sample=initial_sample,
                                                                 num_samples=1,
                                                                 burn_in=50,
                                                                 retain_every=1)[0, :, 0]
                coordination[:, t] = inferred_coordination

        return coordination, ()

    @staticmethod
    def _get_coordination_proposal(previous_coordination_sample: np.ndarray):

        # Since coordination is constrained to 0 and 1, we don't expect a high variance.
        # We set variance to be smaller than 0.01 such that MCMC don't do big jumps and ends up
        # overestimating coordination.
        std = 0.005
        new_coordination_sample = norm(previous_coordination_sample, std).rvs()

        if previous_coordination_sample.shape[0] == 1:
            # The norm.rvs function does not preserve the dimensions of a unidimensional array.
            # We need to correct that if we are working with a single trial sample.
            new_coordination_sample = np.array([[new_coordination_sample]])

        # Hastings factor
        factor = 1

        return new_coordination_sample, factor

    def _get_coordination_posterior_unormalized_logprob(self,
                                                        proposed_coordination_sample: np.ndarray,
                                                        previous_coordination_sample: np.ndarray,
                                                        next_coordination_sample: Optional[np.ndarray],
                                                        scc: float,
                                                        saa: float,
                                                        evidence: LatentVocalicsDataset,
                                                        latent_vocalics: np.ndarray,
                                                        time_step: int):
        log_posterior = norm(previous_coordination_sample, scc).logpdf(
            proposed_coordination_sample)
        if next_coordination_sample is not None:
            log_posterior += norm(proposed_coordination_sample, scc).logpdf(
                next_coordination_sample)

        log_posterior = log_posterior.flatten()
        log_posterior += super()._get_latent_vocalics_term_for_coordination_posterior_unormalized_logprob(
            proposed_coordination_sample, saa, evidence, latent_vocalics, time_step)

        return log_posterior

    def _update_latent_parameters_coordination(self, gibbs_step: int, evidence: LatentVocalicsDataset,
                                               logger: BaseLogger):
        if self.var_cc is None:
            a = self.a_vcc + evidence.num_trials * (evidence.num_time_steps - 1) / 2
            x = self.coordination_samples_[gibbs_step, :, 1:]
            y = self.coordination_samples_[gibbs_step, :, :evidence.num_time_steps - 1]
            b = self.b_vcc + np.square(x - y).sum() / 2
            self.vcc_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()
            if self.vcc_samples_[gibbs_step] == np.nan:
                self.vcc_samples_[gibbs_step] = np.inf

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
    def _sample_coordination_from_prior(self, num_particles: int, new_particles: LatentVocalicsParticles,
                                        series: LatentVocalicsDataSeries):
        if series.coordination is None:
            new_particles.coordination = np.ones(num_particles) * self.initial_coordination
        else:
            new_particles.coordination = np.ones(num_particles) * series.coordination[0]

    def _sample_coordination_from_transition_to(self, time_step: int, states: List[LatentVocalicsParticles],
                                                new_particles: LatentVocalicsParticles,
                                                series: LatentVocalicsDataSeries):
        previous_particles = states[time_step - 1]
        if series.coordination is None:
            new_particles.coordination = norm(previous_particles.coordination, np.sqrt(self.var_cc)).rvs()
        else:
            num_particles = len(previous_particles.coordination)
            new_particles.coordination = np.ones(num_particles) * series.coordination[time_step]
