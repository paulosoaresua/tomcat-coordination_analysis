from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

from datetime import datetime

import numpy as np
from scipy.stats import norm, invgamma
from tqdm import tqdm

from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries
from coordination.model.particle_filter import Particles
from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingLatentVocalics, \
    LatentVocalicsDataSeries, LatentVocalicsDataset, LatentVocalicsParticles, LatentVocalicsSamples


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
                 f: Callable = lambda x, s: x,
                 g: Callable = lambda x: x):
        super().__init__(initial_coordination, num_vocalic_features, num_speakers, a_vcc, b_vcc, a_va, b_va, a_vaa,
                         b_vaa, a_vo, b_vo, f, g)

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------

    def _generate_coordination_samples(self, num_samples: int, num_time_steps: int) -> np.ndarray:
        scc = np.sqrt(self.var_cc)
        samples = np.zeros((num_samples, num_time_steps))

        for t in range(num_time_steps):
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
        states = self.coordination_samples_[gibbs_step]
        for t in range(evidence.num_time_steps):
            if t > 0:
                ll += norm(states[:, t - 1], scale=scc).logpdf(states[:, t]).sum()

        return ll

    def _sample_coordination_on_fit(self, gibbs_step: int, evidence: LatentVocalicsDataset, time_steps: np.ndarray,
                                    job_num: int) -> np.ndarray:
        coordination = self.coordination_samples_[gibbs_step - 1].copy()

        if evidence.coordination is None:
            scc = np.sqrt(self.vcc_samples_[gibbs_step - 1])

            proposal = lambda x: norm(loc=x, scale=0.1).rvs()
            for t in tqdm(time_steps, desc="Sampling Coordination", position=job_num, leave=False):
                def unormalized_log_posterior(sample: np.ndarray):
                    log_posterior = norm(loc=coordination[:, t - 1], scale=scc).logpdf(sample)
                    if t < evidence.num_time_steps - 1:
                        log_posterior += norm(loc=sample, scale=scc).logpdf(coordination[:, t + 1])

                    # TODO - Proportion due to latent vocalics
                    # norm(loc=np.clip(sample[:, np.newaxis], a_min=0, a_max=1), scale=sso).logpdf(
                    #     evidence.observations[:, :, t]).sum(axis=1)

                    return log_posterior

                def acceptance_criterion(previous_sample: np.ndarray, new_sample: np.ndarray):
                    p1 = unormalized_log_posterior(previous_sample)
                    p2 = unormalized_log_posterior(new_sample)

                    return np.minimum(1, np.exp(p2 - p1))

                if t > 0:
                    state_sampler = MCMC(1, 100, 1, state_proposal, acceptance_criterion)
                    states[:, t] = state_sampler.generate_samples(states[:, t])[0]

        return states[:, time_steps]

    def _update_latent_parameters_coordination(self, gibbs_step: int, evidence: LatentVocalicsDataset):
        # Variance of the State Transition
        if self.var_cc is None:
            a = self.a_vcc + evidence.num_trials * (evidence.num_time_steps - 1) / 2
            x = self.coordination_samples_[gibbs_step, :, 1:]
            y = self.coordination_samples_[gibbs_step, :, :evidence.num_time_steps - 1]
            b = self.b_vcc + np.square(x - y).sum() / 2
            self.vcc_samples_[gibbs_step] = invgamma(a=a, scale=b).rvs()
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
