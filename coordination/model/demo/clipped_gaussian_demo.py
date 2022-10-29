from __future__ import annotations

import random
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, invgamma
from tqdm import tqdm

from coordination.common.log import BaseLogger, TensorBoardLogger
from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.inference.mcmc import MCMC
from coordination.model.particle_filter import Particles
from coordination.model.pgm import PGM, Samples


class ClippedGaussianDemoParticles(Particles):
    state: np.ndarray

    def _keep_particles_at(self, indices: np.ndarray):
        self.state = self.state[indices]


class ClippedGaussianDemoSamples(Samples):
    states: np.ndarray
    observations: np.ndarray

    @property
    def size(self):
        return self.observations.shape[0]


class ClippedGaussianDemoDataSeries(EvidenceDataSeries):

    def __init__(self, uuid: str, observations: np.ndarray, states: Optional[np.ndarray] = None):
        super().__init__(uuid)
        self.states = states
        self.observations = observations

    @property
    def num_time_steps(self):
        return self.observations.shape[1]

    @property
    def dim_observation(self):
        return self.observations.shape[0]


class ClippedGaussianDemoDataset(EvidenceDataset):

    def __init__(self, series: List[ClippedGaussianDemoDataSeries]):
        super().__init__(series)

        self.states = None if series[0].states is None else np.zeros((len(series), series[0].num_time_steps))
        self.observations = np.zeros((len(series), series[0].dim_observation, series[0].num_time_steps))

        for i, series in enumerate(series):
            if series.states is not None:
                self.states[i] = series.states
            self.observations[i] = series.observations


class ClippedGaussianDemo(PGM):
    def __init__(self, initial_state: float, dim_observations: int, a_vst: float, b_vst: float, a_vso: float,
                 b_vso: float):
        super().__init__()

        self.initial_state = initial_state
        self.dim_observations = dim_observations

        self.var_state_transition: Optional[float] = None
        self.var_state_observation: Optional[np.ndarray] = None

        # Parameters of the prior distributions (Inverse-Gamma) of the variances
        self.a_vst = a_vst
        self.b_vst = b_vst
        self.a_vso = a_vso
        self.b_vso = b_vso

        # Negative log-likelihood and samples collected during training
        self.vst_samples_ = np.array([])
        self.vso_samples_ = np.array([])
        self.state_samples_ = np.array([])

    # ---------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------------------------------------

    def sample(self, num_samples: int, num_time_steps: int, seed: Optional[int], *args,
               **kwargs) -> ClippedGaussianDemoSamples:
        """
        Regular ancestral sampling procedure.
        """
        super().sample(num_samples, num_time_steps, seed)

        samples = ClippedGaussianDemoSamples()
        samples.states = np.zeros((num_samples, num_time_steps))
        samples.observations = np.zeros((num_samples, self.dim_observations, num_time_steps))
        sst = np.sqrt(self.var_state_transition)
        sso = np.sqrt(self.var_state_observation)

        for t in range(num_time_steps):
            if t == 0:
                samples.states[:, 0] = self.initial_state
            else:
                transition_distribution = norm(loc=samples.states[:, t - 1], scale=sst)
                samples.states[:, t] = transition_distribution.rvs()

            means = np.clip(samples.states[:, t], a_min=0, a_max=1)[:, np.newaxis].repeat(self.dim_observations,
                                                                                          axis=1)
            emission_distribution = norm(loc=means, scale=sso)

            samples.observations[:, :, t] = emission_distribution.rvs()

        return samples

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------

    def _initialize_gibbs(self, burn_in: int, evidence: ClippedGaussianDemoDataset):
        super()._initialize_gibbs(burn_in, evidence)
        # History of samples in each Gibbs step
        self.state_samples_ = np.zeros((burn_in + 1, evidence.num_trials, evidence.num_time_steps))
        self.vst_samples_ = np.zeros(burn_in + 1)
        self.vso_samples_ = np.zeros(burn_in + 1)

        # 1. Latent variables and parameters initialization
        if evidence.states is None:
            self.state_samples_[0] = norm(loc=np.zeros((evidence.num_trials, evidence.num_time_steps)), scale=0.1).rvs()
            self.state_samples_[0, :, 0] = self.initial_state
        else:
            self.state_samples_[0] = evidence.states

        if self.var_state_transition is None:
            self.vst_samples_[0] = invgamma(a=self.a_vst, scale=self.b_vst).rvs()
        else:
            self.vst_samples_[0] = self.var_state_transition

        if self.var_state_observation is None:
            self.vso_samples_[0] = invgamma(a=self.a_vso, scale=self.b_vso).rvs()
        else:
            self.vso_samples_[0] = self.var_state_observation

    def _compute_joint_loglikelihood_at(self, gibbs_step: int, evidence: ClippedGaussianDemoDataset) -> float:
        sst = np.sqrt(self.vst_samples_[gibbs_step])
        sso = np.sqrt(self.vso_samples_[gibbs_step])

        ll = 0
        states = self.state_samples_[gibbs_step]
        for t in range(evidence.num_time_steps):
            if t > 0:
                ll += norm(states[:, t - 1], scale=sst).logpdf(states[:, t]).sum()

            ll += norm(np.clip(states[:, t], a_min=0, a_max=1)[:, np.newaxis], scale=sso).logpdf(
                evidence.observations[:, :, t]).sum()

        return ll

    def _gibbs_step(self, gibbs_step: int, evidence: ClippedGaussianDemoDataset, time_steps: np.ndarray,
                    job_num: int):
        states = self.state_samples_[gibbs_step - 1].copy()

        if evidence.states is None:
            sst = np.sqrt(self.vst_samples_[gibbs_step - 1])
            sso = np.sqrt(self.vso_samples_[gibbs_step - 1])

            state_proposal = lambda x: norm(loc=x, scale=0.1).rvs()
            for t in tqdm(time_steps, desc="Sampling over time", position=job_num, leave=False):
                def unormalized_log_posterior(sample: np.ndarray):
                    log_posterior = norm(loc=states[:, t - 1], scale=sst).logpdf(sample) + \
                                    norm(loc=np.clip(sample[:, np.newaxis], a_min=0, a_max=1), scale=sso).logpdf(
                                        evidence.observations[:, :, t]).sum(axis=1)
                    if t < evidence.num_time_steps - 1:
                        log_posterior += norm(loc=sample, scale=sst).logpdf(states[:, t + 1])

                    return log_posterior

                def acceptance_criterion(previous_sample: np.ndarray, new_sample: np.ndarray):
                    p1 = unormalized_log_posterior(previous_sample)
                    p2 = unormalized_log_posterior(new_sample)

                    return np.minimum(1, np.exp(p2 - p1))

                if t > 0:
                    state_sampler = MCMC(1, 100, 1, state_proposal, acceptance_criterion)
                    states[:, t] = state_sampler.generate_samples(states[:, t])[0]

        return states[:, time_steps]

    def _retain_samples_from_latent(self, gibbs_step: int, latents: Any, time_steps: np.ndarray):
        """
        The only latent variable return by _gibbs_step is a state variable
        """
        self.state_samples_[gibbs_step, :, time_steps] = latents.T

    def _update_latent_parameters(self, gibbs_step: int, evidence: ClippedGaussianDemoDataset, logger: BaseLogger):
        # Variance of the State Transition
        if self.var_state_transition is None:
            a = self.a_vst + evidence.num_trials * (evidence.num_time_steps - 1) / 2
            b = self.b_vst + np.square(
                self.state_samples_[gibbs_step, :, 1:] - self.state_samples_[gibbs_step, :,
                                                         :evidence.num_time_steps - 1]).sum() / 2
            self.vst_samples_[gibbs_step] = invgamma(a=a, scale=b).rvs()
        else:
            # Given
            self.vst_samples_[gibbs_step] = self.var_state_transition

        # Variance of the State Transition
        if self.var_state_observation is None:
            a = self.a_vso + evidence.num_trials * evidence.num_time_steps * self.dim_observations / 2
            b = self.b_vso + np.square(
                np.clip(self.state_samples_[gibbs_step], a_min=0, a_max=1)[:, np.newaxis,
                :] - evidence.observations).sum() / 2
            self.vso_samples_[gibbs_step] = invgamma(a=a, scale=b).rvs()
        else:
            # Given
            self.vso_samples_[gibbs_step] = self.var_state_observation

    def _retain_parameters(self):
        self.var_state_transition = self.vst_samples_[-1]
        self.var_state_observation = self.vso_samples_[-1]

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------

    def _summarize_particles(self, particles: List[ClippedGaussianDemoParticles]) -> np.ndarray:
        # Mean and variance over time
        summary = np.zeros((2, len(particles)))

        for t, particles_in_time in enumerate(particles):
            summary[0, t] = particles_in_time.state.mean()
            summary[1, t] = particles_in_time.state.var()

        return summary

    def _sample_from_prior(self, num_particles: int,
                           series: ClippedGaussianDemoDataSeries) -> ClippedGaussianDemoParticles:
        new_particles = ClippedGaussianDemoParticles()
        new_particles.state = np.ones(num_particles) * self.initial_state

        return new_particles

    def _sample_from_transition_to(self, time_step: int, states: List[ClippedGaussianDemoParticles],
                                   series: ClippedGaussianDemoDataSeries) -> ClippedGaussianDemoParticles:

        new_particles = ClippedGaussianDemoParticles()
        means = states[time_step - 1].state
        new_particles.state = norm(loc=means, scale=np.sqrt(self.var_state_transition)).rvs()

        return new_particles

    def _calculate_evidence_log_likelihood_at(self, time_step: int, states: List[ClippedGaussianDemoParticles],
                                              series: ClippedGaussianDemoDataSeries):
        means = np.clip(states[time_step - 1].state, a_min=0, a_max=1)[:, np.newaxis]
        lls = norm(loc=means, scale=np.sqrt(self.var_state_observation)).logpdf(series.observations[:, time_step]).sum(
            axis=1)

        return lls

    # def _resample_at(time_step: int, series: SeriesData):
    #     True


if __name__ == "__main__":
    TIME_STEPS = 20
    NUM_SAMPLES = 100
    DIM_OBSERVATION = 100
    model = ClippedGaussianDemo(0.5, DIM_OBSERVATION, 1e-1, 1e-1, 1e-1, 1e-1)
    model.var_state_transition = 0.1
    model.var_state_observation = 0.5

    np.random.seed(0)
    random.seed(0)
    samples = model.sample(NUM_SAMPLES, TIME_STEPS, seed=0)

    full_evidence = ClippedGaussianDemoDataset(
        [ClippedGaussianDemoDataSeries(f"{i}", samples.observations[i], samples.states[i]) for i in
         range(samples.size)])
    partial_evidence = ClippedGaussianDemoDataset(
        [ClippedGaussianDemoDataSeries(f"{i}", samples.observations[i]) for i in
         range(samples.size)])

    # Provide complete data to compute true NLL
    model.fit(full_evidence, 1, seed=0)
    true_nll = model.nll_[-1]

    model.var_state_transition = None
    model.var_state_observation = None

    tb_logger = TensorBoardLogger("/Users/paulosoares/code/tomcat-coordination/boards")
    model.fit(partial_evidence, 10, seed=0, num_jobs=4, logger=tb_logger)

    print(model.var_state_transition)
    print(model.var_state_observation)

    plt.figure()
    plt.plot(range(len(model.vst_samples_)), model.vst_samples_)
    plt.title("VST")
    plt.show()

    plt.figure()
    plt.plot(range(len(model.vso_samples_)), model.vso_samples_)
    plt.title("VSO")
    plt.show()

    plt.figure()
    plt.plot(range(len(model.nll_)), model.nll_)
    plt.plot(range(len(model.nll_)), np.ones(len(model.nll_)) * true_nll, linestyle="--")
    plt.title("NLL")
    plt.show()

    # Inference on the 5 first samples

    single_partial_evidence = partial_evidence.get_subset(list(range(5)))
    results = model.predict(partial_evidence, 10000, 0, 5)

    for d in range(5):
        result = results[d]
        plt.figure()
        ts = np.arange(TIME_STEPS)
        plt.plot(ts, result[0], color="tab:orange", marker="o")
        plt.fill_between(ts, result[0] - np.sqrt(result[1]), result[0] + np.sqrt(result[1]), color="tab:orange",
                         alpha=0.5)
        plt.plot(ts, samples.states[d], color="tab:blue", marker="o")
        plt.show()
