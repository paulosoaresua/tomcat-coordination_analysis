from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import copy
import inspect
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, truncnorm, invgamma
from sklearn.base import BaseEstimator
from tqdm import tqdm

from coordination.common.dataset import InputFeaturesDataset, SeriesData
from coordination.inference.mcmc import MCMC
from coordination.model.coordination_model import CoordinationModel
from coordination.model.particle_filter import Particles, ParticleFilter


class ClippedStateParticles(Particles):
    state: np.ndarray

    def _keep_particles_at(self, indices: np.ndarray):
        self.state = self.state[indices]


class ConstrainedLDSClippedGaussianSamples:
    states: np.ndarray
    observations: np.ndarray


class ConstrainedLDSClippedGaussianEvidence:
    def __init__(self, states: Optional[np.ndarray], observations: np.ndarray):
        self.states = states
        self.observations = observations

    @property
    def size(self):
        return self.observations.shape[0]

    @property
    def time_steps(self):
        return self.observations.shape[2]


# def state_posterior_sampling_fn(states, observations, sst, sso):
#     T = states.shape[1]
#     state_proposal = lambda x: np.clip(norm(loc=x, scale=0.1).rvs(), a_min=0, a_max=1)
#
#     for t in tqdm(range(T), desc="T", leave=False):
#         def unormalized_log_posterior(sample: np.ndarray):
#             log_posterior = norm(loc=states[:, t - 1], scale=sst).logpdf(sample) + \
#                             norm(loc=np.clip(sample[:, np.newaxis], a_min=0, a_max=1), scale=sso).logpdf(
#                                 observations[:, :, t]).sum(axis=1)
#             if t < T - 1:
#                 log_posterior += norm(loc=sample, scale=sst).logpdf(states[:, t + 1])
#
#             return log_posterior
#
#         def acceptance_criterion(previous_sample: np.ndarray, new_sample: np.ndarray):
#             p1 = unormalized_log_posterior(previous_sample)
#             p2 = unormalized_log_posterior(new_sample)
#
#             return np.minimum(1, np.exp(p2 - p1))
#
#         if t > 0:
#             state_sampler = MCMC(1, 100, 1, state_proposal, acceptance_criterion)
#             states[:, t] = state_sampler.generate_samples(states[:, t])[0]
#
#     return states


class ConstrainedLDSClippedGaussian(BaseEstimator):
    def __init__(self, initial_state: float, dim_observations: int, a_vst: float, b_vst: float, a_vso: float,
                 b_vso: float, num_jobs: int = 2):
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
        self.num_jobs = num_jobs

        # Negative log-likelihood and samples collected during the fit so we can plot at the end:
        self.nll_ = np.array([])
        self.vst_samples_ = np.array([])
        self.vso_samples_ = np.array([])
        self.state_samples_ = np.array([])

    def sample(self, time_steps: int, num_samples: int) -> Tuple[ConstrainedLDSClippedGaussianSamples, float]:
        """
        Regular ancestral sampling procedure.
        """

        samples = ConstrainedLDSClippedGaussianSamples()
        samples.states = np.zeros((num_samples, time_steps))
        samples.observations = np.zeros((num_samples, self.dim_observations, time_steps))
        sst = np.sqrt(self.var_state_transition)
        sso = np.sqrt(self.var_state_observation)
        ll = 0
        for t in range(time_steps):
            if t == 0:
                samples.states[:, 0] = self.initial_state
            else:
                transition_distribution = norm(loc=samples.states[:, t - 1], scale=sst)
                samples.states[:, t] = transition_distribution.rvs()
                ll += transition_distribution.logpdf(samples.states[:, t]).sum()

            emission_distribution = norm(
                loc=np.clip(samples.states[:, t], a_min=0, a_max=1)[:, np.newaxis].repeat(self.dim_observations,
                                                                                          axis=1), scale=sso)
            samples.observations[:, :, t] = emission_distribution.rvs()
            ll += emission_distribution.logpdf(samples.observations[:, :, t]).sum()

        return samples, -ll

    def _initialize_gibbs(self, burn_in: int, evidence: ConstrainedLDSClippedGaussianEvidence):
        # History of samples and negative log-likelihood in each Gibbs step
        self.state_samples_ = np.zeros((burn_in + 1, evidence.size, evidence.time_steps))
        self.vst_samples_ = np.zeros(burn_in + 1)
        self.vso_samples_ = np.zeros(burn_in + 1)
        self.nll_ = np.zeros(burn_in + 1)

        # 1. Latent variables and parameters initialization
        if evidence.states is None:
            self.state_samples_[0] = norm(loc=np.zeros((evidence.size, evidence.time_steps)), scale=0.1).rvs()
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

    def _compute_loglikelihood_at(self, gibbs_step: int, evidence: ConstrainedLDSClippedGaussianEvidence) -> float:
        # Compute initial log-likelihood
        sst = np.sqrt(self.vst_samples_[gibbs_step])
        sso = np.sqrt(self.vso_samples_[gibbs_step])

        ll = 0
        states = self.state_samples_[gibbs_step]
        for t in range(evidence.time_steps):
            if t > 0:
                ll += norm(states[:, t - 1], scale=sst).logpdf(states[:, t]).sum()

            ll += norm(np.clip(states[:, t], a_min=0, a_max=1)[:, np.newaxis], scale=sso).logpdf(
                evidence.observations[:, :, t]).sum()

        return ll

    def _gibbs_step(self, gibbs_step: int, evidence: ConstrainedLDSClippedGaussianEvidence, time_steps: np.ndarray,
                    index: int):
        states = self.state_samples_[gibbs_step - 1].copy()

        if evidence.states is None:
            sst = np.sqrt(self.vst_samples_[gibbs_step - 1])
            sso = np.sqrt(self.vso_samples_[gibbs_step - 1])

            state_proposal = lambda x: np.clip(norm(loc=x, scale=0.1).rvs(), a_min=0, a_max=1)
            for t in tqdm(time_steps, desc="Sampling over time", position=index, leave=False):
                def unormalized_log_posterior(sample: np.ndarray):
                    log_posterior = norm(loc=states[:, t - 1], scale=sst).logpdf(sample) + \
                                    norm(loc=np.clip(sample[:, np.newaxis], a_min=0, a_max=1), scale=sso).logpdf(
                                        evidence.observations[:, :, t]).sum(axis=1)
                    if t < evidence.time_steps - 1:
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

    def _update_latent_parameters(self, gibbs_step: int, evidence: ConstrainedLDSClippedGaussianEvidence):
        # Variance of the State Transition
        if self.var_state_transition is None:
            a = self.a_vst + evidence.size * (evidence.time_steps - 1) / 2
            b = self.b_vst + np.square(
                self.state_samples_[gibbs_step, :, 1:] - self.state_samples_[gibbs_step, :,
                                                         :evidence.time_steps - 1]).sum() / 2
            self.vst_samples_[gibbs_step] = invgamma(a=a, scale=b).rvs()
        else:
            # Given
            self.vst_samples_[gibbs_step] = self.var_state_transition

        # Variance of the State Transition
        if self.var_state_observation is None:
            a = self.a_vso + evidence.size * evidence.time_steps * self.dim_observations / 2
            b = self.b_vso + np.square(
                np.clip(self.state_samples_[gibbs_step], a_min=0, a_max=1)[:, np.newaxis,
                :] - evidence.observations).sum() / 2
            self.vso_samples_[gibbs_step] = invgamma(a=a, scale=b).rvs()
        else:
            # Given
            self.vso_samples_[gibbs_step] = self.var_state_observation

    def fit(self, evidence: ConstrainedLDSClippedGaussianEvidence, burn_in: int, *args, **kwargs):
        # n = observations.shape[0]
        # T = observations.shape[2]
        #
        # # To guarantee we have at least 2 time-steps per thread
        # assert T >= 2 * self.num_jobs

        # We split the PGM along the time axis. To make sure variables in one chunk is not dependent on variables in
        # another chunk, we create a separate chunk with the variables in the border that will be sampled in the
        # beginning of the Gibbs step.
        parallel_time_step_indices = []
        independent_time_step_indices = []

        if self.num_jobs == 1:
            # No parallel jobs
            independent_time_step_indices = np.arange(evidence.time_steps)
        else:
            time_chunks = np.array_split(np.arange(evidence.time_steps), self.num_jobs)
            for i, time_chunk in enumerate(time_chunks):
                if i == len(time_chunks) - 1:
                    # No need to add the last time index to the independent list since it does not depend on
                    # any variable from another chunk
                    parallel_time_step_indices.append(time_chunk)
                else:
                    independent_time_step_indices.append(time_chunk[-1])
                    parallel_time_step_indices.append(time_chunk[:-1])

        independent_time_step_indices = np.array(independent_time_step_indices)

        # Gibbs Sampling

        # 1. Initialize latent variables
        self._initialize_gibbs(burn_in, evidence)

        #    1.1 Compute initial NLL
        self.nll_[0] = -self._compute_loglikelihood_at(0, evidence)

        # 2. Sample the latent variables from their posterior distributions
        with Pool(self.num_jobs) as pool:
            for i in tqdm(range(1, burn_in + 1), desc="MCMC Step", position=0):
                states = self._gibbs_step(i, evidence, independent_time_step_indices, 1)
                self.state_samples_[i, :, independent_time_step_indices] = states.T
                if self.num_jobs > 1:
                    job_args = [(i, evidence, parallel_time_step_indices[j], j+1) for j in range(self.num_jobs)]
                    for chunk_idx, result in enumerate(pool.starmap(self._gibbs_step, job_args)):
                        self.state_samples_[i, :, parallel_time_step_indices[chunk_idx]] = result.T

                # self.state_samples_[i] = self._gibbs_step(i, evidence, np.arange(evidence.time_steps))

                self._update_latent_parameters(i, evidence)
                self.nll_[i] = -self._compute_loglikelihood_at(i, evidence)

        self.var_state_transition = self.vst_samples_[-1]
        self.var_state_observation = self.vso_samples_[-1]

        return self

    def predict(self, input_features: InputFeaturesDataset, *args, **kwargs) -> List[np.ndarray]:
        if input_features.num_trials > 0:
            assert len(self.mean_prior_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self.std_prior_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self.std_coordinated_latent_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self.std_observed_vocalics) == input_features.series[0].vocalics.num_features

        particle_filter = ParticleFilter(
            num_particles=self.num_particles,
            resample_at_fn=self._resample_at,
            sample_from_prior_fn=self._sample_from_prior,
            sample_from_transition_fn=self._sample_from_transition_to,
            calculate_log_likelihood_fn=self._calculate_log_likelihood_at,
            seed=self.seed
        )

        pbar_outer = None
        if self.show_progress_bar:
            pbar_outer = tqdm(total=input_features.num_trials, desc="Trial", position=0)

        result = []
        for d in range(input_features.num_trials):
            particle_filter.reset_state()
            series = input_features.series[d]

            M = int(series.num_time_steps / 2)
            num_time_steps = M + 1 if self.fix_coordination_on_second_half else series.num_time_steps

            pbar_inner = None
            if self.show_progress_bar:
                pbar_inner = tqdm(total=input_features.series[0].num_time_steps, desc="Time Step", position=1,
                                  leave=False)

            params = np.zeros((2, num_time_steps))
            for t in range(0, series.num_time_steps):
                particle_filter.next(series)

                # We keep generating latent vocalics after M but not coordination. The fixed coordination is given by
                # the set of particles after the last latent vocalics was generated
                real_time = min(t, M) if self.fix_coordination_on_second_half else t
                mean = particle_filter.states[-1].mean()
                variance = particle_filter.states[-1].var()
                params[:, real_time] = [mean, variance]

                if self.show_progress_bar:
                    pbar_inner.update()

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(f"inference/coordination-{series.uuid}", mean, t)

            result.append(params)

            if self.show_progress_bar:
                pbar_outer.update()

            if self.tb_writer is not None:
                self.log_coordination_inference_plot(series, params, series.uuid)

        return result

    def _sample_from_prior(self, series: SeriesData) -> Particles:
        new_particles = self._create_new_particles()
        self._sample_coordination_from_prior(new_particles)
        self._sample_vocalics_from_prior(series, new_particles)

        return new_particles

    def _sample_vocalics_from_prior(self, series: SeriesData, new_particles: LatentVocalicsParticles):
        new_particles.latent_vocalics = {subject: None for subject in series.vocalics.subjects}
        if series.vocalics.mask[0] == 1:
            speaker = series.vocalics.utterances[0].subject_id
            mean = np.ones((self.num_particles, series.vocalics.num_series)) * self.mean_prior_latent_vocalics
            new_particles.latent_vocalics[speaker] = norm(loc=mean, scale=self.std_prior_latent_vocalics).rvs()

    def _sample_from_transition_to(self, time_step: int, states: List[LatentVocalicsParticles],
                                   series: SeriesData) -> LatentVocalicsParticles:
        M = int(series.num_time_steps / 2)
        if not self.fix_coordination_on_second_half or time_step <= M:
            new_particles = self._create_new_particles()
            self._sample_coordination_from_transition(states[time_step - 1], new_particles)
        else:
            # Coordination if fixed
            new_particles = copy.deepcopy(states[time_step - 1])

        self._sample_vocalics_from_transition_to(time_step, states[time_step - 1], new_particles, series)

        return new_particles

    def _sample_vocalics_from_transition_to(self, time_step: int, previous_particles: LatentVocalicsParticles,
                                            new_particles: LatentVocalicsParticles, series: SeriesData):
        new_particles.latent_vocalics = previous_particles.latent_vocalics.copy()
        if series.vocalics.mask[time_step] == 1:
            speaker = series.vocalics.utterances[time_step].subject_id
            A_prev = previous_particles.latent_vocalics[speaker]

            A_prev = self.f(A_prev, 0) if A_prev is not None else np.ones(
                (self.num_particles, series.vocalics.num_series)) * self.mean_prior_latent_vocalics
            if series.vocalics.previous_from_other[time_step] is None:
                new_particles.latent_vocalics[speaker] = norm(loc=A_prev, scale=self.std_prior_latent_vocalics).rvs()
            else:
                other_speaker = series.vocalics.utterances[
                    series.vocalics.previous_from_other[time_step]].subject_id
                B_prev = self.f(previous_particles.latent_vocalics[other_speaker], 1)
                D = B_prev - A_prev
                mean = D * new_particles.coordination[:, np.newaxis] + A_prev
                new_particles.latent_vocalics[speaker] = norm(loc=mean,
                                                              scale=self.std_coordinated_latent_vocalics).rvs()

    def _calculate_log_likelihood_at(self, time_step: int, states: List[LatentVocalicsParticles], series: SeriesData):
        if series.vocalics.mask[time_step] == 1:
            speaker = series.vocalics.utterances[time_step].subject_id
            A_t = states[time_step].latent_vocalics[speaker]
            O_t = series.vocalics.values[:, time_step]
            log_likelihoods = norm(loc=self.g(A_t), scale=self.std_observed_vocalics).logpdf(O_t).sum(axis=1)
        else:
            log_likelihoods = 0

        return log_likelihoods

    @staticmethod
    def _resample_at(time_step: int, series: SeriesData):
        return series.vocalics.mask[time_step] == 1 and series.vocalics.previous_from_other[
            time_step] is not None

    def _sample_coordination_from_prior(self, new_particles: LatentVocalicsParticles):
        raise NotImplementedError

    def _sample_coordination_from_transition(self, previous_particles: LatentVocalicsParticles,
                                             new_particles: LatentVocalicsParticles) -> Particles:
        raise NotImplementedError

    def _create_new_particles(self) -> LatentVocalicsParticles:
        return LatentVocalicsParticles()


if __name__ == "__main__":
    model = ConstrainedLDSClippedGaussian(0, 10, 1e-1, 1e-1, 1e-1, 1e-1, num_jobs=10)
    model.var_state_transition = 0.1
    model.var_state_observation = 0.5
    samples, true_nll = model.sample(1020, 100)

    model.var_state_transition = None
    model.var_state_observation = None

    # model.state_samples_ = samples.states[np.newaxis, :].repeat(101, axis=0)

    evidence = ConstrainedLDSClippedGaussianEvidence(None, samples.observations)
    np.random.seed(0)
    random.seed(0)
    model.fit(evidence, 5)

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
