from __future__ import annotations
from typing import Any, List, Optional, Tuple

from multiprocessing import Pool

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from coordination.common.log import BaseLogger
from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.common.utils import set_seed
from coordination.model.particle_filter import Particles, ParticleFilter


class Samples:

    @property
    def size(self):
        raise NotImplementedError


class PGM(BaseEstimator):
    def __init__(self):
        super().__init__()

        self.nll_ = np.array([])

    def sample(self, num_samples: int, time_steps: int, seed: Optional[int]) -> Tuple[Samples]:
        set_seed(seed)

        # To be implemented by a child class

    def fit(self, evidence: EvidenceDataset, burn_in: int, seed: Optional[int], num_jobs: int = 1,
            logger: BaseLogger = BaseLogger()):

        set_seed(seed)

        # We split the PGM along the time axis. To make sure variables in one chunk is not dependent on variables in
        # another chunk, we create a separate chunk with the variables in the border that will be sampled in the
        # beginning of the Gibbs step.
        parallel_time_step_indices = []
        independent_time_step_indices = []

        num_effective_jobs = min(evidence.time_steps / 2, num_jobs)
        if num_effective_jobs == 1:
            # No parallel jobs
            independent_time_step_indices = np.arange(evidence.time_steps)
        else:
            time_chunks = np.array_split(np.arange(evidence.time_steps), num_effective_jobs)
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
        self.nll_[0] = -self._compute_joint_loglikelihood_at(0, evidence)

        logger.add_scalar("train/nll", self.nll_[0], 0)

        # 2. Sample the latent variables from their posterior distributions
        with Pool(num_effective_jobs) as pool:
            for i in tqdm(range(1, burn_in + 1), desc="Gibbs Step", position=0):
                latents = self._gibbs_step(i, evidence, independent_time_step_indices, 1)
                self._retain_samples_from_latent(i, latents, independent_time_step_indices)

                if num_effective_jobs > 1:
                    job_args = [(i, evidence, parallel_time_step_indices[j], j + 1) for j in range(num_effective_jobs)]
                    for chunk_idx, result in enumerate(pool.starmap(self._gibbs_step, job_args)):
                        self._retain_samples_from_latent(i, result, parallel_time_step_indices[chunk_idx])

                self._update_latent_parameters(i, evidence)
                self.nll_[i] = -self._compute_joint_loglikelihood_at(i, evidence)

                logger.add_scalar("train/nll", self.nll_[i], i)

        self._retain_parameters()

        return self

    # Methods to be implemented by the subclass for parameter estimation
    def _initialize_gibbs(self, burn_in: int, evidence: EvidenceDataset):
        self.nll_ = np.zeros(burn_in + 1)

    def _compute_joint_loglikelihood_at(self, gibbs_step: int, evidence: EvidenceDataset) -> float:
        raise NotImplementedError

    def _gibbs_step(self, gibbs_step: int, evidence: EvidenceDataset, time_steps: np.ndarray, job_num: int) -> Any:
        """
        Return the new samples from the latent variables
        """
        raise NotImplementedError

    def _retain_samples_from_latent(self, gibbs_step: int, latents: Any, time_steps: np.ndarray):
        """
        Update list of samples with new samples from the latent variables
        """
        raise NotImplementedError

    def _update_latent_parameters(self, gibbs_step: int, evidence: EvidenceDataset):
        """
        Update parameters with conjugate priors using the sufficient statistics of previously sampled latent variables.
        """
        raise NotImplementedError

    def _retain_parameters(self):
        """
        use the last parameter sample as the estimate of the model's parameters
        """
        raise NotImplementedError

    def predict(self, evidence: EvidenceDataset, num_particles: int, seed: Optional[int], num_jobs: int = 1) -> List[
        np.ndarray]:

        num_effective_jobs = min(num_jobs, evidence.num_trials)
        trial_chunks = np.array_split(np.arange(evidence.num_trials), num_effective_jobs)
        results = []
        if num_effective_jobs > 1:
            with Pool(num_effective_jobs) as pool:
                job_args = [(evidence.get_subset(trial_chunks[j]), num_particles, seed, 2 * j) for j in
                            range(num_effective_jobs)]
                for result in pool.starmap(self._run_particle_filter_inference, job_args):
                    results.extend(result)
        else:
            results = self._run_particle_filter_inference(evidence, num_particles, seed, 0)

        return results

    def _run_particle_filter_inference(self, evidence: EvidenceDataset, num_particles: int, seed: Optional[int],
                                       main_bar_position: int) -> List[np.ndarray]:
        particle_filter = ParticleFilter(
            num_particles=num_particles,
            resample_at_fn=self._resample_at,
            sample_from_prior_fn=self._sample_from_prior,
            sample_from_transition_fn=self._sample_from_transition_to,
            calculate_log_likelihood_fn=self._calculate_evidence_log_likelihood_at,
            seed=seed
        )

        results = []
        for d in tqdm(range(evidence.num_trials), "Trial", position=main_bar_position):
            particle_filter.reset_state()
            series = evidence.series[d]

            for _ in tqdm(range(series.num_time_steps), desc="Time Step", position=main_bar_position + 1,
                          leave=False):
                particle_filter.next(series)

            results.append(self._summarize_particles(particle_filter.states))

        return results

    # Methods to be implemented by the subclass for inference
    def _sample_from_prior(self, num_particles: int, series: EvidenceDataSeries) -> Particles:
        raise NotImplementedError

    def _sample_from_transition_to(self, time_step: int, new_particles: List[Particles],
                                   series: EvidenceDataSeries) -> Particles:
        raise NotImplementedError

    def _calculate_evidence_log_likelihood_at(self, time_step: int, states: List[Particles],
                                              series: EvidenceDataSeries):
        raise NotImplementedError

    def _resample_at(self, time_step: int, series: EvidenceDataSeries):
        return True

    def _summarize_particles(self, particles: List[Particles]) -> np.ndarray:
        raise NotImplementedError