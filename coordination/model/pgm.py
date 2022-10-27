from __future__ import annotations

from typing import Any, List, Optional, Tuple

from multiprocessing import Pool

import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from coordination.common.dataset import InputFeaturesDataset, DataSeries
from coordination.model.particle_filter import Particles, ParticleFilter


class Samples:
    pass


class PGM(BaseEstimator):
    def __init__(self, time_steps: int, tb_writer: Optional[SummaryWriter] = None):
        super().__init__()

        self.tb_writer = tb_writer
        self.nll_ = np.array([])

    def sample(self, num_samples: int, time_steps: int) -> Tuple[Samples]:
        raise NotImplementedError

    def fit(self, evidence: InputFeaturesDataset, burn_in: int, num_jobs: int = 1, *args, **kwargs):
        # We split the PGM along the time axis. To make sure variables in one chunk is not dependent on variables in
        # another chunk, we create a separate chunk with the variables in the border that will be sampled in the
        # beginning of the Gibbs step.
        parallel_time_step_indices = []
        independent_time_step_indices = []

        if num_jobs == 1:
            # No parallel jobs
            independent_time_step_indices = np.arange(evidence.num_trials)
        else:
            time_chunks = np.array_split(np.arange(evidence.time_steps), num_jobs)
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
        self.tb_writer.add_scalar("train/nll", self.nll_[0])

        # 2. Sample the latent variables from their posterior distributions
        with Pool(num_jobs) as pool:
            for i in tqdm(range(1, burn_in + 1), desc="Gibbs Step", position=0):
                states = self._gibbs_step(i, evidence, independent_time_step_indices, 1)
                self._retain_samples_from_latent(i, states.T, independent_time_step_indices)

                if num_jobs > 1:
                    job_args = [(i, evidence, parallel_time_step_indices[j], j + 1) for j in range(num_jobs)]
                    for chunk_idx, result in enumerate(pool.starmap(self._gibbs_step, job_args)):
                        self._retain_samples_from_latent(result.T, independent_time_step_indices)

                self._update_latent_parameters(i, evidence)
                self.nll_[i] = -self._compute_loglikelihood_at(i, evidence)
                self.tb_writer.add_scalar("train/nll", self.nll_[i])

        self._retain_parameters()

        return self

    # Methods to be implemented by the subclass for parameter estimation
    def _initialize_gibbs(self, burn_in: int, evidence: InputFeaturesDataset):
        self.nll_ = np.zeros(burn_in + 1)

    def _compute_loglikelihood_at(self, gibbs_step: int, evidence: InputFeaturesDataset) -> float:
        raise NotImplementedError

    def _gibbs_step(self, gibbs_step: int, evidence: InputFeaturesDataset, time_steps: np.ndarray, job_num: int) -> Any:
        """
        Return the new samples from the latent variables
        """
        raise NotImplementedError

    def _retain_samples_from_latent(self, gibbs_step: int, evidence: InputFeaturesDataset,
                                    time_steps: np.ndarray) -> Any:
        """
        Update list of samples with new samples from the latent variables
        """
        raise NotImplementedError

    def _update_latent_parameters(self, gibbs_step: int, evidence: InputFeaturesDataset):
        """
        Update parameters with conjugate priors using the sufficient statistics of previously sampled latent variables.
        """
        raise NotImplementedError

    def _retain_parameters(self):
        """
        use the last parameter sample as the estimate of the model's parameters
        """
        raise NotImplementedError

    def predict(self, evidence: InputFeaturesDataset, num_particles: int, seed: int, *args, **kwargs) -> List[
        Particles]:
        particle_filter = ParticleFilter(
            num_particles=num_particles,
            resample_at_fn=self._resample_at,
            sample_from_prior_fn=self._sample_from_prior,
            sample_from_transition_fn=self._sample_from_transition_to,
            calculate_log_likelihood_fn=self._calculate_evidence_log_likelihood_at,
            seed=seed
        )

        results = []
        for d in tqdm(range(evidence.num_trials), desc="Trial", position=0):
            particle_filter.reset_state()
            series = evidence.series[d]

            for _ in tqdm(range(series.num_time_steps), desc="Time Step", position=1, leave=False):
                particle_filter.next(series)

            results.append(self._summarize_particles(particle_filter.states))

        return results

    # Methods to be implemented by the subclass for inference
    def _sample_from_prior(self, series: DataSeries) -> Particles:
        raise NotImplementedError

    def _sample_from_transition_to(self, time_step: int, new_particles: List[Particles],
                                   series: DataSeries) -> Particles:
        raise NotImplementedError

    def _calculate_evidence_log_likelihood_at(self, time_step: int, states: List[Particles], series: DataSeries):
        raise NotImplementedError

    def _resample_at(self, time_step: int, series: DataSeries):
        raise NotImplementedError

    def _sample_coordination_from_prior(self, new_particles: Particles):
        raise NotImplementedError

    def _sample_coordination_from_transition(self, previous_particles: Particles,
                                             new_particles: Particles) -> Particles:
        raise NotImplementedError

    def _create_new_particles(self) -> Particles:
        raise NotImplementedError

    def _summarize_particles(self, particles: List[Particles]) -> np.ndarray:
        raise NotImplementedError
