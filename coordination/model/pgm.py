from __future__ import annotations
from typing import Any, Dict, Generic, List, Optional, TypeVar

from multiprocessing import Pool

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from coordination.callback.callback import Callback
from coordination.common.log import BaseLogger
from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.common.utils import set_seed
from coordination.model.particle_filter import Particles, ParticleFilter


class Samples:

    @property
    def size(self):
        raise NotImplementedError


class ParticlesSummary:
    pass


SP = TypeVar('SP')
S = TypeVar('S')


class TrainingHyperParameters:
    """
    The set of hyper-parameters to be used during training. For instance, how to initialize variables.
    """

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class PGM(BaseEstimator, Generic[SP, S]):
    def __init__(self):
        super().__init__()

        self.train = False

        # List of hyper-parameters to be logged by the logger.
        self._hyper_params = {}
        self.nll_ = np.array([])

    def sample(self, num_samples: int, num_time_steps: int, seed: Optional[int], *args, **kwargs) -> SP:
        set_seed(seed)

        return None

    def fit(self, evidence: EvidenceDataset, train_hyper_parameters: TrainingHyperParameters, burn_in: int,
            seed: Optional[int], num_jobs: int = 1, logger: BaseLogger = BaseLogger(),
            callbacks: List[Callback] = None):

        if callbacks is None:
            callbacks = []

        self.train = True

        for callback in callbacks:
            callback.reset()

        hparams = self._hyper_params
        hparams.update(train_hyper_parameters.to_dict())
        hparams["burn_in"] = burn_in
        hparams["seed"] = seed
        hparams["num_jobs"] = num_jobs
        hparams["num_trials"] = evidence.num_trials
        hparams["num_time_steps"] = evidence.num_time_steps
        logger.add_hyper_params(hparams)
        set_seed(seed)

        # Gibbs Sampling

        # 1. Initialize latent variables
        self._initialize_gibbs(evidence, train_hyper_parameters, burn_in, seed, num_jobs)

        #    1.1 Compute initial NLL
        self.nll_ = np.zeros(burn_in + 1)
        self.nll_[0] = -self._compute_joint_loglikelihood_at(0, evidence, train_hyper_parameters)

        logger.add_scalar("train/nll", self.nll_[0], 0)

        with Pool(self._get_max_num_jobs()) as pool:
            for i in tqdm(range(1, burn_in + 1), desc="Gibbs Step", position=0):
                self._update_latent_variables(i, evidence, train_hyper_parameters, pool)
                self._update_parameters(i, evidence, train_hyper_parameters, pool)

                self.nll_[i] = -self._compute_joint_loglikelihood_at(i, evidence, train_hyper_parameters)
                logger.add_scalar("train/nll", self.nll_[i], i)
                self._log_parameters(i, logger)

                for callback in callbacks:
                    callback.check(self, i)

                if not self.train:
                    break

        return self

    def _initialize_gibbs(self, evidence: EvidenceDataset, train_hyper_parameters: TrainingHyperParameters,
                          burn_in: int, seed: int, num_jobs: int):
        raise NotImplementedError

    def _get_max_num_jobs(self) -> int:
        raise NotImplementedError

    def _update_latent_variables(self, gibbs_step: int, evidence: EvidenceDataset,
                                 train_hyper_parameters: TrainingHyperParameters, pool: Pool):
        raise NotImplementedError

    def _update_parameters(self, gibbs_step: int, evidence: EvidenceDataset,
                           train_hyper_parameters: TrainingHyperParameters, pool: Pool):
        """
        Update parameters with conjugate priors using the sufficient statistics of previously sampled latent variables.
        """
        raise NotImplementedError

    def _compute_joint_loglikelihood_at(self, gibbs_step: int, evidence: EvidenceDataset,
                                        train_hyper_parameters: TrainingHyperParameters) -> float:
        raise NotImplementedError

    def _log_parameters(self, gibbs_step: int, logger: BaseLogger):
        raise NotImplementedError

    def predict(self, evidence: EvidenceDataset, num_particles: int, seed: Optional[int], num_jobs: int = 1) -> List[S]:

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
                                       main_bar_position: int) -> List[ParticlesSummary]:
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

            results.append(self._summarize_particles(series, particle_filter.states))

        return results

    # Methods to be implemented by the subclass for inference
    def _sample_from_prior(self, num_particles: int, series: EvidenceDataSeries) -> Particles:
        raise NotImplementedError

    def _sample_from_transition_to(self, time_step: int, num_particles: int, new_particles: List[Particles],
                                   series: EvidenceDataSeries) -> Particles:
        raise NotImplementedError

    def _calculate_evidence_log_likelihood_at(self, time_step: int, states: List[Particles],
                                              series: EvidenceDataSeries):
        raise NotImplementedError

    def _resample_at(self, time_step: int, series: EvidenceDataSeries):
        return True

    def _summarize_particles(self, series: EvidenceDataSeries, particles: List[Particles]) -> ParticlesSummary:
        raise NotImplementedError
