from typing import Any, Callable, Dict

import numpy as np


class MCMC:

    def __init__(self, proposal_fn: Callable, proposal_fn_kwargs: Dict[str, Any], log_prob_fn: Callable,
                 log_prob_fn_kwargs: Dict[str, Any]):
        self.proposal_fn = proposal_fn
        self.proposal_fn_kwargs = proposal_fn_kwargs
        self.log_prob_fn = log_prob_fn
        self.log_prob_fn_kwargs = log_prob_fn_kwargs

        self.acceptance_rates_ = np.array([])
        self.log_probs_ = np.array([])

    def generate_samples(self, initial_sample: np.ndarray, num_samples: int, burn_in: int, retain_every: int):
        assert np.ndim(initial_sample) == 2

        accepted_samples = 0

        num_iterations = burn_in + num_samples
        num_effective_samples = int(np.ceil(num_samples / retain_every))

        num_chains, num_dims = initial_sample.shape

        samples = np.zeros((num_iterations, num_chains, num_dims))
        effective_samples = np.zeros((num_effective_samples, num_chains, num_dims))
        self.acceptance_rates_ = np.zeros((num_iterations, num_chains))
        self.log_probs_ = np.zeros((num_iterations, num_chains))

        samples[0] = initial_sample
        self.log_probs_[0] = self.log_prob_fn(initial_sample, **self.log_prob_fn_kwargs)

        if burn_in <= 0:
            effective_samples[0] = initial_sample
            j = 1
        else:
            j = 0
        for i in range(1, num_iterations):
            new_sample, factor = self.proposal_fn(samples[i - 1], **self.proposal_fn_kwargs)
            u = np.random.rand(num_chains)

            # Compute Hastins ratio
            log_prob_new_sample = self.log_prob_fn(new_sample, **self.log_prob_fn_kwargs)
            A = np.exp(log_prob_new_sample - self.log_probs_[i - 1]) * factor
            samples[i] = np.where((u < A)[:, np.newaxis], new_sample, samples[i - 1])
            self.log_probs_[i] = np.where((u < A), log_prob_new_sample, self.log_probs_[i - 1])

            accepted_samples += (u < A)
            self.acceptance_rates_[i] = accepted_samples / (i * num_chains)

            if i >= burn_in and (i - burn_in - 1) % retain_every == 0:
                effective_samples[j] = samples[i]
                j += 1

        return effective_samples
