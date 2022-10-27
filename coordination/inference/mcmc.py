from typing import Any, Callable

import numpy as np


class MCMC:

    def __init__(self, num_samples: int, burn_in: int, retain_every: int, proposal: Callable,
                 acceptance_criterion: Callable):
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.retain_every = retain_every
        self.proposal = proposal
        self.acceptance_criterion = acceptance_criterion

    def generate_samples(self, initial_sample: Any):
        num_effective_samples = self.burn_in + self.num_samples * self.retain_every
        effective_samples = []
        samples = [initial_sample]
        for i in range(1, num_effective_samples):
            new_sample = self.proposal(samples[i - 1])
            u = np.random.rand(len(new_sample))
            A = self.acceptance_criterion(samples[-1], new_sample)
            samples_to_keep = np.copy(samples[i - 1])
            samples_to_keep[u < A] = new_sample[u < A]
            samples.append(samples_to_keep)

            if i >= self.burn_in and (i - self.burn_in + 1) % self.retain_every == 0:
                effective_samples.append(samples[i])

        return np.array(effective_samples)
