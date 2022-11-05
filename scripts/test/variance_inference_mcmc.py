from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

from coordination.common.distribution import truncnorm
from coordination.common.utils import set_seed
from coordination.inference.mcmc import MCMC

set_seed(0)

NUM_SAMPLES = 100
v = np.ones(NUM_SAMPLES) * 0.09
m = np.ones(NUM_SAMPLES) * 0.7
data = truncnorm(m, np.sqrt(v)).rvs()


def proposal(previous_var_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = 0.005
    sample = lognorm(loc=0, s=s, scale=previous_var_sample).rvs()

    nominator = lognorm(loc=0, s=s, scale=sample).logpdf(previous_var_sample)
    denominator = lognorm(loc=0, s=s, scale=previous_var_sample).logpdf(sample)
    factor = np.exp(nominator - denominator).sum(axis=1)

    if isinstance(sample, float):
        sample = np.array([[sample]])

    return sample, factor


def log_prob(sample: np.ndarray, m: np.ndarray, data: np.ndarray):
    data = data[np.newaxis, :, np.newaxis].repeat(sample.shape[0], axis=0)
    sample = sample[:, np.newaxis, :].repeat(data.shape[1], axis=1)
    std = np.sqrt(np.log(sample))
    log_posterior = truncnorm(m, std).logpdf(data).sum(axis=2).sum(axis=1)

    return log_posterior


sampler = MCMC(proposal, {}, log_prob, {"m": m, "data": data})
initial_sample = np.exp(np.array([[0.1], [0.2], [0.3]]))
var_samples = np.log(sampler.generate_samples(initial_sample, 500, 0, 1))

plt.figure()
plt.plot(np.arange(var_samples.shape[0]), var_samples[:, 0, 0], label="C1")
plt.plot(np.arange(var_samples.shape[0]), var_samples[:, 1, 0], label="C2")
plt.plot(np.arange(var_samples.shape[0]), var_samples[:, 2, 0], label="C3")
print(f"Real/ Estimated: {v[0]} / {var_samples[-1, :, 0]}")
plt.title("Var")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(sampler.acceptance_rates_.shape[0]), sampler.acceptance_rates_[:, 0], label="C1")
plt.plot(np.arange(sampler.acceptance_rates_.shape[0]), sampler.acceptance_rates_[:, 1], label="C2")
plt.plot(np.arange(sampler.acceptance_rates_.shape[0]), sampler.acceptance_rates_[:, 2], label="C3")
plt.title("Acceptance Rate")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(sampler.log_probs_.shape[0]), -sampler.log_probs_[:, 0], label="C1")
plt.plot(np.arange(sampler.log_probs_.shape[0]), -sampler.log_probs_[:, 1], label="C2")
plt.plot(np.arange(sampler.log_probs_.shape[0]), -sampler.log_probs_[:, 2], label="C3")
plt.title("NLL")
plt.legend()
plt.show()
