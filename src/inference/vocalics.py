from scipy.stats import bernoulli
from typing import Any, Callable, Dict, List
import numpy as np
from tqdm import tqdm
from scipy.stats import norm, truncnorm

EPSILON = 1E-16


def estimate_discrete_coordination(series_a: np.ndarray, series_b: np.ndarray, prior_p: float,
                                   transition_p: float, pa: Callable, pb: Callable, mask_a: Any,
                                   mask_b: Any) -> np.ndarray:
    """
    Exact estimation of coordination marginals
    """

    T = series_a.shape[0]

    if mask_a is None:
        # Consider values of series A at all time steps
        mask_a = np.ones(T)
    if mask_b is None:
        # Consider values of series B at all time steps
        mask_b = np.ones(T)

    c_forward = np.zeros((T, 2))
    c_backwards = np.zeros((T, 2))
    c_leaves = np.ones((T, 2)) * 0.5
    transition_matrix = np.array([[1 - transition_p, transition_p], [transition_p, 1 - transition_p]])

    # Forward messages
    last_ta = None
    last_tb = None
    for t in range(T):
        # Contribution of the previous coordination sample to the marginal
        if t == 0:
            c_forward[t] = np.array([1 - prior_p, prior_p], dtype=float)
        else:
            c_forward[t] = np.matmul(c_forward[t - 1], transition_matrix)

        # Contribution of the vocalic component to the marginal
        previous_a = None if last_ta is None else series_a[last_ta]
        previous_b = None if last_tb is None else series_b[last_tb]

        if previous_b is not None:
            c_leaves[t] *= \
                np.ones(2) * (1 - mask_a[t]) + np.array([pa(series_a[t], previous_a, previous_b, 0),
                                                         pa(series_a[t], previous_a, previous_b, 1)]) * mask_a[t]
        if previous_a is not None:
            c_leaves[t] *= \
                np.ones(2) * (1 - mask_b[t]) + np.array([pb(series_b[t], previous_b, previous_a, 0),
                                                         pb(series_b[t], previous_b, previous_a, 1)]) * mask_b[t]
        c_forward[t] *= c_leaves[t]
        c_forward[t] /= np.sum(c_forward[t])

        if mask_a[t] == 1:
            last_ta = t
        if mask_b[t] == 1:
            last_tb = t

    # Backward messages
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            c_backwards[t] = np.ones(2)
        else:
            # Because the transition probability is symmetric. We can multiply a value in the future by the transition
            # matrix as is to estimate probabilities in the past.
            c_backwards[t] = np.matmul(c_backwards[t + 1], transition_matrix)

        # Contribution of the vocalic component to the marginal
        c_backwards[t] *= c_leaves[t]
        c_backwards[t] /= np.sum(c_backwards[t])

    c_backwards = np.roll(c_backwards, shift=-1, axis=0)
    c_backwards[-1] = 1
    c_marginals = c_forward * np.matmul(c_backwards, transition_matrix)
    c_marginals /= np.sum(c_marginals, axis=1, keepdims=True)

    return c_marginals


def estimate_continuous_coordination(gibbs_steps: int, series_a: np.ndarray, series_b: np.ndarray,
                                     mean_a: float, mean_b: float, mask_a: Any, mask_b: Any,
                                     mean_shift_coupling: float = 0) -> np.ndarray:
    """
    Approximate estimation of coordination marginals.

    We will use Gibbs Sampling to infer coordination at each time step. Because of the transition entanglement between
    coordination, we cannot generate the samples for all the time steps at the same time. However, we can split the
    variables between even/odd time steps arrays and sample them in sequence at each gibbs step to speed up computation.

    The posterior of the coordination variable can be written in a closed form. It is a truncated normal distribution
    with mean and variance define as below.
    """

    T = series_a.shape[0]

    if mask_a is None:
        # Consider values of series A at all time steps
        mask_a = np.ones(T)
    if mask_b is None:
        # Consider values of series B at all time steps
        mask_b = np.ones(T)

    even_indices = np.arange(0, T, 2)[1:]  # t = 0 has no previous neighbor
    odd_indices = np.arange(1, T, 2)

    c_samples = np.zeros((gibbs_steps, T + 1))

    # Initialization
    mask_ab = np.ones(T)
    mask_ba = np.ones(T)
    last_as = series_a
    last_bs = series_b
    observed_a = False
    observed_b = False
    for t in range(T):
        if t > 0:
            mean = c_samples[0, t - 1]
            std = 0.1
            c_samples[0, t] = truncnorm.rvs((0 - mean) / std, (1 - mean) / std, loc=mean, scale=std)
            last_as[t] = mask_a[t] * last_as[t] + (1 - mask_a[t]) * last_as[t - 1]
            last_bs[t] = mask_b[t] * last_bs[t] + (1 - mask_b[t]) * last_bs[t - 1]

        if not observed_a or (observed_a and mask_b[t] == 0):
            mask_ab[t] = 0
        if not observed_b or (observed_b and mask_a[t] == 0):
            mask_ba[t] = 0
        if not observed_a:
            observed_a = mask_a[t] == 1
        if not observed_b:
            observed_b = mask_b[t] == 1

    # MCMC
    for s in tqdm(range(gibbs_steps)):
        # Sample even coordination
        variances = (2 + np.sum(
            mask_ba[even_indices][:, np.newaxis] * (mean_a - last_bs[even_indices - 1] - mean_shift_coupling) ** 2 +
            mask_ab[even_indices][:, np.newaxis] * (mean_b - last_as[even_indices - 1] - mean_shift_coupling) ** 2,
            axis=1))
        variances[-1] -= 1  # The last time step only counts the previous coordination value
        variances = 1 / variances
        means = (np.sum(
            mask_ba[even_indices][:, np.newaxis] * (mean_a - last_bs[even_indices - 1] - mean_shift_coupling) * (
                    mean_a - series_a[even_indices]) +
            mask_ab[even_indices][:, np.newaxis] * (mean_b - last_as[even_indices - 1] - mean_shift_coupling) * (
                    mean_b - series_b[even_indices]), axis=1) +
                 c_samples[s, even_indices - 1] + c_samples[s, even_indices + 1]) * variances

        stds = np.sqrt(variances)
        c_samples[s, even_indices] = truncnorm.rvs((0 - means) / stds, (1 - means) / stds, loc=means, scale=stds)

        # Sample odd coordination
        variances = (2 + np.sum(
            mask_ba[odd_indices][:, np.newaxis] * (mean_a - last_bs[odd_indices - 1] - mean_shift_coupling) ** 2 +
            mask_ab[odd_indices][:, np.newaxis] * (mean_b - last_as[odd_indices - 1] - mean_shift_coupling) ** 2,
            axis=1))
        variances[-1] -= 1  # The last time step only counts the previous coordination value
        variances = 1 / variances
        means = (np.sum(
            mask_ba[odd_indices][:, np.newaxis] * (mean_a - last_bs[odd_indices - 1] - mean_shift_coupling) * (
                    mean_a - series_a[odd_indices]) +
            mask_ab[odd_indices][:, np.newaxis] * (mean_b - last_as[odd_indices - 1] - mean_shift_coupling) * (
                    mean_b - series_b[odd_indices]), axis=1) +
                 c_samples[s, odd_indices] + c_samples[s, odd_indices + 1]) * variances

        stds = np.sqrt(variances)
        c_samples[s, odd_indices] = truncnorm.rvs((0 - means) / stds, (1 - means) / stds, loc=means, scale=stds)

        if s < gibbs_steps - 1:
            c_samples[s + 1] = c_samples[s]

    return c_samples[:, :-1]
