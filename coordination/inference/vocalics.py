from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from tqdm import tqdm

from scipy.stats import bernoulli
from scipy.stats import norm, truncnorm

EPSILON = 1E-16


class DiscreteCoordinationInference:

    def __init__(self, series_a: np.ndarray, series_b: np.ndarray, prior_c: float,
                 pc: float, mean_a: np.array, std_a: np.array, mean_b: np.ndarray, std_b: np.ndarray,
                 std_ab: np.ndarray, mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]):
        """
        This class estimates discrete coordination with message passing.
        The series A and B can contain  multiple features (n), e.g. picth and intensity.

        @param series_a: n x T vector with average vocalics for process A.
        @param series_b: n x T vector with average vocalics for process B.
        @param prior_c: prior distribution of coordination, i.e., distribution of coordination at time t = 0
        @param pc: probability that coordination does not change from time t to time t+1
        @param mean_a: mean of the prior of series A when there's no coordination
        @param std_a: standard deviation of series A when there's no coordination
        @param mean_b: mean of the prior of series B when there's no coordination
        @param std_b: standard deviation of series B when there's no coordination
        @param std_ab: standard deviation of series A and B when there's coordination
        @param mask_a: binary array indicating the time steps when there are values for series A
        @param mask_b: binary array indicating the time steps when there are values for series B
        @return:
        """

        self.series_a = series_a
        self.series_b = series_b
        self.prior_c = prior_c
        self.pc = pc
        self.mean_a = mean_a
        self.std_a = std_a
        self.mean_b = mean_b
        self.std_b = std_b
        self.std_ab = std_ab

        self.num_features, self.time_steps = series_a.shape  # n and T
        self.mid_time_step = int(self.time_steps / 2)  # M

        self.mask_a = np.ones(self.time_steps) if mask_a is None else mask_a
        self.mask_b = np.ones(self.time_steps) if mask_b is None else mask_b

        # C_{t-1} to C_t and vice-versa since the matrix is symmetric
        self.transition_matrix = np.array([[pc, 1 - pc], [1 - pc, pc]])

    def estimate_marginals(self):
        m_comp2coord = self.__get_messages_from_components_to_coordination()
        m_forward = self.__forward(m_comp2coord)
        m_backwards = self.__backwards(m_comp2coord)

        m_backwards = np.roll(m_backwards, shift=-1, axis=0)
        m_backwards[-1] = 1
        # alpha(C_t) * beta(C_{t+1}) x Transition Matrix
        c_marginals = m_forward * np.matmul(m_backwards, self.transition_matrix)
        c_marginals /= np.sum(c_marginals, axis=1, keepdims=True)

        return c_marginals

    def __forward(self, m_comp2coord: np.ndarray):
        m_forward = np.zeros((self.mid_time_step + 1, 2))
        for t in range(self.mid_time_step + 1):
            # Transform to log scale for numerical stability

            # Contribution of the previous coordination sample to the marginal
            if t == 0:
                m_forward[t] = np.log(np.array([1 - self.prior_c, self.prior_c], dtype=float) + EPSILON)
            else:
                m_forward[t] = np.log(np.matmul(m_forward[t - 1], self.transition_matrix) + EPSILON)

            # Contribution of the components to the coordination marginal
            if t == self.mid_time_step:
                # All the components contributions after t = M
                m_forward[t] += np.sum(np.log(m_comp2coord[t:]) + EPSILON, axis=0)
            else:
                m_forward[t] += np.log(m_comp2coord[t] + EPSILON)

            # Message normalization
            m_forward[t] -= np.max(m_forward[t])
            m_forward[t] = np.exp(m_forward[t])
            m_forward[t] /= np.sum(m_forward[t])

        return m_forward

    def __backwards(self, m_comp2coord: np.ndarray):
        m_backwards = np.zeros((self.mid_time_step + 1, 2))
        for t in range(self.mid_time_step, -1, -1):
            # Transform to log scale for numerical stability

            # Contribution of the next coordination sample to the marginal
            if t == self.mid_time_step:
                # All the components contributions after t = M
                m_backwards[t] = np.sum(np.log(m_comp2coord[t:] + EPSILON), axis=0)
            else:
                m_backwards[t] = np.log(np.matmul(m_backwards[t + 1], self.transition_matrix) + EPSILON)
                m_backwards[t] += np.log(m_comp2coord[t] + EPSILON)

            # Message normalization
            m_backwards[t] -= np.max(m_backwards[t])
            m_backwards[t] = np.exp(m_backwards[t])
            m_backwards[t] /= np.sum(m_backwards[t])

        return m_backwards

    def __get_messages_from_components_to_coordination(self):
        def get_message_from_individual_component_to_coordination(current_value_main_series: np.ndarray,
                                                                  previous_value_main_series: np.ndarray,
                                                                  previous_value_other_series: np.ndarray,
                                                                  prior_mean_main_series: np.ndarray,
                                                                  prior_std_main_series: np.ndarray,
                                                                  coupling_std: np.ndarray,
                                                                  mask_main_series: int):
            # This term will be 0.5 only if there are no observations for the component at the current time step.
            # We use it so that c_leaves = [0.5, 0.5] instead of [0, 0] in these cases for numerical stability
            # with vector operations later when passing the messages around.
            addition_factor = (1 - mask_main_series) * 0.5
            prior_cdf = np.prod(
                norm.pdf(current_value_main_series, loc=prior_mean_main_series, scale=prior_std_main_series))

            if previous_value_other_series is None:
                # Nothing can be inferred about coordination
                c0 = 0.5
                c1 = 0.5
            else:
                # For C_t = 0
                c0 = addition_factor + mask_main_series * (prior_cdf if previous_value_main_series is None else np.prod(
                    norm.pdf(
                        current_value_main_series, loc=previous_value_main_series, scale=prior_std_main_series)))

                # For C_t = 1
                c1 = addition_factor + mask_main_series * np.prod(
                    norm.pdf(current_value_main_series, loc=previous_value_other_series, scale=coupling_std))

            if c0 <= EPSILON and c1 <= EPSILON:
                c0 = 0.5
                c1 = 0.5

            return np.array([c0, c1])

        last_ta = None  # last time step with observed value for the series A
        last_tb = None  # last time step with observed value for the series B
        m_comp2coord = np.zeros((self.time_steps, 2))
        for t in range(self.time_steps):
            previous_a = None if last_ta is None else self.series_a[:, last_ta]
            previous_b = None if last_tb is None else self.series_b[:, last_tb]

            # Message from A_t to C_t
            m_comp2coord[t] = get_message_from_individual_component_to_coordination(self.series_a[:, t],
                                                                                    previous_a,
                                                                                    previous_b,
                                                                                    self.mean_a,
                                                                                    self.std_a,
                                                                                    self.std_ab,
                                                                                    self.mask_a[t])

            # Message from B_t to C_t
            m_comp2coord[t] *= get_message_from_individual_component_to_coordination(self.series_b[:, t],
                                                                                     previous_b,
                                                                                     previous_a,
                                                                                     self.mean_b,
                                                                                     self.std_b,
                                                                                     self.std_ab,
                                                                                     self.mask_b[t])

            if self.mask_a[t] == 1:
                last_ta = t
            if self.mask_b[t] == 1:
                last_tb = t

        return m_comp2coord


def estimate_discrete_coordination(series_a: np.ndarray, series_b: np.ndarray, prior_c: float,
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
            c_forward[t] = np.array([1 - prior_c, prior_c], dtype=float)
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
