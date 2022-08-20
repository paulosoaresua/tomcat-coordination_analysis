import numpy as np
from scipy.stats import norm

from coordination.entity.sparse_series import SparseSeries

EPSILON = 1E-16


class DiscreteCoordinationInferenceFromVocalics:

    def __init__(self, series_a: SparseSeries, series_b: SparseSeries, p_prior_coordination: float,
                 p_coordination_transition: float, mean_prior_a: np.array, mean_prior_b: np.ndarray,
                 std_prior_a: np.array, std_prior_b: np.array, std_uncoordinated_a: np.ndarray,
                 std_uncoordinated_b: np.ndarray, std_coordinated_a: np.ndarray, std_coordinated_b: np.ndarray):
        """
        This class estimates discrete coordination with message passing.
        The series A and B can contain multiple series, one for each feature (e.g. pitch and intensity).

        @param series_a: n x T vector with average vocalics for subject A.
        @param series_b: n x T vector with average vocalics for subject B.
        @param p_prior_coordination: probability of coordination at the initial timestep.
        @param p_coordination_transition: probability that coordination changes from time t to time t+1.
        @param mean_prior_a: mean of the distribution of A at the first time A is observed.
        @param mean_prior_b: mean of the distribution of B at the first time B is observed.
        @param std_prior_a: standard deviation of the distribution of A at the first time A is observed.
        @param std_prior_b: standard deviation of the distribution of B at the first time B is observed.
        @param std_uncoordinated_a: standard deviation of series A when there's no coordination
        @param std_uncoordinated_b: standard deviation of series V when there's no coordination
        @param std_coordinated_a: standard deviation of series A when there's coordination
        @param std_coordinated_b: standard deviation of series V when there's coordination
        """

        assert len(mean_prior_a) == series_a.num_series
        assert len(mean_prior_b) == series_a.num_series
        assert len(std_prior_a) == series_a.num_series
        assert len(std_prior_b) == series_a.num_series
        assert len(std_uncoordinated_a) == series_a.num_series
        assert len(std_uncoordinated_b) == series_a.num_series
        assert len(std_coordinated_a) == series_a.num_series
        assert len(std_coordinated_b) == series_a.num_series

        self._series_a = series_a
        self._series_b = series_b
        self._p_prior_coordination = p_prior_coordination
        self._mean_prior_a = mean_prior_a
        self._mean_prior_b = mean_prior_b
        self._std_prior_a = std_prior_a
        self._std_prior_b = std_prior_b
        self._std_uncoordinated_a = std_uncoordinated_a
        self._std_uncoordinated_b = std_uncoordinated_b
        self._std_coordinated_a = std_coordinated_a
        self._std_coordinated_b = std_coordinated_b

        self._num_features, self._time_steps = series_a.values.shape  # n and T
        self._mid_time_step = int(self._time_steps / 2)  # M

        # C_{t-1} to C_t and vice-versa since the matrix is symmetric
        self._prior_vector = np.array([1 - p_prior_coordination, p_prior_coordination])
        self._transition_matrix = np.array([
            [1 - p_coordination_transition, p_coordination_transition],
            [p_coordination_transition, 1 - p_coordination_transition]])

    def estimate_marginals(self):
        m_comp2coord = self._get_messages_from_components_to_coordination()
        m_forward = self._forward(m_comp2coord)
        m_backwards = self._backwards(m_comp2coord)

        m_backwards = np.roll(m_backwards, shift=-1, axis=1)
        m_backwards[:, -1] = 1

        # alpha(C_t) * beta(C_{t+1}) x Transition Matrix
        c_marginals = m_forward * np.matmul(m_backwards.T, self._transition_matrix.T).T
        c_marginals /= np.sum(c_marginals, axis=0, keepdims=True)

        return c_marginals

    def _forward(self, m_comp2coord: np.ndarray) -> np.ndarray:
        m_forward = np.zeros((2, self._mid_time_step + 1))
        for t in range(self._mid_time_step + 1):
            # Transform to log scale for numerical stability

            # Contribution of the previous coordination sample to the marginal
            if t == 0:
                m_forward[:, t] = np.log(np.array(self._prior_vector, dtype=float) + EPSILON)
            else:
                m_forward[:, t] = np.log(np.matmul(m_forward[:, t - 1], self._transition_matrix) + EPSILON)

            # Contribution of the components to the coordination marginal
            if t == self._mid_time_step:
                # All the components contributions after t = M
                m_forward[:, t] += np.sum(np.log(m_comp2coord[:, t:]) + EPSILON, axis=1)
            else:
                m_forward[:, t] += np.log(m_comp2coord[:, t] + EPSILON)

            # Message normalization
            m_forward[:, t] -= np.max(m_forward[:, t])
            m_forward[:, t] = np.exp(m_forward[:, t])
            m_forward[:, t] /= np.sum(m_forward[:, t])

        return m_forward

    def _backwards(self, m_comp2coord: np.ndarray) -> np.ndarray:
        m_backwards = np.zeros((2, self._mid_time_step + 1))
        for t in range(self._mid_time_step, -1, -1):
            # Transform to log scale for numerical stability

            # Contribution of the next coordination sample to the marginal
            if t == self._mid_time_step:
                # All the components contributions after t = M
                m_backwards[:, t] = np.sum(np.log(m_comp2coord[:, t:] + EPSILON), axis=1)
            else:
                m_backwards[:, t] = np.log(np.matmul(m_backwards[:, t + 1], self._transition_matrix.T) + EPSILON)
                m_backwards[:, t] += np.log(m_comp2coord[:, t] + EPSILON)

            # Message normalization
            m_backwards[:, t] -= np.max(m_backwards[:, t])
            m_backwards[:, t] = np.exp(m_backwards[:, t])
            m_backwards[:, t] /= np.sum(m_backwards[:, t])

        return m_backwards

    def _get_messages_from_components_to_coordination(self) -> np.ndarray:
        def get_message_from_individual_series_to_coordination(current_self: np.ndarray,
                                                               previous_self: np.ndarray,
                                                               previous_other: np.ndarray,
                                                               prior_mean: np.ndarray,
                                                               prior_std: np.ndarray,
                                                               std_uncoordinated: np.ndarray,
                                                               std_coordinated: np.ndarray,
                                                               mask_self: int):
            # This term will be 0.5 only if there are no observations for the component at the current time step.
            # We use it so that c_leaves = [0.5, 0.5] instead of [0, 0] in these cases for numerical stability
            # with vector operations later when passing the messages around.
            addition_factor = (1 - mask_self) * 0.5
            prior = norm(loc=prior_mean, scale=prior_std)

            if previous_other is None:
                # Nothing can be inferred about coordination
                c0 = 0.5
                c1 = 0.5
            else:
                # For C_t = 0
                transition_uncoordinated = norm(loc=previous_self, scale=std_uncoordinated)
                c0 = addition_factor + mask_self * (
                    np.prod(prior.pdf(current_self)) if previous_self is None else np.prod(
                        transition_uncoordinated.pdf(current_self)))

                # For C_t = 1
                transition_coordinated = norm(loc=previous_other, scale=std_coordinated)
                c1 = addition_factor + mask_self * np.prod(transition_coordinated.pdf(current_self))

            if c0 <= EPSILON and c1 <= EPSILON:
                # For numerical stability
                c0 = 0.5
                c1 = 0.5

            return np.array([c0, c1])

        f = get_message_from_individual_series_to_coordination

        last_ta = None  # last time step with observed value for the series A
        last_tb = None  # last time step with observed value for the series B
        m_comp2coord = np.zeros((2, self._time_steps))
        for t in range(self._time_steps):
            previous_a = None if last_ta is None else self._series_a.values[:, last_ta]
            previous_b = None if last_tb is None else self._series_b.values[:, last_tb]

            # Message from A_t to C_t
            m_comp2coord[:, t] = f(self._series_a.values[:, t], previous_a, previous_b, self._mean_prior_a,
                                   self._std_prior_a, self._std_uncoordinated_a, self._std_coordinated_a,
                                   self._series_a.mask[t])

            # Message from B_t to C_t
            m_comp2coord[:, t] *= f(self._series_b.values[:, t], previous_b, previous_a, self._mean_prior_b,
                                    self._std_prior_b, self._std_uncoordinated_b, self._std_coordinated_b,
                                    self._series_b.mask[t])

            if self._series_a.mask[t] == 1:
                last_ta = t
            if self._series_b.mask[t] == 1:
                last_tb = t

        return m_comp2coord

#
# def estimate_discrete_coordination(series_a: np.ndarray, series_b: np.ndarray, prior_c: float,
#                                    transition_p: float, pa: Callable, pb: Callable, mask_a: Any,
#                                    mask_b: Any) -> np.ndarray:
#     """
#     Exact estimation of coordination marginals
#     """
#
#     T = series_a.shape[0]
#
#     if mask_a is None:
#         # Consider values of series A at all time steps
#         mask_a = np.ones(T)
#     if mask_b is None:
#         # Consider values of series B at all time steps
#         mask_b = np.ones(T)
#
#     c_forward = np.zeros((T, 2))
#     c_backwards = np.zeros((T, 2))
#     c_leaves = np.ones((T, 2)) * 0.5
#     transition_matrix = np.array([[1 - transition_p, transition_p], [transition_p, 1 - transition_p]])
#
#     # Forward messages
#     last_ta = None
#     last_tb = None
#     for t in range(T):
#         # Contribution of the previous coordination sample to the marginal
#         if t == 0:
#             c_forward[t] = np.array([1 - prior_c, prior_c], dtype=float)
#         else:
#             c_forward[t] = np.matmul(c_forward[t - 1], transition_matrix)
#
#         # Contribution of the vocalic component to the marginal
#         previous_a = None if last_ta is None else series_a[last_ta]
#         previous_b = None if last_tb is None else series_b[last_tb]
#
#         if previous_b is not None:
#             c_leaves[t] *= \
#                 np.ones(2) * (1 - mask_a[t]) + np.array([pa(series_a[t], previous_a, previous_b, 0),
#                                                          pa(series_a[t], previous_a, previous_b, 1)]) * mask_a[t]
#         if previous_a is not None:
#             c_leaves[t] *= \
#                 np.ones(2) * (1 - mask_b[t]) + np.array([pb(series_b[t], previous_b, previous_a, 0),
#                                                          pb(series_b[t], previous_b, previous_a, 1)]) * mask_b[t]
#         c_forward[t] *= c_leaves[t]
#         c_forward[t] /= np.sum(c_forward[t])
#
#         if mask_a[t] == 1:
#             last_ta = t
#         if mask_b[t] == 1:
#             last_tb = t
#
#     # Backward messages
#     for t in range(T - 1, -1, -1):
#         if t == T - 1:
#             c_backwards[t] = np.ones(2)
#         else:
#             # Because the transition probability is symmetric. We can multiply a value in the future by the transition
#             # matrix as is to estimate probabilities in the past.
#             c_backwards[t] = np.matmul(c_backwards[t + 1], transition_matrix)
#
#         # Contribution of the vocalic component to the marginal
#         c_backwards[t] *= c_leaves[t]
#         c_backwards[t] /= np.sum(c_backwards[t])
#
#     c_backwards = np.roll(c_backwards, shift=-1, axis=0)
#     c_backwards[-1] = 1
#     c_marginals = c_forward * np.matmul(c_backwards, transition_matrix)
#     c_marginals /= np.sum(c_marginals, axis=1, keepdims=True)
#
#     return c_marginals
#
#
# def estimate_continuous_coordination(gibbs_steps: int, series_a: np.ndarray, series_b: np.ndarray,
#                                      mean_a: float, mean_b: float, mask_a: Any, mask_b: Any,
#                                      mean_shift_coupling: float = 0) -> np.ndarray:
#     """
#     Approximate estimation of coordination marginals.
#
#     We will use Gibbs Sampling to infer coordination at each time step. Because of the transition entanglement between
#     coordination, we cannot generate the samples for all the time steps at the same time. However, we can split the
#     variables between even/odd time steps arrays and sample them in sequence at each gibbs step to speed up computation.
#
#     The posterior of the coordination variable can be written in a closed form. It is a truncated normal distribution
#     with mean and variance define as below.
#     """
#
#     T = series_a.shape[0]
#
#     if mask_a is None:
#         # Consider values of series A at all time steps
#         mask_a = np.ones(T)
#     if mask_b is None:
#         # Consider values of series B at all time steps
#         mask_b = np.ones(T)
#
#     even_indices = np.arange(0, T, 2)[1:]  # t = 0 has no previous neighbor
#     odd_indices = np.arange(1, T, 2)
#
#     c_samples = np.zeros((gibbs_steps, T + 1))
#
#     # Initialization
#     mask_ab = np.ones(T)
#     mask_ba = np.ones(T)
#     last_as = series_a
#     last_bs = series_b
#     observed_a = False
#     observed_b = False
#     for t in range(T):
#         if t > 0:
#             mean = c_samples[0, t - 1]
#             std = 0.1
#             c_samples[0, t] = truncnorm.rvs((0 - mean) / std, (1 - mean) / std, loc=mean, scale=std)
#             last_as[t] = mask_a[t] * last_as[t] + (1 - mask_a[t]) * last_as[t - 1]
#             last_bs[t] = mask_b[t] * last_bs[t] + (1 - mask_b[t]) * last_bs[t - 1]
#
#         if not observed_a or (observed_a and mask_b[t] == 0):
#             mask_ab[t] = 0
#         if not observed_b or (observed_b and mask_a[t] == 0):
#             mask_ba[t] = 0
#         if not observed_a:
#             observed_a = mask_a[t] == 1
#         if not observed_b:
#             observed_b = mask_b[t] == 1
#
#     # MCMC
#     for s in tqdm(range(gibbs_steps)):
#         # Sample even coordination
#         variances = (2 + np.sum(
#             mask_ba[even_indices][:, np.newaxis] * (mean_a - last_bs[even_indices - 1] - mean_shift_coupling) ** 2 +
#             mask_ab[even_indices][:, np.newaxis] * (mean_b - last_as[even_indices - 1] - mean_shift_coupling) ** 2,
#             axis=1))
#         variances[-1] -= 1  # The last time step only counts the previous coordination value
#         variances = 1 / variances
#         means = (np.sum(
#             mask_ba[even_indices][:, np.newaxis] * (mean_a - last_bs[even_indices - 1] - mean_shift_coupling) * (
#                     mean_a - series_a[even_indices]) +
#             mask_ab[even_indices][:, np.newaxis] * (mean_b - last_as[even_indices - 1] - mean_shift_coupling) * (
#                     mean_b - series_b[even_indices]), axis=1) +
#                  c_samples[s, even_indices - 1] + c_samples[s, even_indices + 1]) * variances
#
#         stds = np.sqrt(variances)
#         c_samples[s, even_indices] = truncnorm.rvs((0 - means) / stds, (1 - means) / stds, loc=means, scale=stds)
#
#         # Sample odd coordination
#         variances = (2 + np.sum(
#             mask_ba[odd_indices][:, np.newaxis] * (mean_a - last_bs[odd_indices - 1] - mean_shift_coupling) ** 2 +
#             mask_ab[odd_indices][:, np.newaxis] * (mean_b - last_as[odd_indices - 1] - mean_shift_coupling) ** 2,
#             axis=1))
#         variances[-1] -= 1  # The last time step only counts the previous coordination value
#         variances = 1 / variances
#         means = (np.sum(
#             mask_ba[odd_indices][:, np.newaxis] * (mean_a - last_bs[odd_indices - 1] - mean_shift_coupling) * (
#                     mean_a - series_a[odd_indices]) +
#             mask_ab[odd_indices][:, np.newaxis] * (mean_b - last_as[odd_indices - 1] - mean_shift_coupling) * (
#                     mean_b - series_b[odd_indices]), axis=1) +
#                  c_samples[s, odd_indices] + c_samples[s, odd_indices + 1]) * variances
#
#         stds = np.sqrt(variances)
#         c_samples[s, odd_indices] = truncnorm.rvs((0 - means) / stds, (1 - means) / stds, loc=means, scale=stds)
#
#         if s < gibbs_steps - 1:
#             c_samples[s + 1] = c_samples[s]
#
#     return c_samples[:, :-1]
