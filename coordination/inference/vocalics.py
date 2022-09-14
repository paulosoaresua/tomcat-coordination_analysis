from typing import Callable

import numpy as np
from scipy.stats import norm

from coordination.component.speech.vocalics_component import VocalicsSparseSeries
from coordination.inference.lds import apply_marginal_property, apply_conditional_property, pdf_projection

EPSILON = 1E-16


class DiscreteCoordinationInferenceFromVocalics:

    def __init__(self, series_a: VocalicsSparseSeries, series_b: VocalicsSparseSeries, p_prior_coordination: float,
                 p_coordination_transition: float, mean_prior_a: np.array, mean_prior_b: np.ndarray,
                 std_prior_a: np.array, std_prior_b: np.array, std_uncoordinated_a: np.ndarray,
                 std_uncoordinated_b: np.ndarray, std_coordinated_a: np.ndarray, std_coordinated_b: np.ndarray,
                 f: Callable = lambda x, s: x):
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
        self._f = f

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

        f_msg = get_message_from_individual_series_to_coordination

        m_comp2coord = np.zeros((2, self._time_steps))
        previous_values_a = self._series_a.get_previous_values()
        previous_values_b = self._series_b.get_previous_values()
        previous_values_same_source_a = self._series_a.get_previous_values_same_source()
        previous_values_same_source_b = self._series_b.get_previous_values_same_source()
        for t in range(self._time_steps):
            A_t = self._f(self._series_a.values[:, t], 0)
            B_t = self._f(self._series_b.values[:, t], 1)

            # Message from A_t to C_t
            previous_a = previous_values_same_source_a[t]
            previous_a = None if previous_a is None else self._f(previous_a, 0)
            previous_b = previous_values_b[t]
            previous_b = None if previous_b is None else self._f(previous_b, 0)
            m_comp2coord[:, t] = f_msg(A_t, previous_a, previous_b, self._mean_prior_a,
                                       self._std_prior_a, self._std_uncoordinated_a, self._std_coordinated_a,
                                       self._series_a.mask[t])

            # Message from B_t to C_t
            previous_a = previous_values_a[t]
            previous_a = None if previous_a is None else self._f(previous_a, 0)
            previous_b = previous_values_same_source_b[t]
            previous_b = None if previous_b is None else self._f(previous_b, 0)
            m_comp2coord[:, t] *= f_msg(B_t, previous_b, previous_a, self._mean_prior_b,
                                        self._std_prior_b, self._std_uncoordinated_b, self._std_coordinated_b,
                                        self._series_b.mask[t])

        return m_comp2coord


class ContinuousCoordinationInferenceFromVocalics:

    def __init__(self, series_a: VocalicsSparseSeries, series_b: VocalicsSparseSeries, mean_prior_coordination: float,
                 std_prior_coordination: float, std_coordination_drifting: float, mean_prior_a: np.array,
                 mean_prior_b: np.ndarray, std_prior_a: np.array, std_prior_b: np.array, std_coupling_a: np.ndarray,
                 std_coupling_b: np.ndarray, f: Callable = lambda x, s: x):

        assert len(mean_prior_a) == series_a.num_series
        assert len(mean_prior_b) == series_a.num_series
        assert len(std_prior_a) == series_a.num_series
        assert len(std_prior_b) == series_a.num_series
        assert len(std_coupling_a) == series_a.num_series
        assert len(std_coupling_b) == series_a.num_series

        self._series_a = series_a
        self._series_b = series_b
        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting
        self._mean_prior_a = mean_prior_a
        self._mean_prior_b = mean_prior_b
        self._std_prior_a = std_prior_a
        self._std_prior_b = std_prior_b
        self._std_coupling_a = std_coupling_a
        self._std_coupling_b = std_coupling_b
        self._f = f

        self._num_features, self._time_steps = series_a.values.shape  # n and T
        self._mid_time_step = int(self._time_steps / 2)  # M

    def estimate_means_and_variances(self) -> np.ndarray:
        filter_params = self._kalman_filter(False)
        return self._rts_smoother(filter_params, True)

    def filter(self) -> np.ndarray:
        return self._kalman_filter(True)

    def _kalman_filter(self, constrained: bool) -> np.ndarray:
        var_C = self._std_coordination_drifting ** 2
        var_BA = self._std_coupling_b ** 2
        var_AB = self._std_coupling_a ** 2

        params = np.zeros((2, self._mid_time_step + 1))
        params[:, 0] = np.array([self._mean_prior_coordination, self._std_prior_coordination ** 2])
        previous_values_a = self._series_a.get_previous_values()
        previous_values_b = self._series_b.get_previous_values()
        previous_values_same_source_a = self._series_a.get_previous_values_same_source()
        previous_values_same_source_b = self._series_b.get_previous_values_same_source()
        for t in range(1, self._mid_time_step):
            m_previous, v_previous = params[:, t - 1]

            """
            Coordination moves because of drifting:
            N(C_t | m_t, s_t) sim Integral [N(C_t | C_{t-1}, var_C) * N(C_{t-1}, m_{t-1}, v_{t-1})] dC_{t-1} 
                                = N(C_t | m_{t-1}, v_{t-1} + var_C)

            """
            mean, var = apply_marginal_property(1, 0, var_C, m_previous, v_previous)


            """
            Coordination moves because of drifting and vocalics:

            N(C_t | m_t, s_t) sim Integral [
                                    N(C_t | C_{t-1}, var_C) * N(C_{t-1}, m_{t-1}, v_{t-1}) *
                                    p(A_t | A_{t-1}, B_{t-1}, C_t, f) * p(A_t | A_{t-1}, B_{t-1}, C_t, f)
                                    ] dC_{t-1}

                                = p(A_t | A_{t-1}, B_{t-1}, C_t, f) * p(A_t | A_{t-1}, B_{t-1}, C_t, f) *
                                    Integral [N(C_t | C_{t-1}, var_C) * N(C_{t-1}, m_{t-1}, v_{t-1})] dC_{t-1} 

                                = N(A_t | D_{t-1} * C_t + A_{t-1}, var_AB) *
                                    N(B_t | -D_{t-1} * C_t + B_{t-1}, var_BA) *
                                    N(C_t | m_{t-1}, v_{t-1} + var_C) -- Obtained by the integral property previously

                                We can apply the conditional property recursively to obtain the mean and 
                                variances of the final distribution. We must apply the conditional property
                                to each dimension of A and B.
            """

            A_t = self._f(self._series_a.values[:, t], 0)
            B_t = self._f(self._series_b.values[:, t], 1)

            previous_a = previous_values_same_source_a[t]
            previous_a = None if previous_a is None else self._f(previous_a, 0)
            previous_b = previous_values_b[t]
            previous_b = None if previous_b is None else self._f(previous_b, 1)
            if previous_a is not None and previous_b is not None:
                D = previous_b - previous_a
                for i in range(self._num_features):
                    if self._series_a.mask[t] == 1:
                        mean, var = apply_conditional_property(A_t[i], D[i], previous_a[i], var_AB[i], mean, var)

            previous_a = previous_values_a[t]
            previous_a = None if previous_a is None else self._f(previous_a, 0)
            previous_b = previous_values_same_source_b[t]
            previous_b = None if previous_b is None else self._f(previous_b, 1)
            if previous_a is not None and previous_b is not None:
                D = previous_a - previous_b
                for i in range(self._num_features):
                    if self._series_b.mask[t] == 1:
                        mean, var = apply_conditional_property(B_t[i], D[i], previous_b[i], var_BA[i], mean, var)

            if constrained:
                mean, var = pdf_projection(mean, var, 0, 1)

            params[:, t] = [mean, var]

        # All the remaining vocalics contribute to the final coordination at t == M
        mean, var = params[:, self._mid_time_step - 1]
        for t in range(self._mid_time_step, self._time_steps):
            A_t = self._f(self._series_a.values[:, t], 0)
            B_t = self._f(self._series_b.values[:, t], 1)

            previous_a = previous_values_same_source_a[t]
            previous_a = None if previous_a is None else self._f(previous_a, 0)
            previous_b = previous_values_b[t]
            previous_b = None if previous_b is None else self._f(previous_b, 1)
            if previous_a is not None and previous_b is not None:
                D = previous_b - previous_a
                for i in range(self._num_features):
                    if self._series_a.mask[t] == 1:
                        mean, var = apply_conditional_property(A_t[i], D[i], previous_a[i], var_AB[i], mean, var)

            previous_a = previous_values_a[t]
            previous_a = None if previous_a is None else self._f(previous_a, 0)
            previous_b = previous_values_same_source_b[t]
            previous_b = None if previous_b is None else self._f(previous_b, 1)
            if previous_a is not None and previous_b is not None:
                D = previous_a - previous_b
                for i in range(self._num_features):
                    if self._series_b.mask[t] == 1:
                        mean, var = apply_conditional_property(B_t[i], D[i], previous_b[i], var_BA[i], mean, var)

        if constrained:
            mean, var = pdf_projection(mean, var, 0, 1)

        params[:, self._mid_time_step] = [mean, var]

        return params

    def _rts_smoother(self, filter_params: np.ndarray, constrained: bool) -> np.ndarray:
        var_C = self._std_coordination_drifting ** 2

        params = np.zeros((2, self._mid_time_step + 1))  # Coordination mean and variance over time
        params[:, self._mid_time_step] = filter_params[:, -1]

        for t in range(self._mid_time_step - 1, -1, -1):
            K = (filter_params[1, t] ** 2) / (filter_params[1, t] ** 2 + var_C)

            mean_filter, var_filter = filter_params[:, t]
            mean, var = apply_marginal_property(K, (1 - K) * mean_filter, (1 - K) * var_filter, *params[:, t + 1])

            if constrained:
                mean, var = pdf_projection(mean, var, 0, 1)
            params[:, t] = [mean, var]

        return params
