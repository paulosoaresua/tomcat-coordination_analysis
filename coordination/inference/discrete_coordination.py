from typing import Callable

import numpy as np
from scipy.stats import norm

from coordination.component.speech.common import VocalicsSparseSeries

EPSILON = 1E-16


class DiscreteCoordinationInferenceFromVocalics:

    def __init__(self, vocalic_series: VocalicsSparseSeries, p_prior_coordination: float,
                 p_coordination_transition: float, mean_prior_vocalics: np.array, std_prior_vocalics: np.array,
                 std_uncoordinated_vocalics: np.ndarray, std_coordinated_vocalics: np.ndarray,
                 f: Callable = lambda x, s: x, fix_coordination_on_second_half: bool = True):
        """
        This class estimates discrete coordination with message passing.
        The series A and B can contain multiple series, one for each feature (e.g. pitch and intensity).

        @param vocalic_series: contains n x T vector with average vocalics for subject.
        @param p_prior_coordination: probability of coordination at the initial timestep.
        @param p_coordination_transition: probability that coordination changes from time t to time t+1.
        @param mean_prior_vocalics: mean of the distribution of vocalics at the first time it is observed.
        @param std_prior_vocalics: standard deviation of the distribution of vocalics at the first time it is observed.
        @param std_uncoordinated_vocalics: standard deviation of vocalics series when there's no coordination.
        @param std_coordinated_vocalics: standard deviation of the vocalic series when there's coordination.
        @param fix_coordination_on_second_half: whether coordination in the second half of the mission should be fixed.
        """

        assert len(mean_prior_vocalics) == vocalic_series.num_series
        assert len(std_prior_vocalics) == vocalic_series.num_series
        assert len(std_uncoordinated_vocalics) == vocalic_series.num_series
        assert len(std_coordinated_vocalics) == vocalic_series.num_series

        self._vocalic_series = vocalic_series
        self._p_prior_coordination = p_prior_coordination
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_uncoordinated_vocalics = std_uncoordinated_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

        self._num_features, self._time_steps = vocalic_series.values.shape  # n and T

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
        M = int(self._time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else self._time_steps

        m_forward = np.zeros((2, num_time_steps))
        for t in range(num_time_steps):
            # Transform to log scale for numerical stability

            # Contribution of the previous coordination sample to the marginal
            if t == 0:
                m_forward[:, t] = np.log(np.array(self._prior_vector, dtype=float) + EPSILON)
            else:
                m_forward[:, t] = np.log(np.matmul(m_forward[:, t - 1], self._transition_matrix) + EPSILON)

            # Contribution of the components to the coordination marginal
            if t == M and self._fix_coordination_on_second_half:
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
        M = int(self._time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else self._time_steps

        m_backwards = np.zeros((2, num_time_steps))
        for t in range(num_time_steps - 1, -1, -1):
            # Transform to log scale for numerical stability

            # Contribution of the next coordination sample to the marginal
            if t == num_time_steps - 1:
                # All the components contributions after t = M (or last element for non-fixed coordination in the
                # second half)
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
        prior = norm(loc=self._mean_prior_vocalics, scale=self._std_prior_vocalics)

        m_comp2coord = np.zeros((2, self._time_steps))
        for t in range(self._time_steps):
            if self._vocalic_series.mask[t] == 0:
                # We cannot tell anything about coordination if there's no observation
                m_comp2coord[:, t] = np.array([0.5, 0.5])
                continue

            # A represents the current vocalic value and the previous vocalic value from the same speaker.
            # B carries the most recent vocalic value from a different speaker than A.
            A_t = self._f(self._vocalic_series.values[:, t], 0)
            A_prev = None if self._vocalic_series.previous_from_self[t] is None else self._f(
                self._vocalic_series.values[:, self._vocalic_series.previous_from_self[t]], 0)
            B_prev = None if self._vocalic_series.previous_from_other[t] is None else self._f(
                self._vocalic_series.values[:, self._vocalic_series.previous_from_other[t]], 1)

            if B_prev is None:
                # Nothing can be inferred about coordination if there's no previous observation from another subject
                # to check for dependency
                m_comp2coord[:, t] = np.array([0.5, 0.5])
            else:
                # For C_t = 0
                if A_prev is None:
                    c0 = np.prod(prior.pdf(A_t))
                else:
                    transition_uncoordinated = norm(loc=A_prev, scale=self._std_uncoordinated_vocalics)
                    c0 = np.prod(transition_uncoordinated.pdf(A_t))

                # For C_t = 1
                transition_coordinated = norm(loc=B_prev, scale=self._std_coordinated_vocalics)
                c1 = np.prod(transition_coordinated.pdf(A_t))

                if c0 <= EPSILON and c1 <= EPSILON:
                    # For numerical stability
                    c0 = 0.5
                    c1 = 0.5

                m_comp2coord[:, t] = np.array([c0, c1])

        return m_comp2coord
