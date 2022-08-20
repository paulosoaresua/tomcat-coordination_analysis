import unittest
from unittest.mock import MagicMock

import numpy as np
from scipy.stats import norm

from coordination.inference.vocalics import DiscreteCoordinationInferenceFromVocalics
from coordination.entity.sparse_series import SparseSeries


class DiscreteCoordinationParameters:
    # [WARN] Changing these values will break the tests
    values_a = np.array([[0.2, 0.4, 0, 0.5, 0, 0.6],
                         [0.3, 0.2, 0, 0.1, 0, 0.7]])
    mask_a = np.array([1, 1, 0, 1, 0, 1])

    values_b = np.array([[0, 0.1, 0.3, 0, 0, 0.8],
                         [0, 0.3, 0.4, 0, 0, 0.9]])
    mask_b = np.array([0, 1, 1, 0, 0, 1])

    series_a = SparseSeries(values_a, mask_a)
    series_b = SparseSeries(values_b, mask_b)

    p_prior_coordination = 0
    p_coordination_transition = 0.1
    mean_prior = np.array([0, 0])
    std_prior = np.array([1, 1])
    std_ab = np.array([1, 1])


class TestDiscreteCoordinationInference(unittest.TestCase):
    def setUp(self):
        self.params = DiscreteCoordinationParameters

    def test_message_from_components_to_coordination(self):
        inference_engine = DiscreteCoordinationInferenceFromVocalics(self.params.series_a,
                                                                     self.params.series_b,
                                                                     self.params.p_prior_coordination,
                                                                     self.params.p_coordination_transition,
                                                                     self.params.mean_prior,
                                                                     self.params.mean_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab)

        T = self.params.series_a.values.shape[1]

        m_a2coord_expected = np.zeros((T, 2))
        m_b2coord_expected = np.zeros((T, 2))

        # t = 0
        m_a2coord_expected[0, 0] = 0.5  # No previous value of B at this time step
        m_a2coord_expected[0, 1] = 0.5

        m_b2coord_expected[0, 0] = 0.5  # No observation for B at this time step
        m_b2coord_expected[0, 1] = 0.5

        # t = 1
        m_a2coord_expected[1, 0] = 0.5  # No previous value of B at this time step
        m_a2coord_expected[1, 1] = 0.5

        m_b2coord_expected[1, 0] = np.prod(
            norm.pdf([0.1, 0.3], loc=self.params.mean_prior, scale=self.params.std_prior))
        m_b2coord_expected[1, 1] = np.prod(norm.pdf([0.1, 0.3], loc=[0.2, 0.3], scale=self.params.std_ab))

        # t = 2
        m_a2coord_expected[2, 0] = 0.5  # No observation for A at this time step
        m_a2coord_expected[2, 1] = 0.5

        m_b2coord_expected[2, 0] = np.prod(norm.pdf([0.3, 0.4], loc=[0.1, 0.3], scale=self.params.std_prior))
        m_b2coord_expected[2, 1] = np.prod(norm.pdf([0.3, 0.4], loc=[0.4, 0.2], scale=self.params.std_ab))

        # t = 3 = M
        m_a2coord_expected[3, 0] = np.prod(norm.pdf([0.5, 0.1], loc=[0.4, 0.2], scale=self.params.std_prior))
        m_a2coord_expected[3, 1] = np.prod(norm.pdf([0.5, 0.1], loc=[0.3, 0.4], scale=self.params.std_ab))

        m_b2coord_expected[3, 0] = 0.5  # No observation for B at this time step
        m_b2coord_expected[3, 1] = 0.5

        # t = 4
        m_a2coord_expected[4, 0] = 0.5  # No observation for A at this time step
        m_a2coord_expected[4, 1] = 0.5

        m_b2coord_expected[4, 0] = 0.5  # No observation for B at this time step
        m_b2coord_expected[4, 1] = 0.5

        # t = 5
        m_a2coord_expected[5, 0] = np.prod(norm.pdf([0.6, 0.7], loc=[0.5, 0.1], scale=self.params.std_prior))
        m_a2coord_expected[5, 1] = np.prod(norm.pdf([0.6, 0.7], loc=[0.3, 0.4], scale=self.params.std_ab))

        m_b2coord_expected[5, 0] = np.prod(norm.pdf([0.8, 0.9], loc=[0.3, 0.4], scale=self.params.std_prior))
        m_b2coord_expected[5, 1] = np.prod(norm.pdf([0.8, 0.9], loc=[0.5, 0.1], scale=self.params.std_ab))

        # Message from components to coordination
        m_comp2coord_expected = m_a2coord_expected * m_b2coord_expected
        m_comp2coord_actual = inference_engine._get_messages_from_components_to_coordination()

        np.testing.assert_allclose(m_comp2coord_expected.T, m_comp2coord_actual, atol=1e-5)

    def test_forward_messages(self):
        inference_engine = DiscreteCoordinationInferenceFromVocalics(self.params.series_a,
                                                                     self.params.series_b,
                                                                     self.params.p_prior_coordination,
                                                                     self.params.p_coordination_transition,
                                                                     self.params.mean_prior,
                                                                     self.params.mean_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab)

        T = self.params.series_a.values.shape[1]
        M = int(T / 2)

        # Transition matrix
        A = np.array([[1 - self.params.p_coordination_transition, self.params.p_coordination_transition],
                      [self.params.p_coordination_transition, 1 - self.params.p_coordination_transition]])

        m_forward_expected = np.zeros((M + 1, 2))

        m_comp2coord = np.array([[0.3, 0.8],
                                 [0.4, 0.7],
                                 [0.5, 0.5],
                                 [0.6, 0.3],
                                 [0.25, 0.25],
                                 [0.3, 0.9]]).T

        # t = 0
        m_forward_expected[0, 0] = (1 - self.params.p_prior_coordination) * 0.3
        m_forward_expected[0, 1] = self.params.p_prior_coordination * 0.8
        m_forward_expected[0] = m_forward_expected[0] / np.sum(m_forward_expected[0])

        # t = 1
        m_forward_expected[1] = m_forward_expected[0] @ A * [0.4, 0.7]
        m_forward_expected[1] = m_forward_expected[1] / np.sum(m_forward_expected[1])

        # t = 2
        m_forward_expected[2] = m_forward_expected[1] @ A * [0.5, 0.5]
        m_forward_expected[2] = m_forward_expected[2] / np.sum(m_forward_expected[2])

        # t = 3 = M
        m_forward_expected[3] = m_forward_expected[2] @ A * [0.6, 0.3] * [0.25, 0.25] * [0.3, 0.9]
        m_forward_expected[3] = m_forward_expected[3] / np.sum(m_forward_expected[3])

        # Message from components to coordination
        m_forward_actual = inference_engine._forward(m_comp2coord)

        np.testing.assert_allclose(m_forward_expected.T, m_forward_actual, atol=1e-5)

    def test_backwards_messages(self):
        inference_engine = DiscreteCoordinationInferenceFromVocalics(self.params.series_a,
                                                                     self.params.series_b,
                                                                     self.params.p_prior_coordination,
                                                                     self.params.p_coordination_transition,
                                                                     self.params.mean_prior,
                                                                     self.params.mean_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab)

        T = self.params.series_a.values.shape[1]
        M = int(T / 2)

        # Transition matrix
        A = np.array([[1 - self.params.p_coordination_transition, self.params.p_coordination_transition],
                      [self.params.p_coordination_transition, 1 - self.params.p_coordination_transition]])

        m_backwards_expected = np.zeros((M + 1, 2))

        m_comp2coord = np.array([[0.3, 0.8],
                                 [0.4, 0.7],
                                 [0.5, 0.5],
                                 [0.6, 0.3],
                                 [0.25, 0.25],
                                 [0.3, 0.9]]).T

        # t = 3 = M
        m_backwards_expected[3] = np.array([0.6, 0.3]) * [0.25, 0.25] * [0.3, 0.9]
        m_backwards_expected[3] = m_backwards_expected[3] / np.sum(m_backwards_expected[3])

        # t = 2
        m_backwards_expected[2] = m_backwards_expected[3] @ A * [0.5, 0.5]
        m_backwards_expected[2] = m_backwards_expected[2] / np.sum(m_backwards_expected[2])

        # t = 1
        m_backwards_expected[1] = m_backwards_expected[2] @ A * [0.4, 0.7]
        m_backwards_expected[1] = m_backwards_expected[1] / np.sum(m_backwards_expected[1])

        # t = 0
        m_backwards_expected[0] = m_backwards_expected[1] @ A * [0.3, 0.8]
        m_backwards_expected[0] = m_backwards_expected[0] / np.sum(m_backwards_expected[0])

        # Message from components to coordination
        m_backwards_actual = inference_engine._backwards(m_comp2coord)

        np.testing.assert_allclose(m_backwards_expected.T, m_backwards_actual, atol=1e-5)

    def test_marginals(self):
        inference_engine = DiscreteCoordinationInferenceFromVocalics(self.params.series_a,
                                                                     self.params.series_b,
                                                                     self.params.p_prior_coordination,
                                                                     self.params.p_coordination_transition,
                                                                     self.params.mean_prior,
                                                                     self.params.mean_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_prior,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab,
                                                                     self.params.std_ab)

        # Overwrite method to return the values we want for this test
        inference_engine._get_messages_from_components_to_coordination = MagicMock(
            return_value=np.array([[0.3, 0.8],
                                   [0.4, 0.7],
                                   [0.5, 0.5],
                                   [0.6, 0.3],
                                   [0.25, 0.25],
                                   [0.3, 0.9]]).T
        )

        T = self.params.series_a.values.shape[1]
        M = int(T / 2)

        # Transition matrix
        A = np.array([[1 - self.params.p_coordination_transition, self.params.p_coordination_transition],
                      [self.params.p_coordination_transition, 1 - self.params.p_coordination_transition]])

        m_marginals_expected = np.zeros((M + 1, 2))
        m_forward_expected = np.zeros((M + 1, 2))
        m_backwards_expected = np.zeros((M + 1, 2))

        # Forward
        # t = 0
        m_forward_expected[0, 0] = (1 - self.params.p_prior_coordination) * 0.3
        m_forward_expected[0, 1] = self.params.p_prior_coordination * 0.8
        m_forward_expected[0] = m_forward_expected[0] / np.sum(m_forward_expected[0])

        # t = 1
        m_forward_expected[1] = m_forward_expected[0] @ A * [0.4, 0.7]
        m_forward_expected[1] = m_forward_expected[1] / np.sum(m_forward_expected[1])

        # t = 2
        m_forward_expected[2] = m_forward_expected[1] @ A * [0.5, 0.5]
        m_forward_expected[2] = m_forward_expected[2] / np.sum(m_forward_expected[2])

        # t = 3 = M
        m_forward_expected[3] = m_forward_expected[2] @ A * [0.6, 0.3] * [0.25, 0.25] * [0.3, 0.9]
        m_forward_expected[3] = m_forward_expected[3] / np.sum(m_forward_expected[3])

        # Backwards
        # t = 3 = M
        m_backwards_expected[3] = np.array([0.6, 0.3]) * [0.25, 0.25] * [0.3, 0.9]
        m_backwards_expected[3] = m_backwards_expected[3] / np.sum(m_backwards_expected[3])

        # t = 2
        m_backwards_expected[2] = m_backwards_expected[3] @ A * [0.5, 0.5]
        m_backwards_expected[2] = m_backwards_expected[2] / np.sum(m_backwards_expected[2])

        # t = 1
        m_backwards_expected[1] = m_backwards_expected[2] @ A * [0.4, 0.7]
        m_backwards_expected[1] = m_backwards_expected[1] / np.sum(m_backwards_expected[1])

        # t = 0
        m_backwards_expected[0] = m_backwards_expected[1] @ A * [0.3, 0.8]
        m_backwards_expected[0] = m_backwards_expected[0] / np.sum(m_backwards_expected[0])

        # Marginals
        # t = 0
        m_marginals_expected[0] = m_forward_expected[0] * (m_backwards_expected[1] @ A)
        m_marginals_expected[0] = m_marginals_expected[0] / np.sum(m_marginals_expected[0])

        # t = 1
        m_marginals_expected[1] = m_forward_expected[1] * (m_backwards_expected[2] @ A)
        m_marginals_expected[1] = m_marginals_expected[1] / np.sum(m_marginals_expected[1])

        # t = 2
        m_marginals_expected[2] = m_forward_expected[2] * (m_backwards_expected[3] @ A)
        m_marginals_expected[2] = m_marginals_expected[2] / np.sum(m_marginals_expected[2])

        # t = 3 = M
        m_marginals_expected[3] = m_forward_expected[3]
        m_marginals_expected[3] = m_marginals_expected[3] / np.sum(m_marginals_expected[3])

        m_marginals_actual = inference_engine.estimate_marginals()

        np.testing.assert_allclose(m_marginals_expected.T, m_marginals_actual, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
