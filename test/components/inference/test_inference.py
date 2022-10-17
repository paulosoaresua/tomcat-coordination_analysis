import unittest
from unittest.mock import MagicMock

import copy
from datetime import datetime

import numpy as np
from scipy.stats import norm

from coordination.common.dataset import InputFeaturesDataset, SeriesData
from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries
from coordination.model.discrete_coordination import DiscreteCoordinationInferenceFromVocalics


class DiscreteCoordinationParameters:
    # [WARN] Changing these values will break the tests
    #     x            x            x  x  x
    # A   B   A    B   A  B    A    B  A  B  A    B
    values = np.array([[0.2, 0, 0.4, 0.1, 0, 0.3, 0.5, 0, 0, 0, 0.6, 0.8],
                       [0.3, 0, 0.2, 0.3, 0, 0.4, 0.1, 0, 0, 0, 0.7, 0.9]])
    mask = np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])

    # Subject A speaks at even and B at odd time steps
    utterance_a = SegmentedUtterance("A", datetime.now(), datetime.now(), "")
    utterance_b = SegmentedUtterance("B", datetime.now(), datetime.now(), "")
    utterances = []
    for t in range(len(values)):
        if t % 2 == 0:
            utterances.append(utterance_a)
        else:
            utterances.append(utterance_b)

    previous_from_self = [None, None, 0, None, None, 3, 2, None, None, None, 6, 5]
    previous_from_other = [None, None, None, 2, None, 2, 5, None, None, None, 5, 10]

    vocalic_series = VocalicsSparseSeries(values=values, mask=mask, utterances=utterances,
                                          previous_from_self=previous_from_self,
                                          previous_from_other=previous_from_other)
    dataset = InputFeaturesDataset([SeriesData(vocalic_series)])

    p_prior_coordination = 0
    p_coordination_transition = 0.1
    mean_prior_vocalics = np.array([0, 0])
    std_prior_vocalics = np.array([1, 1])
    std_uncoordinated_vocalics = np.array([1, 1])
    std_coordinated_vocalics = np.array([1, 1])

    inference_engine = DiscreteCoordinationInferenceFromVocalics(p_prior_coordination,
                                                                 p_coordination_transition,
                                                                 mean_prior_vocalics,
                                                                 std_prior_vocalics,
                                                                 std_uncoordinated_vocalics,
                                                                 std_coordinated_vocalics,
                                                                 fix_coordination_on_second_half=True)


class TestDiscreteCoordinationInference(unittest.TestCase):
    def setUp(self):
        self.params = DiscreteCoordinationParameters

    def test_message_from_components_to_coordination(self):
        T = self.params.vocalic_series.values.shape[1]

        m_comp2coord_expected = np.zeros((T, 2))

        # t = 0 - A
        m_comp2coord_expected[0, 0] = 0.5  # No value of other at this time step
        m_comp2coord_expected[0, 1] = 0.5

        # t = 1 - B
        m_comp2coord_expected[1, 0] = 0.5  # No value of self at this time step
        m_comp2coord_expected[1, 1] = 0.5

        # t = 2 - A
        m_comp2coord_expected[2, 0] = 0.5  # No previous value of other at this time step
        m_comp2coord_expected[2, 1] = 0.5

        # t = 3 - B
        m_comp2coord_expected[3, 0] = np.prod(
            norm.pdf([0.1, 0.3], loc=self.params.mean_prior_vocalics, scale=self.params.std_prior_vocalics))
        m_comp2coord_expected[3, 1] = np.prod(
            norm.pdf([0.1, 0.3], loc=[0.4, 0.2], scale=self.params.std_coordinated_vocalics))

        # t = 4 - A
        m_comp2coord_expected[4, 0] = 0.5  # No value of self at this time step
        m_comp2coord_expected[4, 1] = 0.5

        # t = 5 - B
        m_comp2coord_expected[5, 0] = np.prod(
            norm.pdf([0.3, 0.4], loc=[0.1, 0.3], scale=self.params.std_uncoordinated_vocalics))
        m_comp2coord_expected[5, 1] = np.prod(
            norm.pdf([0.3, 0.4], loc=[0.4, 0.2], scale=self.params.std_coordinated_vocalics))

        # t = 6 - A
        m_comp2coord_expected[6, 0] = np.prod(
            norm.pdf([0.5, 0.1], loc=[0.4, 0.2], scale=self.params.std_uncoordinated_vocalics))
        m_comp2coord_expected[6, 1] = np.prod(
            norm.pdf([0.5, 0.1], loc=[0.3, 0.4], scale=self.params.std_coordinated_vocalics))

        # t = 7 - B
        m_comp2coord_expected[7, 0] = 0.5  # No value of self at this time step
        m_comp2coord_expected[7, 1] = 0.5

        # t = 8 - A
        m_comp2coord_expected[8, 0] = 0.5  # No value of self at this time step
        m_comp2coord_expected[8, 1] = 0.5

        # t = 9 - B
        m_comp2coord_expected[9, 0] = 0.5  # No value of self at this time step
        m_comp2coord_expected[9, 1] = 0.5

        # t = 10 - A
        m_comp2coord_expected[10, 0] = np.prod(
            norm.pdf([0.6, 0.7], loc=[0.5, 0.1], scale=self.params.std_uncoordinated_vocalics))
        m_comp2coord_expected[10, 1] = np.prod(
            norm.pdf([0.6, 0.7], loc=[0.3, 0.4], scale=self.params.std_coordinated_vocalics))

        # t = 11 - B
        m_comp2coord_expected[11, 0] = np.prod(
            norm.pdf([0.8, 0.9], loc=[0.3, 0.4], scale=self.params.std_uncoordinated_vocalics))
        m_comp2coord_expected[11, 1] = np.prod(
            norm.pdf([0.8, 0.9], loc=[0.6, 0.7], scale=self.params.std_coordinated_vocalics))

        # Message from components to coordination
        m_comp2coord_actual = self.params.inference_engine._get_messages_from_components_to_coordination(
            self.params.dataset.series[0])

        np.testing.assert_allclose(m_comp2coord_expected.T, m_comp2coord_actual, atol=1e-5)

    def test_forward_messages(self):
        T = self.params.vocalic_series.values.shape[1]
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
                                 [0.3, 0.9],
                                 [0.2, 0.8],
                                 [0.7, 0.5],
                                 [0.1, 0.9],
                                 [0.5, 0.9],
                                 [0.2, 0.2],
                                 [0.3, 0.4]]).T

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

        # t = 3
        m_forward_expected[3] = m_forward_expected[2] @ A * [0.6, 0.3]
        m_forward_expected[3] = m_forward_expected[3] / np.sum(m_forward_expected[3])

        # t = 4
        m_forward_expected[4] = m_forward_expected[3] @ A * [0.25, 0.25]
        m_forward_expected[4] = m_forward_expected[4] / np.sum(m_forward_expected[4])

        # t = 5
        m_forward_expected[5] = m_forward_expected[4] @ A * [0.3, 0.9]
        m_forward_expected[5] = m_forward_expected[5] / np.sum(m_forward_expected[5])

        # t = 6
        m_forward_expected[6] = m_forward_expected[5] @ A * [0.2, 0.8] * [0.7, 0.5] * [0.1, 0.9] * [0.5, 0.9] * \
                                [0.2, 0.2] * [0.3, 0.4]
        m_forward_expected[6] = m_forward_expected[6] / np.sum(m_forward_expected[6])

        # Message from components to coordination
        m_forward_actual = self.params.inference_engine._forward(m_comp2coord, self.params.dataset.series[0])

        np.testing.assert_allclose(m_forward_expected.T, m_forward_actual, atol=1e-5)

    def test_backwards_messages(self):
        T = self.params.vocalic_series.values.shape[1]
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
                                 [0.3, 0.9],
                                 [0.2, 0.8],
                                 [0.7, 0.5],
                                 [0.1, 0.9],
                                 [0.5, 0.9],
                                 [0.2, 0.2],
                                 [0.3, 0.4]]).T

        # t = 6 = M
        m_backwards_expected[6] = np.array([0.2, 0.8]) * [0.7, 0.5] * [0.1, 0.9] * [0.5, 0.9] * \
                                  [0.2, 0.2] * [0.3, 0.4]
        m_backwards_expected[6] = m_backwards_expected[6] / np.sum(m_backwards_expected[6])

        # t = 5
        m_backwards_expected[5] = m_backwards_expected[6] @ A * [0.3, 0.9]
        m_backwards_expected[5] = m_backwards_expected[5] / np.sum(m_backwards_expected[5])

        # t = 4
        m_backwards_expected[4] = m_backwards_expected[5] @ A * [0.25, 0.25]
        m_backwards_expected[4] = m_backwards_expected[4] / np.sum(m_backwards_expected[4])

        # t = 3
        m_backwards_expected[3] = m_backwards_expected[4] @ A * [0.6, 0.3]
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
        m_backwards_actual = self.params.inference_engine._backwards(m_comp2coord, self.params.dataset.series[0])

        np.testing.assert_allclose(m_backwards_expected.T, m_backwards_actual, atol=1e-5)

    def test_marginals(self):
        T = self.params.vocalic_series.values.shape[1]
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

        # t = 3
        m_forward_expected[3] = m_forward_expected[2] @ A * [0.6, 0.3]
        m_forward_expected[3] = m_forward_expected[3] / np.sum(m_forward_expected[3])

        # t = 4
        m_forward_expected[4] = m_forward_expected[3] @ A * [0.25, 0.25]
        m_forward_expected[4] = m_forward_expected[4] / np.sum(m_forward_expected[4])

        # t = 5
        m_forward_expected[5] = m_forward_expected[4] @ A * [0.3, 0.9]
        m_forward_expected[5] = m_forward_expected[5] / np.sum(m_forward_expected[5])

        # t = 6
        m_forward_expected[6] = m_forward_expected[5] @ A * [0.2, 0.8] * [0.7, 0.5] * [0.1, 0.9] * [0.5, 0.9] * \
                                [0.2, 0.2] * [0.3, 0.4]
        m_forward_expected[6] = m_forward_expected[6] / np.sum(m_forward_expected[6])

        # Backwards
        # t = 6 = M
        m_backwards_expected[6] = np.array([0.2, 0.8]) * [0.7, 0.5] * [0.1, 0.9] * [0.5, 0.9] * \
                                  [0.2, 0.2] * [0.3, 0.4]
        m_backwards_expected[6] = m_backwards_expected[6] / np.sum(m_backwards_expected[6])

        # t = 5
        m_backwards_expected[5] = m_backwards_expected[6] @ A * [0.3, 0.9]
        m_backwards_expected[5] = m_backwards_expected[5] / np.sum(m_backwards_expected[5])

        # t = 4
        m_backwards_expected[4] = m_backwards_expected[5] @ A * [0.25, 0.25]
        m_backwards_expected[4] = m_backwards_expected[4] / np.sum(m_backwards_expected[4])

        # t = 3
        m_backwards_expected[3] = m_backwards_expected[4] @ A * [0.6, 0.3]
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

        # t = 3
        m_marginals_expected[3] = m_forward_expected[3] * (m_backwards_expected[4] @ A)
        m_marginals_expected[3] = m_marginals_expected[3] / np.sum(m_marginals_expected[3])

        # t = 4
        m_marginals_expected[4] = m_forward_expected[4] * (m_backwards_expected[5] @ A)
        m_marginals_expected[4] = m_marginals_expected[4] / np.sum(m_marginals_expected[4])

        # t = 5
        m_marginals_expected[5] = m_forward_expected[5] * (m_backwards_expected[6] @ A)
        m_marginals_expected[5] = m_marginals_expected[5] / np.sum(m_marginals_expected[5])

        # t = 6 = M
        m_marginals_expected[6] = m_forward_expected[6]
        m_marginals_expected[6] = m_marginals_expected[6] / np.sum(m_marginals_expected[6])

        # Overwrite method to return the values we want for this test
        mock_inference = copy.deepcopy(self.params.inference_engine)
        mock_inference._get_messages_from_components_to_coordination = MagicMock(
            return_value=np.array([[0.3, 0.8],
                                   [0.4, 0.7],
                                   [0.5, 0.5],
                                   [0.6, 0.3],
                                   [0.25, 0.25],
                                   [0.3, 0.9],
                                   [0.2, 0.8],
                                   [0.7, 0.5],
                                   [0.1, 0.9],
                                   [0.5, 0.9],
                                   [0.2, 0.2],
                                   [0.3, 0.4]]).T
        )
        m_marginals_actual = mock_inference.predict(self.params.dataset)[0][0]

        np.testing.assert_allclose(m_marginals_expected[:, 1], m_marginals_actual, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
