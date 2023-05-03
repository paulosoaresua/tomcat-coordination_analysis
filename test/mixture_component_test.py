import unittest

import numpy as np
import pytensor.tensor as ptt

from coordination.common.activation_function import ActivationFunction
from coordination.component.mixture_component import mixture_logp


class TestMixtureComponent(unittest.TestCase):

    def test_logp_with_self_dependency(self):
        # 3 subjects, 2 features and 2 time steps
        mixture_component = ptt.constant(
            np.array([[[0.1, 0.3], [0.2, 0.4]], [[0.2, 0.4], [0.3, 0.5]], [[0.3, 0.5], [0.4, 0.6]]]))
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        mixture_weights = ptt.constant(np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]))
        coordination = ptt.constant(np.array([0.1, 0.7]))

        expander_aux_mask_matrix = []
        aggregator_aux_mask_matrix = []
        for subject in range(3):
            expander_aux_mask_matrix.append(np.delete(np.eye(3), subject, axis=0))
            aux = np.zeros((3, 2))
            aux[subject] = 1
            aux = aux * mixture_weights[subject][None, :]
            aggregator_aux_mask_matrix.append(aux)

        expander_aux_mask_matrix = np.concatenate(expander_aux_mask_matrix, axis=0)
        aggregator_aux_mask_matrix = ptt.concatenate(aggregator_aux_mask_matrix, axis=1)

        # We add a bias of 0 in the first layer. The hidden and output layers do nothing.
        input_layer_f = ptt.constant(
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ])
        )
        hidden_layers_f = ptt.constant(np.array([
            # h1
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ]),

            # h2
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ])
        ])).reshape((2 * 13, 12))
        output_layer_f = ptt.constant(
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ])
        )
        activation_function_number_f = ptt.constant(ActivationFunction.NAME_TO_NUMBER["linear"])

        estimated_logp = mixture_logp(mixture_component=mixture_component,
                                      initial_mean=initial_mean,
                                      sigma=sigma,
                                      mixture_weights=mixture_weights,
                                      coordination=coordination,
                                      input_layer_f=input_layer_f,
                                      hidden_layers_f=hidden_layers_f,
                                      output_layer_f=output_layer_f,
                                      activation_function_number_f=activation_function_number_f,
                                      expander_aux_mask_matrix=ptt.constant(
                                          expander_aux_mask_matrix),
                                      aggregation_aux_mask_matrix=aggregator_aux_mask_matrix,
                                      prev_time_diff_subject=ptt.repeat((ptt.arange(2) - 1)[None, :], 6, axis=0),
                                      prev_diff_subject_mask=ptt.constant(np.array([[0, 1]]).repeat(6, axis=0)),
                                      self_dependent=ptt.constant(np.array(True)))
        real_logp = -5.081544319609303e+02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_logp_without_self_dependency(self):
        # 3 subjects, 2 features and 2 time steps
        mixture_component = ptt.constant(
            np.array([[[0.1, 0.3], [0.2, 0.4]], [[0.2, 0.4], [0.3, 0.5]], [[0.3, 0.5], [0.4, 0.6]]]))
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        mixture_weights = ptt.constant(np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]))
        coordination = ptt.constant(np.array([0.1, 0.7]))

        expander_aux_mask_matrix = []
        aggregator_aux_mask_matrix = []
        for subject in range(3):
            expander_aux_mask_matrix.append(np.delete(np.eye(3), subject, axis=0))
            aux = np.zeros((3, 2))
            aux[subject] = 1
            aux = aux * mixture_weights[subject][None, :]
            aggregator_aux_mask_matrix.append(aux)

        expander_aux_mask_matrix = np.concatenate(expander_aux_mask_matrix, axis=0)
        aggregator_aux_mask_matrix = ptt.concatenate(aggregator_aux_mask_matrix, axis=1)

        # We add a bias of 1 in the first layer. The hidden and output layers do nothing.
        input_layer_f = ptt.constant(
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ])
        )
        hidden_layers_f = ptt.constant(np.array([
            # h1
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ]),

            # h2
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ])
        ])).reshape((2 * 13, 12))
        output_layer_f = ptt.constant(
            np.vstack([
                np.eye(3 * (3 - 1) * 2),  # s * (s-1) * d
                np.zeros(12)  # bias
            ])
        )
        activation_function_number_f = ptt.constant(ActivationFunction.NAME_TO_NUMBER["linear"])

        estimated_logp = mixture_logp(mixture_component=mixture_component,
                                      initial_mean=initial_mean,
                                      sigma=sigma,
                                      mixture_weights=mixture_weights,
                                      coordination=coordination,
                                      input_layer_f=input_layer_f,
                                      hidden_layers_f=hidden_layers_f,
                                      output_layer_f=output_layer_f,
                                      activation_function_number_f=activation_function_number_f,
                                      expander_aux_mask_matrix=ptt.constant(
                                          expander_aux_mask_matrix),
                                      aggregation_aux_mask_matrix=aggregator_aux_mask_matrix,
                                      prev_time_diff_subject=ptt.repeat((ptt.arange(2) - 1)[None, :], 6, axis=0),
                                      prev_diff_subject_mask=ptt.constant(np.array([[0, 1]]).repeat(6, axis=0)),
                                      self_dependent=ptt.constant(np.array(False)))
        real_logp = -4.257365244151745e+02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
