import unittest

import numpy as np
import pytensor.tensor as ptt

from coordination.common.activation_function import ActivationFunction
from coordination.common.functions import one_hot_encode
from coordination.component.serialized_component import blending_logp, \
    blending_logp_no_self_dependency, mixture_logp, mixture_logp_no_self_dependency


class TestSerializedComponent(unittest.TestCase):

    def test_logp_with_self_dependency(self):
        # 2 subjects, 2 features and 4 time steps
        serialized_component = ptt.constant(
            np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03]]))
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7, 0.5]))
        prev_time_same_subject = ptt.constant(np.array([-1, 0, -1, 1]))
        prev_time_diff_subject = ptt.constant(np.array([-1, -1, 1, 2]))
        prev_time_same_subject_mask = ptt.switch(prev_time_same_subject >= 0, 1, 0)
        prev_time_diff_subject_mask = ptt.switch(prev_time_diff_subject >= 0, 1, 0)
        subjects = ptt.constant(np.array([0, 0, 1, 0]))
        curr_subjects = one_hot_encode(np.minimum(subjects.eval(), 0), 1)
        prev_diff_subjects = one_hot_encode(subjects[prev_time_diff_subject].eval(), 2)
        pairs = ptt.constant(
            np.einsum("ijk,jlk->ilk", prev_diff_subjects[:, None, :], curr_subjects[None, :, :]).reshape(2, 4))

        # We add a bias of 1 in the first layer. The hidden and output layers do nothing.
        input_layer_f = ptt.constant([
            [1, 0],  # features
            [0, 1],
            [0, 0],  # pairs
            [0, 0],
            [1, 1],  # bias
        ])
        hidden_layers_f = ptt.constant([
            # h1
            [[1, 0],
             [0, 1],
             [0, 0]],   # bias

            # h2
            [[1, 0],
             [0, 1],
             [0, 0]],  # bias
        ]).reshape(2 * 3, 2)
        output_layer_f = ptt.constant([
            [1, 0],
            [0, 1],
            [0, 0],  # bias
        ])
        activation_function_number_f = ptt.constant(ActivationFunction.NAME_TO_NUMBER["linear"])

        estimated_logp = blending_logp(serialized_component=serialized_component,
                                       initial_mean=initial_mean[subjects].T,
                                       sigma=sigma[subjects].T,
                                       coordination=coordination,
                                       input_layer_f=input_layer_f,
                                       hidden_layers_f=hidden_layers_f,
                                       output_layer_f=output_layer_f,
                                       activation_function_number_f=activation_function_number_f,
                                       prev_time_same_subject=prev_time_same_subject,
                                       prev_time_diff_subject=prev_time_diff_subject,
                                       prev_same_subject_mask=prev_time_same_subject_mask,
                                       prev_diff_subject_mask=prev_time_diff_subject_mask,
                                       pairs=pairs)

        real_logp = -1.839006347788640e+03
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

        estimated_logp = mixture_logp(serialized_component=serialized_component,
                                      initial_mean=initial_mean[subjects].T,
                                      sigma=sigma[subjects].T,
                                      coordination=coordination,
                                      input_layer_f=input_layer_f,
                                      hidden_layers_f=hidden_layers_f,
                                      output_layer_f=output_layer_f,
                                      activation_function_number_f=activation_function_number_f,
                                      prev_time_same_subject=prev_time_same_subject,
                                      prev_time_diff_subject=prev_time_diff_subject,
                                      prev_same_subject_mask=prev_time_same_subject_mask,
                                      prev_diff_subject_mask=prev_time_diff_subject_mask,
                                      pairs=pairs)

        real_logp = -5.587311433139678e+02
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_logp_without_self_dependency(self):
        # 2 subjects, 2 features and 4 time steps
        serialized_component = ptt.constant(
            np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03]]))
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7, 0.5]))
        prev_time_diff_subject = ptt.constant(np.array([-1, -1, 1, 2]))
        prev_time_diff_subject_mask = ptt.switch(prev_time_diff_subject >= 0, 1, 0)
        subjects = ptt.constant(np.array([0, 0, 1, 0]))

        curr_subjects = one_hot_encode(np.minimum(subjects.eval(), 0), 1)
        prev_diff_subjects = one_hot_encode(subjects[prev_time_diff_subject].eval(), 2)
        pairs = ptt.constant(
            np.einsum("ijk,jlk->ilk", prev_diff_subjects[:, None, :], curr_subjects[None, :, :]).reshape(2, 4))

        # We add a bias of 0.5 in the first layer. The hidden and output layers do nothing.
        input_layer_f = ptt.constant([
            [1,   0],  # features
            [0,   1],
            [0,   0],  # pairs
            [0,   0],
            [.5, .5],  # bias
        ])
        hidden_layers_f = ptt.constant([
            # h1
            [[1, 0],
             [0, 1],
             [0, 0]],  # bias

            # h2
            [[1, 0],
             [0, 1],
             [0, 0]],  # bias
        ]).reshape(2 * 3, 2)
        output_layer_f = ptt.constant([
            [1, 0],
            [0, 1],
            [0, 0],  # bias
        ])
        activation_function_number_f = ptt.constant(ActivationFunction.NAME_TO_NUMBER["linear"])

        estimated_logp = blending_logp_no_self_dependency(serialized_component=serialized_component,
                                                          initial_mean=initial_mean[subjects].T,
                                                          sigma=sigma[subjects].T,
                                                          coordination=coordination,
                                                          input_layer_f=input_layer_f,
                                                          hidden_layers_f=hidden_layers_f,
                                                          output_layer_f=output_layer_f,
                                                          activation_function_number_f=activation_function_number_f,
                                                          prev_time_diff_subject=prev_time_diff_subject,
                                                          prev_diff_subject_mask=prev_time_diff_subject_mask,
                                                          pairs=pairs)
        real_logp = -6.010202366775294e+02
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

        estimated_logp = mixture_logp_no_self_dependency(serialized_component=serialized_component,
                                                         initial_mean=initial_mean[subjects].T,
                                                         sigma=sigma[subjects].T,
                                                         coordination=coordination,
                                                         input_layer_f=input_layer_f,
                                                         hidden_layers_f=hidden_layers_f,
                                                         output_layer_f=output_layer_f,
                                                         activation_function_number_f=activation_function_number_f,
                                                         prev_time_diff_subject=prev_time_diff_subject,
                                                         prev_diff_subject_mask=prev_time_diff_subject_mask,
                                                         pairs=pairs)
        real_logp = -3.712311433139679e+02
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
