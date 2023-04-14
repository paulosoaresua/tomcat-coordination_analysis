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
        prev_same_subjects = ptt.constant(one_hot_encode(subjects[prev_time_same_subject].eval(), 2))
        prev_diff_subjects = ptt.constant(one_hot_encode(subjects[prev_time_diff_subject].eval(), 2))

        # We add a bias of 1 in the first layer. The second layer does nothing. It's just to test the model
        # can be deeper.
        f_nn_weights = ptt.constant([
            # num_features (2) + 2 ohe (size 2) + bias = 7 rows
            np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [1, 1]]),
            np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0]]),
        ]).reshape((2 * 7, 2))
        f_activation_function_number = ActivationFunction.NAME_TO_NUMBER["linear"]

        estimated_logp = blending_logp(serialized_component=serialized_component,
                                       initial_mean=initial_mean[subjects].T,
                                       sigma=sigma[subjects].T,
                                       coordination=coordination,
                                       f_nn_weights=f_nn_weights,
                                       f_activation_function_number=f_activation_function_number,
                                       prev_time_same_subject=prev_time_same_subject,
                                       prev_time_diff_subject=prev_time_diff_subject,
                                       prev_same_subject_mask=prev_time_same_subject_mask,
                                       prev_diff_subject_mask=prev_time_diff_subject_mask,
                                       prev_same_subjects=prev_same_subjects,
                                       prev_diff_subjects=prev_diff_subjects)

        real_logp = -1.839006347788640e+03
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

        estimated_logp = mixture_logp(serialized_component=serialized_component,
                                      initial_mean=initial_mean[subjects].T,
                                      sigma=sigma[subjects].T,
                                      coordination=coordination,
                                      f_nn_weights=f_nn_weights,
                                      f_activation_function_number=f_activation_function_number,
                                      prev_time_same_subject=prev_time_same_subject,
                                      prev_time_diff_subject=prev_time_diff_subject,
                                      prev_same_subject_mask=prev_time_same_subject_mask,
                                      prev_diff_subject_mask=prev_time_diff_subject_mask,
                                      prev_same_subjects=prev_same_subjects,
                                      prev_diff_subjects=prev_diff_subjects)

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
        prev_diff_subjects = ptt.constant(one_hot_encode(subjects[prev_time_diff_subject].eval(), 2))

        # We add a bias of 0.5 in the first layer. The second layer does nothing. It's just to test the model
        # can be deeper.
        f_nn_weights = ptt.constant([
            # num_features (2) + 1 ohe (size 2) + bias = 5 rows
            np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [.5, .5]]),
            np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0]]),
        ]).reshape((2 * 5, 2))
        f_activation_function_number = ActivationFunction.NAME_TO_NUMBER["linear"]

        estimated_logp = blending_logp_no_self_dependency(serialized_component=serialized_component,
                                                          initial_mean=initial_mean[subjects].T,
                                                          sigma=sigma[subjects].T,
                                                          coordination=coordination,
                                                          f_nn_weights=f_nn_weights,
                                                          f_activation_function_number=f_activation_function_number,
                                                          prev_time_diff_subject=prev_time_diff_subject,
                                                          prev_diff_subject_mask=prev_time_diff_subject_mask,
                                                          prev_diff_subjects=prev_diff_subjects)
        real_logp = -6.010202366775294e+02
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

        estimated_logp = mixture_logp_no_self_dependency(serialized_component=serialized_component,
                                                         initial_mean=initial_mean[subjects].T,
                                                         sigma=sigma[subjects].T,
                                                         coordination=coordination,
                                                         f_nn_weights=f_nn_weights,
                                                         f_activation_function_number=f_activation_function_number,
                                                         prev_time_diff_subject=prev_time_diff_subject,
                                                         prev_diff_subject_mask=prev_time_diff_subject_mask,
                                                         prev_diff_subjects=prev_diff_subjects)
        real_logp = -3.712311433139679e+02
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
