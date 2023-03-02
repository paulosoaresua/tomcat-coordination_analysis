import unittest

import numpy as np
import pytensor.tensor as ptt

from coordination.component.serialized_component import serialized_logp_with_self_dependency, \
    serialized_logp_without_self_dependency


class TestMixtureComponent(unittest.TestCase):

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

        estimated_logp = serialized_logp_with_self_dependency(serialized_component=serialized_component,
                                                              initial_mean=initial_mean[subjects].T,
                                                              sigma=sigma[subjects].T,
                                                              coordination=coordination,
                                                              prev_time_same_subject=prev_time_same_subject,
                                                              prev_time_diff_subject=prev_time_diff_subject,
                                                              prev_same_subject_mask=prev_time_same_subject_mask,
                                                              prev_diff_subject_mask=prev_time_diff_subject_mask)
        real_logp = -4.303952366775294e+02

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

        estimated_logp = serialized_logp_without_self_dependency(serialized_component=serialized_component,
                                                                 initial_mean=initial_mean[subjects].T,
                                                                 sigma=sigma[subjects].T,
                                                                 coordination=coordination,
                                                                 prev_time_diff_subject=prev_time_diff_subject,
                                                                 prev_diff_subject_mask=prev_time_diff_subject_mask)

        real_logp = -3.522702366775295e+02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
