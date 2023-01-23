import unittest

import numpy as np
import pytensor.tensor as pt

from coordination.model.components.mixture_component import mixture_logp_with_self_dependency, \
    mixture_logp_without_self_dependency


class TestMixtureComponent(unittest.TestCase):

    def test_logp_with_self_dependency(self):
        # 3 subjects, 2 features and 3 time steps
        mixture_component = pt.constant(
            np.array([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
                      [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]]))
        initial_mean = pt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = pt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        mixture_weights = pt.constant(np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]))
        coordination = pt.constant(np.array([0.1, 0.3, 0.7]))
        prev_time = pt.constant(np.array([-1, -1, 0]))
        prev_time_mask = pt.constant(np.array([0, 0, 1]))
        subject_mask = pt.constant(np.array([1, 0, 1]))
        estimated_logp = mixture_logp_with_self_dependency(mixture_component, initial_mean, sigma, mixture_weights,
                                                           coordination,
                                                           prev_time, prev_time_mask, subject_mask)
        real_logp = -5.081544319609303e+02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_logp_without_self_dependency(self):
        # 3 subjects, 2 features and 3 time steps
        mixture_component = pt.constant(
            np.array([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
                      [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]]))
        initial_mean = pt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = pt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        mixture_weights = pt.constant(np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]))
        coordination = pt.constant(np.array([0.1, 0.3, 0.7]))
        prev_time = pt.constant(np.array([-1, -1, 0]))
        prev_time_mask = pt.constant(np.array([0, 0, 1]))[None, :]
        subject_mask = pt.constant(np.array([1, 0, 1]))[None, :]
        estimated_logp = mixture_logp_without_self_dependency(mixture_component, initial_mean, sigma, mixture_weights,
                                                              coordination, prev_time, prev_time_mask, subject_mask)
        real_logp = -4.257365244151745e+02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
