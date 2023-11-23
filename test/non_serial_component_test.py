import unittest

import numpy as np
import pytensor.tensor as ptt

from coordination.component.non_serial_component import logp


class TestMixtureComponent(unittest.TestCase):
    def test_logp_with_self_dependency(self):
        # 3 subjects, 2 features and 2 time steps
        sample = ptt.constant(
            np.array(
                [
                    [[0.1, 0.3], [0.2, 0.4]],
                    [[0.2, 0.4], [0.3, 0.5]],
                    [[0.3, 0.5], [0.4, 0.6]],
                ]
            )
        )
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        coordination = ptt.constant(np.array([0.1, 0.7]))

        estimated_logp = logp(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            self_dependent=ptt.constant(np.array(True)),
        )
        real_logp = -5.973064092657509e02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_logp_without_self_dependency(self):
        # 3 subjects, 2 features and 2 time steps
        sample = ptt.constant(
            np.array(
                [
                    [[0.1, 0.3], [0.2, 0.4]],
                    [[0.2, 0.4], [0.3, 0.5]],
                    [[0.3, 0.5], [0.4, 0.6]],
                ]
            )
        )
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        coordination = ptt.constant(np.array([0.1, 0.7]))

        estimated_logp = logp(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            self_dependent=ptt.constant(np.array(False)),
        )
        real_logp = -4.673480759324176e02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
