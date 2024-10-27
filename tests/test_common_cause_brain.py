from unittest import TestCase
import unittest
import numpy as np
import pytensor.tensor as ptt

from coordination.module.common_cause.common_cause_gaussian_2d import CommonCauseGaussian2D
from coordination.module.common_cause.common_cause_gaussian_2d import log_prob
from coordination.module.latent_component.non_serial_2d_gaussian_latent_component import common_cause_log_prob

from scipy.stats import norm



class Test2DCommonCause(TestCase):
    def test_non_serial_log_prob(self):
        # Shape of sample (1, 2, T = 4)
        sample = ptt.constant(np.array([
            [
                [0.1, -0.4, 0.5, 0.6],
                [0.5, 0.2, 0.1, 0.3]
            ]
        ]))
        # The shapes below are (1, 2) representing (1 subject, position and velocity)
        initial_mean = ptt.constant(np.array([[0.3, 0.4]]))  # at time t = 0
        sigma = ptt.constant(np.array([[0.01, 0.02]]))  # at all times

        # Log-probability
        # lp_t0 + lp_t1 + lp_t2 + lp_t3
        # lp_t0 = log-prob of N(0.1 | 0.3, 0.01^2) + log-prob of N(0.5 | 0.4, 0.02^2)
        # lp_t1 = log-prob of N(-0.4 | 0.1 + 0.5 = 0.6, 0.01^2) + log-prob of N(0.2 | 0.5, 0.02^2)
        # lp_t2 = log-prob of N(0.5 | -0.4 + 0.2 = -0.2, 0.01^2) + log-prob of N(0.1 | 0.2, 0.02^2)
        # lp_t3 = log-prob of N(0.6 | 0.5 + 0.1 = 0.6, 0.01^2) + log-prob of N(0.3 | 0.1, 0.02^2)

        # norm(0.3, 0.01).logpdf(0.1) + norm(0.4, 0.02).logpdf(0.5) + \
        # norm(0.6, 0.01).logpdf(-0.4) + norm(0.5, 0.02).logpdf(0.2) + \
        # norm(-0.2, 0.01).logpdf(0.5) + norm(0.2, 0.02).logpdf(0.1) + \
        # norm(0.6, 0.01).logpdf(0.6) + norm(0.1, 0.02).logpdf(0.3)

        lp = log_prob(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma
        )
        real_lp = -7810.782735499971
        self.assertAlmostEqual(lp.eval(), real_lp)

    def test_common_cause_2d_latent_component_log_prob(self):
        # TODO: [Ming] Find the real value of log prob and implement the common_cause_log_prob.
        # 3 subjects, 2 features and 3 time steps
        sample = ptt.constant(
            np.array([
                    # Subject 1 (X)
                    [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],  # position  # speed
                    # Subject 2 (Y)
                    [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
                    # Subject 3 (Z)
                    [[0.5, 0.6, 0.7], [0.9, 1.0, 1.1]],
                ]
            )
        )
        # Mu
        initial_mean = ptt.constant( # The value of the system latent component at time t = 0
            np.array([[0.3, 0.4],  # subject 1
                    [0.4, 0.5],  # subject 2
                    [0.5, 0.6],  # subject 3
                    ]))
        sigma = ptt.constant(
            np.array([[0.01, 0.02],  # subject 1
                    [0.02, 0.03],  # subject 2
                    [0.03, 0.04],  # subject 2
                ]))
        coordination = ptt.constant(np.array([
                [0.1, 0.2, 0.3],
                [0.5, 0.6, 0.7],
                [0.9, 1.0, 1.1]
                ])
            )
        common_cause = ptt.constant(np.array([[[0.1, -0.4, 0.5], [0.5, 0.2, 0.1]]]))
        '''

total_logp = np.sum(norm.logpdf(np.array([0.1, 0.2]), loc=np.array([0.3, 0.4]), scale=np.array([0.01, 0.02])))
total_logp += np.sum(norm.logpdf(np.array([0.3, 0.4]), loc=np.array([.4, 0.5]), scale=np.array([0.02, 0.03])))
total_logp += np.sum(norm.logpdf(np.array([0.5, 0.9]), loc=np.array([0.5, 0.6]), scale=np.array([0.03, 0.04])))

total_logp += np.sum(norm.logpdf(np.array([0.2, 0.3]), loc=np.dot(np.array([[1, 1], [0, 0.6/1.8]]), np.array([0.1, 0.2])) + np.dot(np.array([[0, 0], [0, 0.2/1.8]]), np.array([0.4, 0.65])) + np.dot(np.array([[0, 0], [0, 1.0/1.8]]), np.array([-0.4, 0.2])), scale=np.array([0.01, 0.02])))
total_logp += np.sum(norm.logpdf(np.array([0.4, 0.5]), loc=np.dot(np.array([[1, 1], [0, 0.6/1.8]]), np.array([0.3, 0.4])) + np.dot(np.array([[0, 0], [0, 0.2/1.8]]), np.array([0.3, 0.55])) + np.dot(np.array([[0, 0], [0, 1.0/1.8]]), np.array([-0.4, 0.2])), scale=np.array([0.02, 0.03])))
total_logp += np.sum(norm.logpdf(np.array([0.6, 1.0]), loc=np.dot(np.array([[1, 1], [0, 0.6/1.8]]), np.array([0.5, 0.9])) + np.dot(np.array([[0, 0], [0, 0.2/1.8]]), np.array([0.2, 0.30])) + np.dot(np.array([[0, 0], [0, 1.0/1.8]]), np.array([-0.4, 0.2])), scale=np.array([0.03, 0.04])))

total_logp += np.sum(norm.logpdf(np.array([0.3, 0.4]), loc=np.dot(np.array([[1, 1], [0, 0.7/2.1]]), np.array([0.2, 0.3])) + np.dot(np.array([[0, 0], [0, 0.3/2.1]]), np.array([0.5, 0.75])) + np.dot(np.array([[0, 0], [0, 1.1/2.1]]), np.array([0.5, 0.1])), scale=np.array([0.01, 0.02])))
total_logp += np.sum(norm.logpdf(np.array([0.5, 0.6]), loc=np.dot(np.array([[1, 1], [0, 0.7/2.1]]), np.array([0.4, 0.5])) + np.dot(np.array([[0, 0], [0, 0.3/2.1]]), np.array([0.4, 0.65])) + np.dot(np.array([[0, 0], [0, 1.1/2.1]]), np.array([0.5, 0.1])), scale=np.array([0.02, 0.03])))
total_logp += np.sum(norm.logpdf(np.array([0.7, 1.1]), loc=np.dot(np.array([[1, 1], [0, 0.7/2.1]]), np.array([0.6, 1.0])) + np.dot(np.array([[0, 0], [0, 0.3/2.1]]), np.array([0.3, 0.40])) + np.dot(np.array([[0, 0], [0, 1.1/2.1]]), np.array([0.5, 0.1])), scale=np.array([0.03, 0.04])))


[[[-46.31, -196.31],
 [-0.13,  -21.67]],

[[-109.51, -197.01],
[ -18.42,  -43.52]],

[[-352.97, -447.41],
[ -94.15, -132.65]]]

-1939.1002105758262

        '''

        lp = common_cause_log_prob(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            common_cause=common_cause,
            self_dependent=ptt.constant(True),
            symmetry_mask=1
        )
        total_logp = -1939.1002105758262

        self.assertAlmostEqual(lp.eval(), total_logp)


if __name__ == "__main__":
    unittest.main()
