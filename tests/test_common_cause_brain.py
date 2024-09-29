from unittest import TestCase

import numpy as np
import pytensor.tensor as ptt

from coordination.module.common_cause.common_cause_gaussian_2d import CommonCauseGaussian2D
from coordination.module.common_cause.common_cause_gaussian_2d import log_prob
from coordination.module.latent_component.non_serial_2d_gaussian_latent_component import common_cause_log_prob

from scipy.stats import norm
lp_t0 = norm.logpdf(0.1, loc=0.3, scale=0.01) + norm.logpdf(0.5, loc=0.4, scale=0.02)



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
        sample = ptt.constant(  # A sample from the system latent component
            np.array(
                [
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
            np.array(
                [
                    [0.3, 0.4],  # subject 1
                    [0.4, 0.5],  # subject 2
                    [0.5, 0.6],  # subject 3
                ]
            )
        )
        sigma = ptt.constant(
            np.array(
                [
                    [0.01, 0.02],  # subject 1
                    [0.02, 0.03],  # subject 2
                    [0.03, 0.04],  # subject 2
                ]
            )
        )
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7]))
        common_cause = ptt.constant(np.array([
            [[0.1, -0.4, 0.5], [0.5, 0.2, 0.1]]
        ]))

        # p(X, Y, Z) =
        #    p(X_0 | C_0, CC_0) p(Y_0 | C_0, CC_0) p(Z_0 | C_0, CC_0)
        #    p(X_1 | C_1, CC_1, X_0) p(Y_1 | C_1, CC_1, Y_0) p(Z_1 | C_1, CC_1, Z_0)
        #    p(X_2 | C_2, CC_2, X_1) p(Y_2 | C_2, CC_2, Y_1) p(Z_2 | C_2, CC_2, Z_1)
        #
        # The following applies to X, Y and Z
        # p(X_0 | C_0, CC_0) = N(X_0 | [(1 - C_0)]Mu_x + CC_0C_0)
        # p(X_1 | C_1, CC_1, X_0) = N(X_1 | (1 - C_1)X_0 + CC_1C_1)
        #
        # So,
        #
        # logp(X_0 | C_0, CC_0) + logp(Y_0 | C_0, CC_0) + logp(Z_0 | C_0, CC_0) =
        # norm(0.1*[0.1, 0.5] + (1 - 0.1)*[0.3, 0.4], [0.01, 0.02]).logpdf([0.1, 0.2]).sum() \
        # norm(0.1*[0.1, 0.5] + (1 - 0.1)*[0.4, 0.5], [0.02, 0.03]).logpdf([0.3, 0.4]).sum() \
        # norm(0.1*[0.1, 0.5] + (1 - 0.1)*[0.5, 0.6], [0.03, 0.04]).logpdf([0.5, 0.9]).sum()
        #
        # logp(X_1 | C_1, CC_1, X_0) + logp(Y_1 | C_1, CC_1, Y_0) + logp(Z_1 | C_1, CC_1, Z_0) =
        # norm(0.3*[-0.4, 0.2] + (1 - 0.3)*[0.1, 0.2], [0.01, 0.02]).logpdf([0.2, 0.3]).sum() \
        # norm(0.3*[-0.4, 0.2] + (1 - 0.3)*[0.3, 0.4], [0.02, 0.03]).logpdf([0.4, 0.5]).sum() \
        # norm(0.3*[-0.4, 0.2] + (1 - 0.3)*[0.5, 0.9], [0.03, 0.04]).logpdf([0.6, 0.1]).sum()
        #
        # logp(X_2 | C_2, CC_2, X_1) + logp(Y_2 | C_2, CC_2, Y_1) + logp(Z_2 | C_2, CC_2, Z_1) =
        # norm(0.7*[0.5, 0.1] + (1 - 0.7)*[0.2, 0.3], [0.01, 0.02]).logpdf([0.3, 0.4]).sum() \
        # norm(0.7*[0.5, 0.1] + (1 - 0.7)*[0.4, 0.5], [0.02, 0.03]).logpdf([0.5, 0.6]).sum() \
        # norm(0.7*[0.5, 0.1] + (1 - 0.7)*[0.6, 0.1], [0.03, 0.04]).logpdf([0.7, 1.1]).sum()
        #
        # All together:
        # norm(0.1*np.array([0.1, 0.5]) + (1 - 0.1)*np.array([0.3, 0.4]), [0.01, 0.02]).logpdf([0.1, 0.2]).sum() + \
        # norm(0.1*np.array([0.1, 0.5]) + (1 - 0.1)*np.array([0.4, 0.5]), [0.02, 0.03]).logpdf([0.3, 0.4]).sum() + \
        # norm(0.1*np.array([0.1, 0.5]) + (1 - 0.1)*np.array([0.5, 0.6]), [0.03, 0.04]).logpdf([0.5, 0.9]).sum() + \
        # norm(0.3*np.array([-0.4, 0.2]) + (1 - 0.3)*np.array([0.1, 0.2]), [0.01, 0.02]).logpdf([0.2, 0.3]).sum() + \
        # norm(0.3*np.array([-0.4, 0.2]) + (1 - 0.3)*np.array([0.3, 0.4]), [0.02, 0.03]).logpdf([0.4, 0.5]).sum() + \
        # norm(0.3*np.array([-0.4, 0.2]) + (1 - 0.3)*np.array([0.5, 0.9]), [0.03, 0.04]).logpdf([0.6, 1.0]).sum() + \
        # norm(0.7*np.array([0.5, 0.1]) + (1 - 0.7)*np.array([0.2, 0.3]), [0.01, 0.02]).logpdf([0.3, 0.4]).sum() + \
        # norm(0.7*np.array([0.5, 0.1]) + (1 - 0.7)*np.array([0.4, 0.5]), [0.02, 0.03]).logpdf([0.5, 0.6]).sum() + \
        # norm(0.7*np.array([0.5, 0.1]) + (1 - 0.7)*np.array([0.6, 1.0]), [0.03, 0.04]).logpdf([0.7, 1.1]).sum()

        real_lp = -1170.1510201486265

        lp = common_cause_log_prob(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            common_cause=common_cause,
            self_dependent=ptt.constant(True),
            symmetry_mask=1
        )

        self.assertAlmostEqual(lp.eval(), real_lp)
