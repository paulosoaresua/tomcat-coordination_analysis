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
        '''
        sample: (3,2,3) - 3 subjects, 2 features, 3 time steps.
        initial_mean: (3,2) - Initial mean for each subject.
        sigma: (3,2) - Standard deviation for each subject.
        coordination: (3) - Coordination values across 3 time steps.
        common_cause: (1,2,3) - Needs to be expanded to match subjects. => (3,2,3)
        '''

        '''
        # Time t=0 
        mu_t0 = coordination[0] * common_cause[:, 0] + (1 - coordination[0]) * mu_0[0]
        log_prob_x0_p = norm.logpdf(x[0, 0, 0], loc=mu_t0[0], scale=sigma[0, 0])
        log_prob_x0_v = norm.logpdf(x[0, 1, 0], loc=mu_t0[1], scale=sigma[0, 1])

        # Time t=1 
        mu_t1 = coordination[1] * common_cause[:, 1] + (1 - coordination[1]) * x[0, :, 0]
        log_prob_x1_p = norm.logpdf(x[0, 0, 1], loc=mu_t1[0], scale=sigma[0, 0])
        log_prob_x1_v = norm.logpdf(x[0, 1, 1], loc=mu_t1[1], scale=sigma[0, 1])

        # Time t=2 
        mu_t2 = coordination[2] * common_cause[:, 2] + (1 - coordination[2]) * x[0, :, 1]
        log_prob_x2_p = norm.logpdf(x[0, 0, 2], loc=mu_t2[0], scale=sigma[0, 0])
        log_prob_x2_v = norm.logpdf(x[0, 1, 2], loc=mu_t2[1], scale=sigma[0, 1])
        '''
        sample = ptt.constant(  # A sample from the system latent component
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
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7]))
        common_cause = ptt.constant(np.array([[[0.1, -0.4, 0.5], [0.5, 0.2, 0.1]]]))
        '''
# Using Matries
# t0
total_logp = np.sum(norm.logpdf(np.array([0.1, 0.2]), loc=np.array([0.3, 0.4]), scale=np.array([0.01, 0.02])))
total_logp += np.sum(norm.logpdf(np.array([0.3, 0.4]), loc=np.array([.4, 0.5]), scale=np.array([0.02, 0.03])))
total_logp += np.sum(norm.logpdf(np.array([0.5, 0.9]), loc=np.array([0.5, 0.6]), scale=np.array([0.03, 0.04])))

# t1
total_logp += np.sum(norm.logpdf(np.array([0.2, 0.3]), loc=np.dot(np.array([[1, 1], [0, 1-0.3]]), np.array([0.1, 0.2])) + np.dot(np.array([[0, 0], [0, 0.3]]), np.array([-0.4, 0.2])), scale=np.array([0.01, 0.02])))
total_logp += np.sum(norm.logpdf(np.array([0.4, 0.5]), loc=np.dot(np.array([[1, 1], [0, 1-0.3]]), np.array([0.3, 0.4])) + np.dot(np.array([[0, 0], [0, 0.3]]), np.array([-0.4, 0.2])), scale=np.array([0.02, 0.03])))
total_logp += np.sum(norm.logpdf(np.array([0.6, 1.0]), loc=np.dot(np.array([[1, 1], [0, 1-0.3]]), np.array([0.5, 0.9])) + np.dot(np.array([[0, 0], [0, 0.3]]), np.array([-0.4, 0.2])), scale=np.array([0.03, 0.04])))

# t2
total_logp += np.sum(norm.logpdf(np.array([0.3, 0.4]), loc=np.dot(np.array([[1, 1], [0, 1-0.7]]), np.array([0.2, 0.3])) + np.dot(np.array([[0, 0], [0, 0.7]]), np.array([0.5, 0.1])), scale=np.array([0.01, 0.02])))
total_logp += np.sum(norm.logpdf(np.array([0.5, 0.6]), loc=np.dot(np.array([[1, 1], [0, 1-0.7]]), np.array([0.4, 0.5])) + np.dot(np.array([[0, 0], [0, 0.7]]), np.array([0.5, 0.1])), scale=np.array([0.02, 0.03])))
total_logp += np.sum(norm.logpdf(np.array([0.7, 1.1]), loc=np.dot(np.array([[1, 1], [0, 1-0.7]]), np.array([0.6, 1.0])) + np.dot(np.array([[0, 0], [0, 0.7]]), np.array([0.5, 0.1])), scale=np.array([0.03, 0.04])))

[[-196.31376835  -47.00691553]
 [  -9.50691553   -2.96793619]
 [   2.58761936  -25.82506271]]
 
[[[ -46.31376835 -196.31376835]
  [  -9.50691553  -69.00691553]]

 [[-109.50691553 -197.00691553]
  [ -11.63460286  -77.63460286]]
  

 [[-352.96793619 -447.41238064]
  [ -27.73131271 -164.23131271]]]
-1988.3003257041823
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
        total_logp = -1988.3003257041823

        self.assertAlmostEqual(lp.eval(), total_logp)


    def test_common_cause_random(self):
        # Set up the inputs
        initial_mean = np.array([[0.3, 0.4],  # subject 1
                                 [0.4, 0.5],  # subject 2
                                 [0.5, 0.6]])  # subject 3
        sigma = np.array([[0.01, 0.02],  # subject 1
                          [0.02, 0.03],  # subject 2
                          [0.03, 0.04]])  # subject 3
        coordination = np.array([0.1, 0.3, 0.7])
        common_cause = np.array([[[0.1, -0.4, 0.5], [0.5, 0.2, 0.1]]])  # shape (1, 2, 3)

        self_dependent = True
        symmetry_mask = 1
        rng = np.random.default_rng(seed=42)

        sample = common_cause_random(
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            self_dependent=self_dependent,
            symmetry_mask=symmetry_mask,
            rng=rng,
            size=(3, 2, 3),  # S x D x T
            common_cause=common_cause
        )

        # Check the sample has the correct shape
        self.assertEqual(sample.shape, (3, 2, 3))

        # Same random seed testing
        rng = np.random.default_rng(seed=42)
        expected_sample = common_cause_random(
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            self_dependent=self_dependent,
            symmetry_mask=symmetry_mask,
            rng=rng,
            size=(3, 2, 3),
            common_cause=common_cause
        )

        np.testing.assert_allclose(sample, expected_sample, atol=1e-10)

        print("Sample generated by common_cause_random: ")
        print(sample)



unittest.main()