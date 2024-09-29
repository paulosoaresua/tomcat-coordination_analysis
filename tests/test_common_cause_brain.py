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



        sample_val = sample.eval()
        initial_mean_val = initial_mean.eval()
        sigma_val = sigma.eval()
        
        logp_t0 = 0
        for i in range(3):  # 3 subjects
            for j in range(2):  # position and speed
                logp_t0 += norm.logpdf(sample_val[i, j, 0], loc=initial_mean_val[i, j], scale=sigma_val[i, j])

        # Function to compute blended mean for time t > 0
        def compute_blended_mean(subject_idx, time_idx):
            previous_values = sample_val[subject_idx, :, time_idx-1]  # Previous values at t-1
            cc_val = common_cause.eval()[0, :, time_idx-1]  # Common cause for all subjects

            # Calculate F and U based on coordination values
            c = coordination.eval()[time_idx]
            F = np.array([[1.0, 1.0], [0.0, 1 - c]])  # F matrix
            U = np.array([[0.0, 0.0], [0.0, c]])  # U matrix

            prev_same_transformed = np.dot(F, previous_values)
            common_cause_transformed = np.dot(U, cc_val)

            blended_mean = prev_same_transformed + common_cause_transformed
            return blended_mean

        # Calculate log-prob for t = 1, 2
        logp_t1_t2 = 0
        for t in range(1, 3):  # For t=1 and t=2
            for i in range(3):  # 3 subjects
                blended_mean = compute_blended_mean(i, t)
                for j in range(2):  # position and speed
                    logp_t1_t2 += norm.logpdf(sample_val[i, j, t], loc=blended_mean[j], scale=sigma_val[i, j])

        # Total log-prob
        total_logp = logp_t0 + logp_t1_t2
        print(total_logp)
        # total_logp = -1856.4044923708486 

        lp = common_cause_log_prob(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            common_cause=common_cause,
            self_dependent=ptt.constant(True),
            symmetry_mask=1
        )

        self.assertAlmostEqual(lp.eval(), total_logp)


unittest.main()