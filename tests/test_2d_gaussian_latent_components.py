from unittest import TestCase

import numpy as np
import pytensor.tensor as ptt

from coordination.module.latent_component.non_serial_2d_gaussian_latent_component import \
    log_prob as non_serial_log_prob
from coordination.module.latent_component.serial_2d_gaussian_latent_component import \
    log_prob as serial_log_prob


class Test2DGaussianLatentComponents(TestCase):
    def test_serial_log_prob(self):
        # 2 subjects, 2 features and 4 time steps
        sample = ptt.constant(
            np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])  # position  # speed
        )
        initial_mean = ptt.constant(
            np.array([[0.3, 0.4], [0.4, 0.5]])  # subject 1  # subject 2
        )
        sigma = ptt.constant(
            np.array([[0.01, 0.02], [0.02, 0.03]])  # subject 1  # subject 2
        )
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7, 0.5]))
        # Subjects: A, A, B, A
        prev_time_same_subject = ptt.constant(np.array([-1, 0, -1, 1]))
        prev_time_diff_subject = ptt.constant(np.array([-1, -1, 1, 2]))
        prev_time_same_subject_mask = ptt.switch(prev_time_same_subject >= 0, 1, 0)
        prev_time_diff_subject_mask = ptt.switch(prev_time_diff_subject >= 0, 1, 0)
        subject_indices = [0, 0, 1, 0]

        # t0: N(0.1| 0.3 + 0.4dt = 0.7, 0.01)N(0.2| 0.4, 0.02)
        # t1: N(0.2| 0.1 + 0.2dt = 0.3, 0.01)N(0.3| 0.2, 0.02)
        # t2: N(0.3| 0.4 + 0.5dt = 0.9, 0.03)N(0.4| 0.5*(1 - 0.7) + 0.7*0.3, 0.03)
        # t3: N(0.4| 0.2 + 0.3dt = 0.5, 0.01)N(0.5| 0.3*(1 - 0.5) + 0.5*0.4, 0.02)
        # norm.logpdf(0.1, loc=0.7, scale=0.01) + norm.logpdf(0.2, loc=0.4, scale=0.02) + \
        # norm.logpdf(0.2, loc=0.3, scale=0.01) + norm.logpdf(0.3, loc=0.2, scale=0.02) + \
        # norm.logpdf(0.3, loc=0.9, scale=0.02) + norm.logpdf(0.4, loc=0.36, scale=0.03) + \
        # norm.logpdf(0.4, loc=0.5, scale=0.01) + norm.logpdf(0.5, loc=0.35, scale=0.02)

        lp = serial_log_prob(
            sample=sample,
            initial_mean=initial_mean[subject_indices].T,
            sigma=sigma[subject_indices].T,
            coordination=coordination,
            prev_time_same_subject=prev_time_same_subject,
            prev_time_diff_subject=prev_time_diff_subject,
            prev_same_subject_mask=prev_time_same_subject_mask,
            prev_diff_subject_mask=prev_time_diff_subject_mask,
            self_dependent=True,
            symmetry_mask=1
        )
        real_lp = -2415.8952366775293
        self.assertAlmostEqual(lp.eval(), real_lp)

    def test_non_serial_log_prob(self):
        # 3 subjects, 2 features and 3 time steps
        sample = ptt.constant(
            np.array(
                [
                    # Subject 1
                    [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],  # position  # speed
                    # Subject 2
                    [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
                    # Subject 3
                    [[0.5, 0.6, 0.7], [0.9, 1.0, 1.1]],
                ]
            )
        )
        initial_mean = ptt.constant(
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

        # t0:
        # Subject 1: N(0.1| 0.3, 0.01)N(0.2| 0.4, 0.02)
        # Subject 2: N(0.3| 0.4, 0.02)N(0.4| 0.5, 0.03)
        # Subject 3: N(0.5| 0.5, 0.03)N(0.9| 0.6, 0.04)
        # t1:
        # Subject 1: N(0.2| 0.1 + 0.2dt = 0.3, 0.01)N(0.3| (1 - 0.3)0.2 + 0.5*0.3(0.4 + 0.9), 0.02)
        # Subject 2: N(0.4| 0.3 + 0.4dt = 0.7, 0.02)N(0.5| (1 - 0.3)0.4 + 0.5*0.3(0.2 + 0.9), 0.03)
        # Subject 3: N(0.6| 0.5 + 0.9dt = 1.4, 0.03)N(1.0| (1 - 0.3)0.9 + 0.5*0.3(0.2 + 0.4), 0.04)
        # t2:
        # Subject 1: N(0.3| 0.2 + 0.3dt = 0.5, 0.01)N(0.4| (1 - 0.7)0.3 + 0.5*0.7(0.5 + 1.0), 0.02)
        # Subject 2: N(0.5| 0.4 + 0.5dt = 0.9, 0.02)N(0.6| (1 - 0.7)0.5 + 0.5*0.7(0.3 + 1.0), 0.03)
        # Subject 3: N(0.7| 0.6 + 1.0dt = 1.6, 0.03)N(1.1| (1 - 0.7)1.0 + 0.5*0.7(0.3 + 0.5), 0.04)

        # norm.logpdf(0.1, loc=0.3, scale=0.01) + norm.logpdf(0.2, loc=0.4, scale=0.02) + \
        # norm.logpdf(0.3, loc=0.4, scale=0.02) + norm.logpdf(0.4, loc=0.5, scale=0.03) + \
        # norm.logpdf(0.5, loc=0.5, scale=0.03) + norm.logpdf(0.9, loc=0.6, scale=0.04) + \
        # norm.logpdf(0.2, loc=0.3, scale=0.01) + norm.logpdf(0.3, loc=0.335, scale=0.02) + \
        # norm.logpdf(0.4, loc=0.7, scale=0.02) + norm.logpdf(0.5, loc=0.445, scale=0.03) + \
        # norm.logpdf(0.6, loc=1.4, scale=0.03) + norm.logpdf(1.0, loc=0.72, scale=0.04) + \
        # norm.logpdf(0.3, loc=0.5, scale=0.01) + norm.logpdf(0.4, loc=0.615, scale=0.02) + \
        # norm.logpdf(0.5, loc=0.9, scale=0.02) + norm.logpdf(0.6, loc=0.605, scale=0.03) + \
        # norm.logpdf(0.7, loc=1.6, scale=0.03) + norm.logpdf(1.1, loc=0.58, scale=0.04)

        lp = non_serial_log_prob(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            self_dependent=True,
            symmetry_mask=1
        )
        real_lp = -1782.8003257041821
        self.assertAlmostEqual(lp.eval(), real_lp)
