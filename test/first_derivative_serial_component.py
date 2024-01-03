from unittest import TestCase

import numpy as np
import pytensor.tensor as ptt

from coordination.module.latent_component.serial_2d_gaussian_latent_component import \
    log_prob


class TestFirstDerivativeSerialComponent:
    def test_log_prob(self):
        sample = ptt.constant(np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03]]))
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7, 0.5]))
        # Subjects: A, A, B, A
        prev_time_same_subject = ptt.constant(np.array([-1, 0, -1, 1]))
        prev_time_diff_subject = ptt.constant(np.array([-1, -1, 1, 2]))
        prev_time_same_subject_mask = ptt.switch(prev_time_same_subject >= 0, 1, 0)
        prev_time_diff_subject_mask = ptt.switch(prev_time_diff_subject >= 0, 1, 0)
        subject_indices = [0, 0, 1, 0]

        # t0: N(0.1| 0.3 + 0.4dt, 0.01)N(0.2| 0.4, 0.02)
        # t1: N(0.2| 0.1 + 0.2dt, 0.01)N(0.3| 0.2, 0.02)
        # t2: N(0.3| 0.4 + 0.5dt, 0.03)N(0.4| 0.5*(1 - 0.7) + 0.7*0.3, 0.03)
        # t3: N(0.4| 0.2 + 0.3dt, 0.01)N(0.5| 0.3*(1 - 0.5) + 0.5*0.4, 0.02)
        # norm.logpdf(0.1, loc=0.7, scale=0.01) + norm.logpdf(0.2, loc=0.4, scale=0.02) + \
        # norm.logpdf(0.2, loc=0.3, scale=0.01) + norm.logpdf(0.3, loc=0.2, scale=0.02) + \
        # norm.logpdf(0.3, loc=0.9, scale=0.02) + norm.logpdf(0.4, loc=0.36, scale=0.03) + \
        # norm.logpdf(0.4, loc=0.5, scale=0.01) + norm.logpdf(0.5, loc=0.35, scale=0.02)

        lp = log_prob(
            sample=sample,
            initial_mean=initial_mean[subject_indices].T,
            sigma=sigma[subject_indices].T,
            coordination=coordination,
            prev_time_same_subject=prev_time_same_subject,
            prev_time_diff_subject=prev_time_diff_subject,
            prev_same_subject_mask=prev_time_same_subject_mask,
            prev_diff_subject_mask=prev_time_diff_subject_mask,
            self_dependent=True,
        )
        real_lp = -2415.8952366775293
        assert np.isclose(lp.eval(), real_lp)
