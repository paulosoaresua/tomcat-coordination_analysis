from unittest import TestCase

import numpy as np
import pytensor.tensor as ptt

from coordination.module.latent_component.non_serial_gaussian_latent_component import \
    log_prob as non_serial_log_prob
from coordination.module.latent_component.serial_gaussian_latent_component import \
    log_prob as serial_log_prob


class TestGaussianLatentComponents(TestCase):
    def test_serial_log_prob_with_self_dependency(self):
        # 2 subjects, 2 features and 4 time steps
        sample = ptt.constant(np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03]]))
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7, 0.5]))
        prev_time_same_subject = ptt.constant(np.array([-1, 0, -1, 1]))
        prev_time_diff_subject = ptt.constant(np.array([-1, -1, 1, 2]))
        prev_time_same_subject_mask = ptt.switch(prev_time_same_subject >= 0, 1, 0)
        prev_time_diff_subject_mask = ptt.switch(prev_time_diff_subject >= 0, 1, 0)
        subjects = ptt.constant(np.array([0, 0, 1, 0]))

        estimated_logp = serial_log_prob(
            sample=sample,
            initial_mean=initial_mean[subjects].T,
            sigma=sigma[subjects].T,
            coordination=coordination,
            prev_time_same_subject=prev_time_same_subject,
            prev_time_diff_subject=prev_time_diff_subject,
            prev_same_subject_mask=prev_time_same_subject_mask,
            prev_diff_subject_mask=prev_time_diff_subject_mask,
            self_dependent=ptt.constant(np.array(True)),
        )

        real_logp = -4.303952366775294e02
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_serial_log_prob_without_self_dependency(self):
        # 2 subjects, 2 features and 4 time steps
        sample = ptt.constant(np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
        initial_mean = ptt.constant(np.array([[0.3, 0.4], [0.4, 0.5]]))
        sigma = ptt.constant(np.array([[0.01, 0.02], [0.02, 0.03]]))
        coordination = ptt.constant(np.array([0.1, 0.3, 0.7, 0.5]))
        prev_time_diff_subject = ptt.constant(np.array([-1, -1, 1, 2]))
        prev_time_diff_subject_mask = ptt.switch(prev_time_diff_subject >= 0, 1, 0)
        subjects = ptt.constant(np.array([0, 0, 1, 0]))

        estimated_logp = serial_log_prob(
            sample=sample,
            initial_mean=initial_mean[subjects].T,
            sigma=sigma[subjects].T,
            coordination=coordination,
            prev_time_same_subject=ptt.constant([]),
            prev_time_diff_subject=prev_time_diff_subject,
            prev_same_subject_mask=ptt.constant([]),
            prev_diff_subject_mask=prev_time_diff_subject_mask,
            self_dependent=ptt.constant(np.array(False)),
        )
        real_logp = -3.522702366775295e02
        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_non_serial_log_prob_with_self_dependency(self):
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

        estimated_logp = non_serial_log_prob(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            self_dependent=ptt.constant(np.array(True)),
        )
        real_logp = -5.973064092657509e02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_non_serial_log_prob_without_self_dependency(self):
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

        estimated_logp = non_serial_log_prob(
            sample=sample,
            initial_mean=initial_mean,
            sigma=sigma,
            coordination=coordination,
            self_dependent=ptt.constant(np.array(False)),
        )
        real_logp = -4.673480759324176e02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
