import unittest

import numpy as np
import pytensor.tensor as pt

from coordination.model.components.mixture_component import mixture_logp_with_self_dependency, \
    mixture_logp_without_self_dependency


class TestMixtureComponent(unittest.TestCase):

    def test_logp_with_self_dependency(self):
        # 3 subjects, 2 features and 2 time steps
        mixture_component = pt.constant(
            np.array([[[0.1, 0.3], [0.2, 0.4]], [[0.2, 0.4], [0.3, 0.5]], [[0.3, 0.5], [0.4, 0.6]]]))
        initial_mean = pt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = pt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        mixture_weights = pt.constant(np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]))
        coordination = pt.constant(np.array([0.1, 0.7]))

        expander_aux_mask_matrix = []
        aggregator_aux_mask_matrix = []
        for subject in range(3):
            expander_aux_mask_matrix.append(np.delete(np.eye(3), subject, axis=0))
            aux = np.zeros((3, 2))
            aux[subject] = 1
            aux = aux * mixture_weights[subject][None, :]
            aggregator_aux_mask_matrix.append(aux)

        expander_aux_mask_matrix = np.concatenate(expander_aux_mask_matrix, axis=0)
        aggregator_aux_mask_matrix = pt.concatenate(aggregator_aux_mask_matrix, axis=1)

        estimated_logp = mixture_logp_with_self_dependency(mixture_component=mixture_component,
                                                           initial_mean=initial_mean,
                                                           sigma=sigma,
                                                           mixture_weights=mixture_weights,
                                                           coordination=coordination,
                                                           expander_aux_mask_matrix=pt.constant(
                                                               expander_aux_mask_matrix),
                                                           aggregation_aux_mask_matrix=aggregator_aux_mask_matrix)
        real_logp = -5.081544319609303e+02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)

    def test_logp_without_self_dependency(self):
        # 3 subjects, 2 features and 2 time steps
        mixture_component = pt.constant(
            np.array([[[0.1, 0.3], [0.2, 0.4]], [[0.2, 0.4], [0.3, 0.5]], [[0.3, 0.5], [0.4, 0.6]]]))
        initial_mean = pt.constant(np.array([[0.3, 0.4], [0.4, 0.5], [0.6, 0.8]]))
        sigma = pt.constant(np.array([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]]))
        mixture_weights = pt.constant(np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]))
        coordination = pt.constant(np.array([0.1, 0.7]))

        expander_aux_mask_matrix = []
        aggregator_aux_mask_matrix = []
        for subject in range(3):
            expander_aux_mask_matrix.append(np.delete(np.eye(3), subject, axis=0))
            aux = np.zeros((3, 2))
            aux[subject] = 1
            aux = aux * mixture_weights[subject][None, :]
            aggregator_aux_mask_matrix.append(aux)

        expander_aux_mask_matrix = np.concatenate(expander_aux_mask_matrix, axis=0)
        aggregator_aux_mask_matrix = pt.concatenate(aggregator_aux_mask_matrix, axis=1)

        estimated_logp = mixture_logp_without_self_dependency(mixture_component=mixture_component,
                                                           initial_mean=initial_mean,
                                                           sigma=sigma,
                                                           mixture_weights=mixture_weights,
                                                           coordination=coordination,
                                                           expander_aux_mask_matrix=pt.constant(
                                                               expander_aux_mask_matrix),
                                                           aggregation_aux_mask_matrix=aggregator_aux_mask_matrix)
        real_logp = -4.257365244151745e+02

        self.assertAlmostEqual(estimated_logp.eval(), real_logp)
