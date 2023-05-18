from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.component.mixture_component import MixtureComponent


def logp(mixture_component: Any,
         initial_mean: Any,
         sigma: Any,
         mixture_weights: Any,
         coordination: Any,
         expander_aux_mask_matrix: ptt.TensorConstant,
         self_dependent: ptt.TensorConstant,
         F_inv: ptt.TensorConstant,
         aggregation_aux_mask_matrix: ptt.TensorConstant):
    num_subjects = mixture_component.shape[0]
    num_features = mixture_component.shape[1]

    # Log probability due to the initial time step in the component's scale.
    total_logp = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(num_subjects, num_features)),
                         mixture_component[..., 0]).sum()

    # Computes the movement backwards according to the system dynamics, this will make the model learn a latent
    # representation that respects the system dynamics.
    mixture_component_previous_time = mixture_component

    # D contains the values from other individuals for each individual
    D = ptt.tensordot(expander_aux_mask_matrix, mixture_component_previous_time, axes=(1, 0))[..., :-1]  # s * (s-1) x d x t
    #
    # # Get previous values for each time step according to an index matrix. Discard the first time step as
    # # there's no previous values in the first time step.
    # D = ptt.take_along_axis(D, prev_time_diff_subject[:, None, :], axis=2)[..., 1:]


    # Previous values from every subject
    P = mixture_component[..., :-1]

    # Previous values from the same subjects
    S_extended = ptt.repeat(P, repeats=(num_subjects - 1), axis=0)


    # The mask will zero out dependencies on D if we have shifts caused by latent lags. In that case, we cannot infer
    # coordination if the values do not exist on all the subjects because of gaps introduced by the shift. So we can
    # only infer the next value of the latent value from its previous one on the same subject,
    C = coordination[None, None, 1:]  # 1 x 1 x t-1
    D = ptt.tensordot(aggregation_aux_mask_matrix, D / (num_subjects - 1), axes=(1, 0))
    mean = (D - P) * C + P

    # We transform points using the system dynamics so that samples that follow such dynamics are accepted
    # with higher probability.
    mixture_component_transformed = ptt.batched_tensordot(F_inv, mixture_component, axes=[(1,), (1,)])
    # Current values from each subject. We extend S and point such that they match the dimensions of D and S.
    # point_extended = ptt.repeat(mixture_component_transformed[..., 1:], repeats=(num_subjects - 1), axis=0)

    # sd = ptt.repeat(sigma, repeats=(num_subjects - 1), axis=0)[:, :, None]
    sd = sigma[:, :, None]

    # blended_mean = ptt.tensordot(aggregation_aux_mask_matrix, mean, axes=(1, 0))
    total_logp += pm.logp(pm.Normal.dist(mu=mean, sigma=sd, shape=mean.shape),
                          mixture_component_transformed[..., 1:]).sum()

    # logp_extended = pm.logp(pm.Normal.dist(mu=mean, sigma=sd, shape=D.shape), point_extended)
    # logp_tmp = logp_extended.reshape((num_subjects, num_subjects - 1, num_features, logp_extended.shape[-1]))
    # total_logp += pm.math.logsumexp(logp_tmp + pm.math.log(mixture_weights[:, :, None, None]), axis=1).sum()

    return total_logp


class MixtureMassSpringDamperComponent(MixtureComponent):

    def __init__(self, uuid: str,
                 num_springs: int,
                 spring_constant: np.ndarray,  # one per spring
                 mass: np.ndarray,  # one per spring
                 damping_coefficient: np.ndarray,  # one per spring
                 dt: float,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 a_mixture_weights: np.ndarray,
                 share_mean_a0_across_springs: bool,
                 share_mean_a0_across_features: bool,
                 share_sd_aa_across_springs: bool,
                 share_sd_aa_across_features: bool):
        """
        Generates a time series of latent states formed by position and velocity in a mass-spring-damper system. We do
        not consider external force in this implementation but it can be easily added if necessary.
        """
        super().__init__(uuid=uuid,
                         num_subjects=num_springs,
                         dim_value=2,  # 2 dimensions: position and velocity
                         self_dependent=self_dependent,
                         mean_mean_a0=mean_mean_a0,
                         sd_mean_a0=sd_mean_a0,
                         sd_sd_aa=sd_sd_aa,
                         a_mixture_weights=a_mixture_weights,
                         share_mean_a0_across_subjects=share_mean_a0_across_springs,
                         share_mean_a0_across_features=share_mean_a0_across_features,
                         share_sd_aa_across_subjects=share_sd_aa_across_springs,
                         share_sd_aa_across_features=share_sd_aa_across_features)

        # We assume the spring_constants are known but this can be changed to have them as latent variables if needed.
        assert spring_constant.ndim == 1 and len(spring_constant) == num_springs

        self.spring_constant = spring_constant
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.dt = dt  # size of the time step

        # Systems dynamics matrix
        F = []
        F_inv = []
        for spring in range(num_springs):
            A = np.array([
                [0, 1],
                [-self.spring_constant[spring] / self.mass[spring],
                 -self.damping_coefficient[spring] / self.mass[spring]]
            ])
            F.append(expm(A * self.dt)[None, ...])  # Fundamental matrix
            F_inv.append(expm(-A * self.dt)[None, ...])  # Fundamental matrix inverse to estimate backward dynamics

        self.F = np.concatenate(F, axis=0)
        self.F_inv = np.concatenate(F_inv, axis=0)

    def _draw_from_system_dynamics(self, time_steps_in_coordination_scale: np.ndarray, sampled_coordination: np.ndarray,
                                   sampled_influencers: np.ndarray, mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:
        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)

        SUM = np.ones((self.num_subjects, self.num_subjects)) - np.eye(self.num_subjects)

        values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps))

        for t in range(num_time_steps):
            if t == 0:
                values[..., t] = norm(loc=mean_a0, scale=sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                C = sampled_coordination[:, time_steps_in_coordination_scale[t]][:, None]

                P = values[..., t - 1]

                D = P

                D = np.einsum("ij,kjl->kil", SUM, D)

                if self.self_dependent:
                    S = P
                else:
                    S = mean_a0

                blended_state = (1 - C) * S + (D / (self.num_subjects - 1)) * C

                mean = np.einsum("ijk,lij->lik", self.F, blended_state)

                # Add some noise
                values[..., t] = norm(loc=mean, scale=sd_aa).rvs()

        return values

    def _get_extra_logp_params(self):
        aggregator_aux_mask_matrix = []
        for subject in range(self.num_subjects):
            aux = np.zeros((self.num_subjects, self.num_subjects - 1))
            aux[subject] = 1
            aggregator_aux_mask_matrix.append(aux)

        aggregator_aux_mask_matrix = ptt.concatenate(aggregator_aux_mask_matrix, axis=1)

        return self.F_inv, aggregator_aux_mask_matrix

    def _get_logp_fn(self):
        return logp

    def _get_random_fn(self):
        return None
