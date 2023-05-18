from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.component.mixture_component import MixtureComponent


def logp(sample: Any,
         initial_mean: Any,
         sigma: Any,
         coordination: Any,
         self_dependent: ptt.TensorConstant,
         F_inv: ptt.TensorConstant):
    """
    This function computes the log-probability of a non-serial mass-spring-damper component. We use the following
    definition in the comments below:

    s: number of subjects
    d: number of dimensions/features of the component
    T: number of time steps in the component's scale
    """

    N = sample.shape[0]
    D = sample.shape[1]

    # logp at the initial time step
    total_logp = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(N, D)), sample[..., 0]).sum()

    # Contains the sum of previous values of other subjects for each subject scaled by 1/(s-1).
    # We discard the last value as that is not a previous value of any other.
    sum_matrix_others = (ptt.ones((N, N)) - ptt.eye(N)) / (N - 1)
    prev_others = ptt.tensordot(sum_matrix_others, sample, axes=(1, 0))[..., :-1]

    # The component's value for a subject depends on its previous value for the same subject.
    prev_same = sample[..., :-1]

    # Coordination does not affect the component in the first time step because the subjects have no previous
    # dependencies at that time.
    c = coordination[None, None, 1:]  # 1 x 1 x t-1

    blended_mean = (prev_others - prev_same) * c + prev_same

    # Match the dimensions of the standard deviation with that of the blended mean
    sd = sigma[:, :, None]

    # We transform the sample using backward dynamics so that we learn to generate samples with the underlying system
    # dynamics. If we just compare a sample with the blended_mean, we are assuming the samples follow a random gaussian
    # walk. Since we know the system dynamics, we can add that to the logp such that the samples are effectively
    # coming from the component's posterior.
    sample_transformed = ptt.batched_tensordot(F_inv, sample, axes=[(1,), (1,)])

    # Index samples starting from the second index (i = 1) so that we can effectively compare current values against
    # previous ones (prev_others and prev_same).
    total_logp += pm.logp(pm.Normal.dist(mu=blended_mean, sigma=sd, shape=blended_mean.shape),
                          sample_transformed[..., 1:]).sum()

    return total_logp


class MixtureMassSpringDamperComponent(MixtureComponent):
    """
    This class models a non-serial latent mass-spring-damper component which individual spring's dynamics influence
    that of the other springs as controlled by coordination.
    """

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
                                   mean_a0: np.ndarray, sd_aa: np.ndarray) -> np.ndarray:
        """
        In this function we use the following notation in the comments:

        n: number of series/samples (first dimension of coordination)
        s: number of subjects
        d: number of features
        """
        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps))

        N = self.num_subjects
        sum_matrix_others = (np.ones((N, N)) - np.eye(N)) / (N - 1)

        for t in range(num_time_steps):
            if t == 0:
                values[..., t] = norm(loc=mean_a0, scale=sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                c = sampled_coordination[:, time_steps_in_coordination_scale[t]][:, None, None]  # n x 1

                prev_others = np.einsum("jk,ikl->ijl", sum_matrix_others, values[..., t - 1])  # s x d
                prev_same = values[..., t - 1]  # s x d

                blended_mean = (prev_others - prev_same) * c + prev_same  # s x d

                # Use the fundamental matrix to generate samples from a Hookean spring system.
                blended_mean_transformed = np.einsum("ijk,lij->lik", self.F, blended_mean)

                values[..., t] = norm(loc=blended_mean_transformed, scale=sd_aa).rvs()

        return values

    def _get_extra_logp_params(self):
        return self.F_inv,

    def _get_logp_fn(self):
        return logp

    def _get_random_fn(self):
        return None
