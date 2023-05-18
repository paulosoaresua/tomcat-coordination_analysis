from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.component.serialized_component import SerializedComponent


def logp(serialized_component: Any,
         initial_mean: Any,
         sigma: Any,
         coordination: Any,
         prev_time_same_subject: ptt.TensorConstant,
         prev_time_diff_subject: ptt.TensorConstant,
         prev_same_subject_mask: Any,
         prev_diff_subject_mask: Any,
         self_dependent: ptt.TensorConstant,
         F_inv: ptt.TensorConstant):
    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    C = coordination[None, :]  # 1 x t
    DM = prev_diff_subject_mask[None, :]  # 1 x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    # If there's no previous observation from the same subject, we use the initial mean.
    S = serialized_component[..., prev_time_same_subject].reshape(serialized_component.shape)  # d x t

    SM = prev_same_subject_mask[None, :]  # 1 x t
    mean = D * C * DM + (1 - C * DM) * (S * SM + (1 - SM) * initial_mean)

    # This function can only receive tensors up to 2 dimensions because serialized_component has 2 dimensions.
    # This is a limitation of PyMC 5.0.2. So, we reshape F_inv before passing to this function and here we reshape
    # it back to it's original 3 dimensions.
    F_inv_reshaped = F_inv.reshape((F_inv.shape[0], 2, 2))

    # We transform points using the system dynamics so that samples that follow such dynamics are accepted
    # with higher probability. The batch dimension will be time, that's why we transpose serialized_component.
    serialized_component_transformed = ptt.batched_tensordot(F_inv_reshaped, serialized_component.T,
                                                             axes=[(2,), (1,)]).T * SM + serialized_component * (1 - SM)

    # serialized_component_transformed = serialized_component

    total_logp = pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=D.shape), serialized_component_transformed).sum()

    return total_logp


class SerializedMassSpringDamperComponent(SerializedComponent):

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
                                   subjects_in_time: np.ndarray, prev_time_same_subject: np.ndarray,
                                   prev_time_diff_subject: np.ndarray, mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:

        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((self.dim_value, num_time_steps))

        for t in range(num_time_steps):
            if self.share_mean_a0_across_subjects:
                subject_idx_mean_a0 = 0
            else:
                subject_idx_mean_a0 = subjects_in_time[t]

            if self.share_sd_aa_across_subjects:
                subject_idx_sd_aa = 0
            else:
                subject_idx_sd_aa = subjects_in_time[t]

            sd = sd_aa[subject_idx_sd_aa]

            if prev_time_same_subject[t] < 0:
                # It is not only when t == 0 because the first utterance of a speaker can be later in the future.
                # t_0 is the initial utterance of one of the subjects only.
                mean = mean_a0[subject_idx_mean_a0]

                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dim_value)
            else:
                C = sampled_coordination[time_steps_in_coordination_scale[t]]

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous value of another subject,
                    # or the previous value of the same subject.
                    S = values[..., prev_time_same_subject[t]]
                else:
                    # When there's no self dependency, the component either depends on the previous value of another subject,
                    # or it is samples around a fixed mean.
                    S = mean_a0[subject_idx_mean_a0]

                prev_diff_mask = (prev_time_diff_subject[t] != -1).astype(int)
                D = values[..., prev_time_diff_subject[t]]

                blended_state = (D - S) * C * prev_diff_mask + S

                mean = np.dot(self.F[subjects_in_time[t]], blended_state[:, None]).T
                # mean = blended_state

                values[:, t] = norm(loc=mean, scale=sd).rvs()

        return values

    def _get_extra_logp_params(self, subjects_in_time: np.ndarray):
        F_inv_reshaped = self.F_inv[subjects_in_time].reshape((len(subjects_in_time), 4))
        return F_inv_reshaped,

    def _get_logp_fn(self):
        return logp

    def _get_random_fn(self):
        # TODO - implement this for prior predictive checks.
        return None
