from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.module.serial_component import SerialComponent


def logp(sample: Any,
         initial_mean: Any,
         sigma: Any,
         coordination: Any,
         prev_time_same_subject: ptt.TensorConstant,
         prev_time_diff_subject: ptt.TensorConstant,
         prev_same_subject_mask: Any,
         prev_diff_subject_mask: Any,
         self_dependent: ptt.TensorConstant,
         F_inv: ptt.TensorConstant):
    """
    This function computes the log-probability of a serial component. We use the following definition in the
    comments below:

    d: number of dimensions/features of the component
    T: number of time steps in the component's scale
    """

    # We use 'prev_time_diff_subject' as meta-data to get the values from partners of the subjects in each time
    # step. We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the sample component is one.
    prev_other = sample[..., prev_time_diff_subject].reshape(sample.shape)  # d x T

    # We use this binary mask to zero out entries with no observations from partners.
    mask_other = prev_diff_subject_mask[None, :]  # 1 x T

    c = coordination[None, :]  # 1 x T

    # The component's value for a subject depends on its previous value for the same subject.
    prev_same = sample[..., prev_time_same_subject].reshape(sample.shape)  # d x T

    # We use this binary mask to zero out entries with no previous observations from the subjects. We use this
    # to determine the time steps that belong to the initial values of the component. Each subject will have their
    # initial value in a different time step hence we cannot just use t=0.
    mask_same = prev_same_subject_mask[None, :]  # 1 x t

    blended_mean = prev_other * c * mask_other + (1 - c * mask_other) * (
            prev_same * mask_same + (1 - mask_same) * initial_mean)

    # This function can only receive tensors up to 2 dimensions because 'sample' has 2 dimensions.
    # This is a limitation of PyMC 5.0.2. So, we reshape F_inv before passing to this function and here we reshape
    # it back to its original 3 dimensions.
    F_inv_reshaped = F_inv.reshape((F_inv.shape[0], 2, 2))

    # We transform the sample using backward dynamics so that we learn to generate samples with the underlying system
    # dynamics. If we just compare a sample with the blended_mean, we are assuming the samples follow a random gaussian
    # walk. Since we know the system dynamics, we can add that to the logp such that the samples are effectively
    # coming from the component's posterior.
    sample_transformed = ptt.batched_tensordot(F_inv_reshaped, sample.T,
                                               axes=[(2,), (1,)]).T * mask_same + sample * (1 - mask_same)

    total_logp = pm.logp(pm.Normal.dist(mu=blended_mean, sigma=sigma, shape=blended_mean.shape),
                         sample_transformed).sum()

    return total_logp


class SerialMassSpringDamperComponent(SerialComponent):

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

    def _draw_from_system_dynamics(self,
                                   time_steps_in_coordination_scale: np.ndarray,
                                   sampled_coordination: np.ndarray,
                                   subjects_in_time: np.ndarray,
                                   prev_time_same_subject: np.ndarray,
                                   prev_time_diff_subject: np.ndarray,
                                   mean_a0: np.ndarray,
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
                # Sample from prior. It is not only when t=0 because the first utterance of a speaker can be later in
                # the future. t=0 is the initial state of one of the subjects only.
                mean = mean_a0[subject_idx_mean_a0]
                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dim_value)
            else:
                c = sampled_coordination[time_steps_in_coordination_scale[t]]

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous value of another subject,
                    # or the previous value of the same subject.
                    prev_same = values[..., prev_time_same_subject[t]]
                else:
                    # When there's no self dependency, the component either depends on the previous value of another subject,
                    # or it is samples around a fixed mean.
                    prev_same = mean_a0[subject_idx_mean_a0]

                mask_other = (prev_time_diff_subject[t] != -1).astype(int)
                prev_other = values[..., prev_time_diff_subject[t]]

                blended_mean = (prev_other - prev_same) * c * mask_other + prev_same

                # Use the fundamental matrix to generate samples from a Hookean spring system.
                blended_mean_transformed = np.dot(self.F[subjects_in_time[t]], blended_mean)

                # We don't blend velocity.
                blended_mean_transformed[1] = np.dot(self.F[subjects_in_time[t]], prev_same)[1]

                values[:, t] = norm(loc=blended_mean_transformed, scale=sd).rvs()

        return values

    def _get_extra_logp_params(self, subjects_in_time: np.ndarray):
        F_inv_reshaped = self.F_inv[subjects_in_time].reshape((len(subjects_in_time), 4))
        return F_inv_reshaped,

    def _get_logp_fn(self):
        return logp

    def _get_random_fn(self):
        return None
