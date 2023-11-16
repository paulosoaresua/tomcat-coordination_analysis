from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.module.latent_component.serial_latent_component import (
    SerialLatentComponent,
    SerialLatentComponentSamples
)
from coordination.module.module import ModuleSamples
from coordination.module.constants import (DEFAULT_TIME_SCALE_DENSITY,
                                           DEFAULT_SUBJECT_REPETITION_FLAG,
                                           DEFAULT_FIXED_SUBJECT_SEQUENCE_FLAG,
                                           DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_LATENT_DIMENSION_SIZE,
                                           DEFAULT_SELF_DEPENDENCY,
                                           DEFAULT_LATENT_MEAN_PARAM,
                                           DEFAULT_LATENT_SD_PARAM,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS)


class SerialMassSpringDamperLatentComponent(SerialLatentComponent):
    """
    This class represents a serial latent component with oscillatory dynamics and position coupling
    determined by coordination and no external force.
    """

    def __init__(self,
                 uuid: str,
                 pymc_model: pm.Model,
                 num_springs: int = DEFAULT_NUM_SUBJECTS,
                 spring_constant: np.ndarray,
                 mass: np.ndarray,
                 damping_coefficient: np.ndarray,
                 dt: float,
                 mean_mean_a0: np.ndarray = DEFAULT_LATENT_MEAN_PARAM,
                 sd_mean_a0: np.ndarray = DEFAULT_LATENT_SD_PARAM,
                 sd_sd_a: np.ndarray = DEFAULT_LATENT_SD_PARAM,
                 share_mean_a0_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
                 share_mean_a0_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
                 share_sd_a_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
                 share_sd_a_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
                 coordination_samples: Optional[ModuleSamples] = None,
                 coordination_random_variable: Optional[pm.Distribution] = None,
                 latent_component_random_variable: Optional[pm.Distribution] = None,
                 mean_a0_random_variable: Optional[pm.Distribution] = None,
                 sd_a_random_variable: Optional[pm.Distribution] = None,
                 sampling_time_scale_density: float = DEFAULT_TIME_SCALE_DENSITY,
                 allow_sampled_subject_repetition: bool = DEFAULT_SUBJECT_REPETITION_FLAG,
                 fix_sampled_subject_sequence: bool = DEFAULT_FIXED_SUBJECT_SEQUENCE_FLAG,
                 time_steps_in_coordination_scale: Optional[np.array] = None,
                 subject_indices: Optional[np.ndarray] = None,
                 prev_time_same_subject: Optional[np.ndarray] = None,
                 prev_time_diff_subject: Optional[np.ndarray] = None,
                 observed_values: Optional[TensorTypes] = None):
        """
        Creates a serial mass-spring-damper latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param num_springs: int = DEFAULT_NUM_SUBJECTS,
        @param spring_constant: spring constant per subject/spring used to calculate the fundamental
            matrix of the motion.
        @param mass: mass per subject/spring used to calculate the fundamental
            matrix of the motion.
        @param damping_coefficient: spring constant per subject used to calculate the fundamental
            matrix of the motion.
        @param dt: the size of each time step to calculate the fundamental matrix of the motion.
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
            latent component).
        @param sd_sd_a: std of the hyper-prior of sigma_a (std of the Gaussian random walk of
        the latent component).
        @param share_mean_a0_across_subjects: whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: whether to use the same mu_a0 for all dimensions.
        @param share_sd_a_across_subjects: whether to use the same sigma_a for all subjects.
        @param share_sd_a_across_dimensions: whether to use the same sigma_a for all dimensions.
        @param coordination_samples: coordination samples to be used in a call to draw_samples.
            This variable must be set before such a call.
        @param coordination_random_variable: coordination random variable to be used in a call to
            update_pymc_model. This variable must be set before such a call.
        @param latent_component_random_variable: latent component random variable to be used in a
            call to update_pymc_model. If not set, it will be created in such a call.
        @param mean_a0_random_variable: random variable to be used in a call to
            update_pymc_model. If not set, it will be created in such a call.
        @param sd_a_random_variable: random variable to be used in a call to
            update_pymc_model. If not set, it will be created in such a call.
        @param sampling_time_scale_density: a number between 0 and 1 indicating the frequency in
            which we have observations for the latent component. If 1, the latent component is
            observed at every time in the coordination timescale. If 0.5, in average, only half of
            the time. The final number of observations is not a deterministic function of the time
            density, as the density is used to define transition probabilities between a subjects
            and a non-existing subject.
        @param allow_sampled_subject_repetition: whether subsequent observations can come from the
            same subject.
        @param fix_sampled_subject_sequence: whether the sequence of subjects is fixed
            (0,1,2,...,0,1,2...) or randomized.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param subject_indices: array of numbers indicating which subject is associated to the
            latent component at every time step (e.g. the current speaker for a speech component).
            In serial components, only one user's latent component is observed at a time. This
            array indicates which user that is. This array contains no gaps. The size of the array
            is the number of observed latent component in time, i.e., latent component time
            indices with an associated subject.
        @param prev_time_same_subject: time indices indicating the previous observation of the
            latent component produced by the same subject at a given time. For instance, the last
            time when the current speaker talked. This variable must be set before a call to
            update_pymc_model.
        @param prev_time_diff_subject: similar to the above but it indicates the most recent time
            when the latent component was observed for a different subject. This variable must be
            set before a call to update_pymc_model.
        @param observed_values: observations for the serial latent component random variable. If
            a value is set, the variable is not latent anymore.
        """
        super().__init__(uuid=uuid,
                         pymc_model=pymc_model,
                         num_subjects=num_subjects,
                         dimension_size=2,
                         self_dependent=True,
                         mean_mean_a0=mean_mean_a0,
                         sd_mean_a0=sd_mean_a0,
                         sd_sd_a=sd_sd_a,
                         share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                         share_mean_a0_across_dimensions=share_mean_a0_across_dimensions,
                         share_sd_a_across_subjects=share_sd_a_across_subjects,
                         share_sd_a_across_dimensions=share_sd_a_across_dimensions,
                         dimension_names=["position", "speed"],
                         coordination_samples=coordination_samples,
                         coordination_random_variable=coordination_random_variable,
                         latent_component_random_variable=latent_component_random_variable,
                         mean_a0_random_variable=mean_a0_random_variable,
                         sd_a_random_variable=sd_a_random_variable,
                         time_steps_in_coordination_scale=time_steps_in_coordination_scale,
                         observed_values=observed_values,
                         sampling_time_scale_density=sampling_time_scale_density,
                         allow_sampled_subject_repetition=allow_sampled_subject_repetition,
                         fix_sampled_subject_sequence=fix_sampled_subject_sequence,
                         subject_indices=subject_indices,
                         prev_time_same_subject=prev_time_same_subject,
                         prev_time_diff_subject=prev_time_diff_subject)

        self.spring_constant = spring_constant
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.dt = dt

        # Fundamental matrices. One per subject.
        F = []
        for spring in range(num_springs):
            A = np.array([
                [0, 1],
                [-self.spring_constant[spring] / self.mass[spring],
                 -self.damping_coefficient[spring] / self.mass[spring]]
            ])
            F.append(expm(A * self.dt)[None, ...])

        self.F = np.concatenate(F, axis=0)

    def _draw_from_system_dynamics(self,
                                   coordination_sampled_series: np.ndarray,
                                   time_steps_in_coordination_scale: np.ndarray,
                                   subjects_in_time: np.ndarray,
                                   prev_time_same_subject: np.ndarray,
                                   prev_time_diff_subject: np.ndarray,
                                   mean_a0: np.ndarray,
                                   sd_a: np.ndarray) -> np.ndarray:
        """
        Draws values from a mass-spring-damper system dynamics using the fundamental matrices.

        @param coordination_sampled_series: sampled values of coordination series.
        @param time_steps_in_coordination_scale: an array of indices representing time steps in the
        coordination scale that match those of the component's scale.
        @param subjects_in_time: an array indicating which subject is responsible for a latent
        component observation at a time.
        @param prev_time_same_subject: an array containing indices to most recent previous times
        the component from the same subject was observed in the component's timescale.
        @param prev_time_diff_subject: an array containing indices to most recent previous times
        the component from a different subject was observed in the component's timescale.
        @param mean_a0: initial mean of the latent component.
        @param sd_a: standard deviation of the Gaussian transition distribution.

        @return: sampled values.
        """

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
                # Sample from prior. It is not only when t=0 because the first observation from a
                # subject can be later in the future. t=0 is the initial state of one of the
                # subjects only.
                mean = mean_a0[subject_idx_mean_a0]
                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dim_value)
            else:
                c = sampled_coordination[time_steps_in_coordination_scale[t]]

                prev_same = values[..., prev_time_same_subject[t]]

                mask_other = (prev_time_diff_subject[t] != -1).astype(int)
                prev_other = values[..., prev_time_diff_subject[t]]

                prev_subject = subjects_in_time[prev_time_diff_subject[t]]

                dt_diff = np.maximum(t - prev_time_diff_subject[t], 1)
                dt_same = np.maximum(t - prev_time_same_subject[t], 1)

                # Bring states to the present with the fundamental matrix and blend these states
                # using coordination.
                F_diff = np.linalg.matrix_power(self.F[prev_subject], dt_diff)
                F_same = np.linalg.matrix_power(self.F[subjects_in_time[t]], dt_same)
                prev_other = np.dot(F_diff, prev_other)
                prev_same = np.dot(F_same, prev_same)

                blended_mean = (prev_other - prev_same) * c * mask_other + prev_same

                # Do not blend speed.
                blended_mean[1] = prev_same[1]

                values[:, t] = norm(loc=blended_mean, scale=sd).rvs()

        return values

    def _get_extra_logp_params(self):
        """
        Gets fundamental matrices per time step.
        """
        Fs_diff = []
        Fs_same = []
        for t in range(len(subjects_in_time)):
            dt_diff = np.maximum(t - prev_time_diff_subject[t], 1)
            dt_same = np.maximum(t - prev_time_same_subject[t], 1)

            curr_subject = subjects_in_time[t]
            diff_subject = subjects_in_time[prev_time_diff_subject[t]]
            Fs_diff.append(np.linalg.matrix_power(self.F[diff_subject], dt_diff))
            Fs_same.append(np.linalg.matrix_power(self.F[curr_subject], dt_same))

        # We need the influencer and influencee's transformations because we first bring the
        # states to the current time and then we blend them.
        Fs_diff_reshaped = np.array(Fs_diff).reshape((len(subjects_in_time), 4))
        Fs_same_reshaped = np.array(Fs_same).reshape((len(subjects_in_time), 4))

        return Fs_diff_reshaped, Fs_same_reshaped

    def _get_logp_fn(self):
        """
        Gets a reference to a log_prob function.
        """
        return log_prob

    def _get_random_fn(self):
        """
        Gets a reference to a random function for prior predictive checks.
        Disabled in this module as it is only used for synthetic data generation.
        """
        return None


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################

def log_prob(sample: ptt.TensorVariable,
             initial_mean: ptt.TensorVariable,
             sigma: ptt.TensorVariable,
             coordination: ptt.TensorVariable,
             prev_time_same_subject: ptt.TensorConstant,
             prev_time_diff_subject: ptt.TensorConstant,
             prev_same_subject_mask: ptt.TensorConstant,
             prev_diff_subject_mask: ptt.TensorConstant,
             self_dependent: ptt.TensorConstant,
             Fs_diff: ptt.TensorConstant,
             Fs_same: ptt.TensorConstant) -> float:
    """
    Computes the log-probability function of a sample.

    @param sample: (dimension x time) a single samples series.
    @param initial_mean: (dimension x time) a series of mean at t0. At each time the mean is associated with the
        subject at that time. The initial mean is only used the first time the user speaks, but we
        repeat the values here over time for uniform vector operations (e.g., we can multiply this
        with other tensors) and we fix the behavior with mask tensors.
    @param sigma: (dimension x time) a series of standard deviations. At each time the standard deviation is
        associated with the subject at that time.
    @param coordination: (time) a series of coordination values. Axis (time).
    @param prev_time_same_subject: (time) a series of time steps pointing to the previous time step
        associated with the same subject. For instance, prev_time_same_subject[t] points to the
        most recent time step where the subject at time t had an observation. If there's no such a
        time, prev_time_same_subject[t] will be -1. Axes (time).
    @param prev_time_diff_subject: (time)  a series of time steps pointing to the previous time step
        associated with a different subject. For instance, prev_time_diff_subject[t] points to the
        most recent time step where a different subject than the one at time t had an observation.
        If there's no such a time, prev_time_diff_subject[t] will be -1.
    @param prev_same_subject_mask: (time) a binary mask with 0 whenever prev_time_same_subject
        is -1.
    @param prev_diff_subject_mask: (time) a binary mask with 0 whenever prev_time_diff_subject
        is -1.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
        Not used by this implementation as self_dependency is fixed to True.
    @param Fs_diff: (time x 4) fundamental matrix to be applied on the time series of previous
        values from a different subject over time.
    @param Fs_same: (time x 4) fundamental matrix to be applied on the time series of previous
        values from the same subject over time.
    @return: log-probability of the sample.
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

    # This function can only receive tensors up to 2 dimensions because 'sample' has 2 dimensions.
    # This is a limitation of PyMC 5.0.2. So, we reshape F before passing to this function and here
    # we reshape it back to its original 3 dimensions.
    Fs_diff_reshaped = Fs_diff.reshape((Fs_diff.shape[0], 2, 2))
    Fs_same_reshaped = Fs_same.reshape((Fs_same.shape[0], 2, 2))

    # We transform the sample using the fundamental matrix so that we learn to generate samples
    # with the underlying system dynamics. If we just compare a sample with the blended_mean, we
    # are assuming the samples follow a random gaussian walk. Since we know the system dynamics,
    # we can add that to the logp such that the samples are effectively coming from the component's
    # posterior.
    prev_other_transformed = ptt.batched_tensordot(Fs_diff_reshaped, prev_other.T,
                                                   axes=[(2,), (1,)]).T
    prev_same_transformed = ptt.batched_tensordot(Fs_same_reshaped, prev_same.T,
                                                  axes=[(2,), (1,)]).T

    blended_mean = prev_other_transformed * c * mask_other + (1 - c * mask_other) * (
            prev_same_transformed * mask_same + (1 - mask_same) * initial_mean)

    # We don't blend speed
    POSITION_COL = ptt.as_tensor(np.array([[1], [0]]))
    SPEED_COL = ptt.as_tensor(np.array([[0], [1]]))
    blended_mean = blended_mean * POSITION_COL + (
            prev_same * mask_same + (1 - mask_same) * initial_mean) * SPEED_COL

    total_logp = pm.logp(pm.Normal.dist(mu=blended_mean, sigma=sigma, shape=blended_mean.shape),
                         sample).sum()

    return total_logp
