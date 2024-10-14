from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.module.constants import (DEFAULT_FIXED_SUBJECT_SEQUENCE_FLAG,
                                           DEFAULT_LATENT_MEAN_PARAM,
                                           DEFAULT_LATENT_SD_PARAM,
                                           DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_SAMPLING_TIME_SCALE_DENSITY,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS,
                                           DEFAULT_SUBJECT_REPETITION_FLAG)
from coordination.module.latent_component.serial_gaussian_latent_component import \
    SerialGaussianLatentComponent
from coordination.module.module import ModuleSamples


class Serial2DGaussianLatentComponent(SerialGaussianLatentComponent):
    """
    This class represents a 2D serial latent component with position and speed. It encodes the
    notion that a change in speed from a component from one subject drives a change in speed
    of the same component in another subject through blending of the speed dimension when
    there's coordination. This encodes the following idea: if there's a raise in A's component's
    speed and B a subsequent raise in B's component's speed, this is an indication of coordination
    regardless of the absolute value of A and B's component amplitudes (positions).
    """

    def __init__(
        self,
        uuid: str,
        pymc_model: pm.Model,
        num_subjects: int = DEFAULT_NUM_SUBJECTS,
        self_dependent: bool = True,
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
        sampling_time_scale_density: float = DEFAULT_SAMPLING_TIME_SCALE_DENSITY,
        allow_sampled_subject_repetition: bool = DEFAULT_SUBJECT_REPETITION_FLAG,
        fix_sampled_subject_sequence: bool = DEFAULT_FIXED_SUBJECT_SEQUENCE_FLAG,
        time_steps_in_coordination_scale: Optional[np.array] = None,
        subject_indices: Optional[np.ndarray] = None,
        prev_time_same_subject: Optional[np.ndarray] = None,
        prev_time_diff_subject: Optional[np.ndarray] = None,
        observed_values: Optional[TensorTypes] = None,
        mean_a0: Optional[Union[float, np.ndarray]] = None,
        sd_a: Optional[Union[float, np.ndarray]] = None,
        initial_samples: Optional[np.ndarray] = None,
        asymmetric_coordination: bool = False,
    ):
        """
        Creates a serial 2D Gaussian latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param self_dependent: whether a state at time t depends on itself at time t-1 or it
            depends on a fixed value given by mean_a0.
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
        @param mean_a0: initial value of the latent component. It needs to be given for sampling
            but not for inference if it needs to be inferred. If not provided now, it can be set
            later via the module parameters variable.
        @param sd_a: standard deviation of the latent component Gaussian random walk. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
        @param initial_samples: samples from the posterior to use during a call to draw_samples.
            This is useful to do predictive checks by sampling data in the future.
        @param asymmetric_coordination: whether coordination is asymmetric or not. If asymmetric,
            the value of a component for one subject depends on the negative of the combination of
            the others.
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=2,  # position and speed
            self_dependent=self_dependent,
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
            prev_time_diff_subject=prev_time_diff_subject,
            mean_a0=mean_a0,
            sd_a=sd_a,
            initial_samples=initial_samples,
            asymmetric_coordination=asymmetric_coordination,
        )

    def _draw_from_system_dynamics(
        self,
        coordination_sampled_series: np.ndarray,
        time_steps_in_coordination_scale: np.ndarray,
        subjects_in_time: np.ndarray,
        prev_time_same_subject: np.ndarray,
        prev_time_diff_subject: np.ndarray,
        mean_a0: np.ndarray,
        sd_a: np.ndarray,
        init_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draws values with the following updating equations for the state of the component at time
        t:

        P_a(t) = P_a(t-1) + V_a(t-1)dt
        S_a(t) = (1 - C(t))*S_a(t-1) + c(t)*S_b(t-1)

        Where, "P" is position, "S" speed, "a" and "b" are the influencee and influencer
        respectively.  We set dt to be 1, meaning it jumps one time step in the component's scale
        instead of "n" time steps in the coordination scale when there are gaps.

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
        values = np.zeros((self.dimension_size, num_time_steps))
        t0 = 0 if init_values is None else init_values.shape[-1]
        if init_values is not None:
            values[..., :t0] = init_values

        for t in range(t0, num_time_steps):
            if self.share_mean_a0_across_subjects:
                subject_idx_mean_a0 = 0
            else:
                subject_idx_mean_a0 = subjects_in_time[t]

            if self.share_sd_a_across_subjects:
                subject_idx_sd_a = 0
            else:
                subject_idx_sd_a = subjects_in_time[t]

            sd = sd_a[subject_idx_sd_a]

            if prev_time_same_subject[t] < 0:
                # Sample from prior. It is not only when t=0 because the first observation from a
                # subject can be later in the future. t=0 is the initial state of one of the
                # subjects only.
                mean = mean_a0[subject_idx_mean_a0]
                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dimension_size)
            else:
                c = coordination_sampled_series[time_steps_in_coordination_scale[t]]
                c_mask = -1 if self.asymmetric_coordination else 1

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous
                    # value of another subject or the previous value of the same subject.
                    prev_same = values[..., prev_time_same_subject[t]]
                else:
                    # When there's no self dependency, the component either depends on the previous
                    # value of another subject or a fixed value (the subject's prior).
                    prev_same = mean_a0[subject_idx_mean_a0]

                prev_other = values[..., prev_time_diff_subject[t]] * c_mask

                # The matrix F multiplied by the state of a component "a" at time t - 1
                # ([P(t-1), S(t-1)]) gives us:
                #
                # P_a(t) = P_a(t-1) + S_a(t-1)dt
                # S_a(t) = (1 - C(t))*S_a(t-1)
                #
                # Then we just need to sum with [0, c(t)*S_b(t-1)] to obtain the updated state of
                # the component. Which can be accomplished with U*[P_b(t-1), S_b(t-1)]
                dt_diff = 1
                F = np.array([[1, dt_diff], [0, 1 - c]])
                U = np.array(
                    [
                        [0, 0],  # position of "b" does not influence position of "a"
                        [
                            0,
                            c,
                        ],  # speed of "b" influences the speed of "a" when there's coordination.
                    ]
                )

                blended_mean = np.dot(F, prev_same) + np.dot(U, prev_other)
                values[:, t] = norm(loc=blended_mean, scale=sd).rvs()

        return values

    def _get_extra_logp_params(self) -> Tuple[Union[TensorTypes, pm.Distribution], ...]:
        """
        Gets fundamental matrices per time step.
        """
        return ()

    def _get_log_prob_fn(self) -> Callable:
        """
        Gets a reference to a log_prob function.
        """
        return log_prob

    def _get_random_fn(self) -> Callable:
        """
        Gets a reference to a random function for prior predictive checks.
        Disabled in this module as it is only used for synthetic data generation.
        """
        return random


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################


def log_prob(
    sample: ptt.TensorVariable,
    initial_mean: ptt.TensorVariable,
    sigma: ptt.TensorVariable,
    coordination: ptt.TensorVariable,
    prev_time_same_subject: ptt.TensorConstant,
    prev_time_diff_subject: ptt.TensorConstant,
    prev_same_subject_mask: ptt.TensorConstant,
    prev_diff_subject_mask: ptt.TensorConstant,
    self_dependent: ptt.TensorConstant,
    symmetry_mask: ptt.TensorConstant,
) -> float:
    """
    Computes the log-probability function of a sample.

    Legend:
    D: number of dimensions
    T: number of time steps

    @param sample: (dimension x time) a single samples series.
    @param initial_mean: (dimension x time) a series of mean at t0. At each time the mean is
        associated with the subject at that time. The initial mean is only used the first time the
        user speaks, but we repeat the values here over time for uniform vector operations (e.g.,
        we can multiply this with other tensors) and we fix the behavior with mask tensors.
    @param sigma: (dimension x time) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values. Axis (time).
    @param prev_time_same_subject: (time) a series of time steps pointing to the previous time step
        associated with the same subject. For instance, prev_time_same_subject[t] points to the
        most recent time step where the subject at time t had an observation. If there's no such a
        time, prev_time_same_subject[t] will be -1. Axes (time).
    @param prev_time_diff_subject: (time)  a series of time steps pointing to the previous time
        step associated with a different subject. For instance, prev_time_diff_subject[t] points
        to the most recent time step where a different subject than the one at time t had an
        observation. If there's no such a time, prev_time_diff_subject[t] will be -1.
    @param prev_same_subject_mask: (time) a binary mask with 0 whenever prev_time_same_subject
        is -1.
    @param prev_diff_subject_mask: (time) a binary mask with 0 whenever prev_time_diff_subject
        is -1.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
        Not used by this implementation as self_dependency is fixed to True.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise.
    @return: log-probability of the sample.
    """

    # We use 'prev_time_diff_subject' as meta-data to get the values from partners of the subjects
    # in each time step. We reshape to guarantee we don't create dimensions with unknown size in
    # case the first dimension of the sample component is one.
    prev_other = (
        sample[..., prev_time_diff_subject].reshape(sample.shape) * symmetry_mask
    )  # D x T

    # The component's value for a subject depends on its previous value for the same subject.
    if self_dependent.eval():
        # The component's value for a subject depends on previous value of the same subject.
        prev_same = sample[..., prev_time_same_subject].reshape(sample.shape)  # (D x T)
    else:
        # The component's value for a subject doesn't depend on previous value of the same subject.
        # At every time step, the value from other subjects is blended with a fixed value given
        # by the component's initial means associated with the subjects over time.
        # (D x T)
        prev_same = initial_mean

    # We use this binary mask to zero out entries with no previous observations from the subjects.
    # We use this to determine the time steps that belong to the initial values of the component.
    # Each subject will have their initial value in a different time step hence we cannot just use
    # t=0.
    mask_same = prev_same_subject_mask[None, :]  # 1 x t

    prev_same = prev_same * mask_same + (1 - mask_same) * initial_mean

    c = coordination * prev_diff_subject_mask

    F = (
        ptt.as_tensor([[1.0, 1.0], [0.0, 1.0]])
        - ptt.as_tensor([[0.0, 0.0], [0.0, 1.0]]) * c[:, None, None]
    )
    U = ptt.as_tensor([[0.0, 0.0], [0.0, 1.0]]) * c[:, None, None]

    # We transform the sample using the fundamental matrix so that we learn to generate samples
    # with the underlying system dynamics. If we just compare a sample with the blended_mean, we
    # are assuming the samples follow a random gaussian walk. Since we know the system dynamics,
    # we can add that to the log-probability such that the samples are effectively coming from the
    # component's posterior.
    prev_same_transformed = ptt.batched_tensordot(F, prev_same.T, axes=[(2,), (1,)]).T
    prev_other_transformed = ptt.batched_tensordot(U, prev_other.T, axes=[(2,), (1,)]).T

    blended_mean = prev_other_transformed + prev_same_transformed

    total_logp = pm.logp(
        pm.Normal.dist(mu=blended_mean, sigma=sigma, shape=blended_mean.shape), sample
    ).sum()

    return total_logp


def random(
    initial_mean: np.ndarray,
    sigma: np.ndarray,
    coordination: np.ndarray,
    prev_time_same_subject: np.ndarray,
    prev_time_diff_subject: np.ndarray,
    prev_same_subject_mask: np.ndarray,
    prev_diff_subject_mask: np.ndarray,
    self_dependent: bool,
    symmetry_mask: int,
    rng: Optional[np.random.Generator] = None,
    size: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """
    Generates samples from of a serial latent component for prior predictive checks.

    Legend:
    D: number of dimensions
    T: number of time steps

    @param initial_mean: (dimension x time) a series of mean at t0. At each time the mean is
        associated with the subject at that time. The initial mean is only used the first time the
        user speaks, but we repeat the values here over time for uniform vector operations (e.g.,
        we can multiply this with other tensors) and we fix the behavior with mask tensors.
    @param sigma: (dimension x time) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param prev_time_same_subject: (time) a series of time steps pointing to the previous time step
        associated with the same subject. For instance, prev_time_same_subject[t] points to the
        most recent time step where the subject at time t had an observation. If there's no such a
        time, prev_time_same_subject[t] will be -1.
    @param prev_time_diff_subject: (time)  a series of time steps pointing to the previous time
        step associated with a different subject. For instance, prev_time_diff_subject[t] points
        to the most recent time step where a different subject than the one at time t had an
        observation. If there's no such a time, prev_time_diff_subject[t] will be -1.
    @param prev_same_subject_mask: (time) a binary mask with 0 whenever prev_time_same_subject
        is -1.
    @param prev_diff_subject_mask: (time) a binary mask with 0 whenever prev_time_diff_subject
        is -1.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise.
    @param rng: random number generator.
    @param size: size of the sample.

    @return: a serial latent component sample.
    """

    # TODO: Unify this with the class sampling method.

    T = coordination.shape[-1]

    sample = np.zeros(size)

    mean_0 = initial_mean if initial_mean.ndim == 1 else initial_mean[..., 0]
    sd_0 = sigma if sigma.ndim == 1 else sigma[..., 0]

    sample[..., 0] = rng.normal(loc=mean_0, scale=sd_0)

    for t in np.arange(1, T):
        prev_other = sample[..., prev_time_diff_subject[t]] * symmetry_mask  # D

        # Previous sample from the same individual
        if self_dependent and prev_same_subject_mask[t] == 1:
            prev_same = sample[..., prev_time_same_subject[t]]
        else:
            if initial_mean.shape[1] == 1:
                prev_same = initial_mean[..., 0]
            else:
                prev_same = initial_mean[..., t]

        c = coordination[t]
        dt_diff = 1
        F = np.array([[1, dt_diff], [0, 1 - c * prev_diff_subject_mask[t]]])
        U = np.array(
            [
                [0, 0],  # position of "b" does not influence position of "a"
                [
                    0,
                    c * prev_diff_subject_mask[t],
                ],  # speed of "b" influences the speed of "a" when there's coordination.
            ]
        )

        blended_mean = np.dot(F, prev_same) + np.dot(U, prev_other)

        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            transition_sample = rng.normal(loc=blended_mean, scale=sigma[..., 0])
        else:
            transition_sample = rng.normal(loc=blended_mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample
