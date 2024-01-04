from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.module.constants import (DEFAULT_LATENT_MEAN_PARAM,
                                           DEFAULT_LATENT_SD_PARAM,
                                           DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_SAMPLING_RELATIVE_FREQUENCY,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS)
from coordination.module.latent_component.non_serial_gaussian_latent_component import \
    NonSerialGaussianLatentComponent
from coordination.module.module import ModuleSamples


class NonSerial2DGaussianLatentComponent(NonSerialGaussianLatentComponent):
    """
    This class represents a 2D latent component with position and speed. It encodes the
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
        mean_mean_a0: np.ndarray = DEFAULT_LATENT_MEAN_PARAM,
        sd_mean_a0: np.ndarray = DEFAULT_LATENT_SD_PARAM,
        sd_sd_a: np.ndarray = DEFAULT_LATENT_SD_PARAM,
        share_mean_a0_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
        share_mean_a0_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
        share_sd_a_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
        share_sd_a_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
        subject_names: Optional[List[str]] = None,
        coordination_samples: Optional[ModuleSamples] = None,
        coordination_random_variable: Optional[pm.Distribution] = None,
        latent_component_random_variable: Optional[pm.Distribution] = None,
        mean_a0_random_variable: Optional[pm.Distribution] = None,
        sd_a_random_variable: Optional[pm.Distribution] = None,
        sampling_relative_frequency: float = DEFAULT_SAMPLING_RELATIVE_FREQUENCY,
        time_steps_in_coordination_scale: Optional[np.array] = None,
        observed_values: Optional[TensorTypes] = None,
    ):
        """
        Creates a non-serial 2D Gaussian latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
            latent component).
        @param sd_sd_a: std of the hyper-prior of sigma_a (std of the Gaussian random walk of
        the latent component).
        @param share_mean_a0_across_subjects: whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: whether to use the same mu_a0 for all dimensions.
        @param share_sd_a_across_subjects: whether to use the same sigma_a for all subjects.
        @param share_sd_a_across_dimensions: whether to use the same sigma_a for all dimensions.
        @param subject_names: the names of each subject of the latent component. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
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
        @param sampling_relative_frequency: a number larger or equal than 1 indicating the
            frequency in of the latent component with respect to coordination for sample data
            generation. For instance, if frequency is 2, there will be one component sample every
            other time step in the coordination scale.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param observed_values: observations for the serial latent component random variable. If
            a value is set, the variable is not latent anymore.
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=2,  # position and speed
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
        )

        self.subject_names = subject_names
        self.sampling_relative_frequency = sampling_relative_frequency

    def _draw_from_system_dynamics(
        self,
        sampled_coordination: np.ndarray,
        time_steps_in_coordination_scale: np.ndarray,
        mean_a0: np.ndarray,
        sd_a: np.ndarray,
    ) -> np.ndarray:
        """
        Draws values with the following updating equations for the state of the component at time
        t:

        P_a(t) = P_a(t-1) + V_a(t-1)dt
        S_a(t) = (1 - C(t))*S_a(t-1) + c(t)*S_b(t-1)

        Where, "P" is position, "S" speed, "S_a" is the previous positions of the subject and S_b
        the scaled sum of the previous positions of other subjects: (S1 + S2 + ... Sn-1) / (n - 1).
        We set dt to be 1, meaning it jumps one time step in the component's scale instead of "n"
        time steps in the coordination scale when there are gaps.

        @param sampled_coordination: sampled values of coordination (all series included).
        @param time_steps_in_coordination_scale: an array of indices representing time steps in the
        coordination scale that match those of the component's scale.
        @param mean_a0: initial mean of the latent component.
        @param sd_a: standard deviation of the Gaussian transition distribution.

        @return: sampled values.
        """

        # Axes legend:
        # n: number of series (first dimension of coordination)
        # s: number of subjects
        # d: dimension size

        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros(
            (num_series, self.num_subjects, self.dimension_size, num_time_steps)
        )

        N = self.num_subjects
        sum_matrix_others = (np.ones((N, N)) - np.eye(N)) / (N - 1)

        for t in range(num_time_steps):
            if t == 0:
                values[..., 0] = norm(loc=mean_a0[None, :], scale=sd_a[None, :]).rvs(
                    size=(num_series, self.num_subjects, self.dimension_size)
                )
            else:
                # n x 1 x 1
                c = sampled_coordination[:, time_steps_in_coordination_scale[t]][
                    :, None, None
                ]

                prev_others = np.dot(sum_matrix_others, values[..., t - 1])  # n x s x d
                prev_same = values[..., t - 1]  # n x s x d

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

                blended_mean = np.einsum("ij,klj->kli", F, prev_same) + np.einsum(
                    "ij,klj->kli", U, prev_others
                )

                # blended_mean = (prev_others - prev_same) * c + prev_same  # n x s x d

                values[..., t] = norm(loc=blended_mean, scale=sd_a[None, :]).rvs()

        return values

    def _get_extra_logp_params(self) -> Tuple[Union[TensorTypes, pm.Distribution], ...]:
        """
        Gets extra parameters to be passed to the log_prob and random functions.
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
        """
        return None


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################


def log_prob(
    sample: ptt.TensorVariable,
    initial_mean: ptt.TensorVariable,
    sigma: ptt.TensorVariable,
    coordination: ptt.TensorVariable,
    self_dependent: ptt.TensorConstant,
) -> float:
    """
    Computes the log-probability function of a sample.

    @param sample: (subject x dimension x time) a single samples series.
    @param initial_mean: (subject x dimension) mean at t0 for each subject.
    @param sigma: (subject x dimension) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @return: log-probability of the sample.
    """

    N = sample.shape[0]
    D = sample.shape[1]

    # log-probability at the initial time step
    total_logp = pm.logp(
        pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(N, D)), sample[..., 0]
    ).sum()

    # Contains the sum of previous values of other subjects for each subject scaled by 1/(s-1).
    # We discard the last value as that is not a previous value of any other.
    sum_matrix_others = (ptt.ones((N, N)) - ptt.eye(N)) / (N - 1)
    prev_others = ptt.tensordot(sum_matrix_others, sample, axes=(1, 0))[..., :-1]
    prev_same = sample[..., :-1]

    # Coordination does not affect the component in the first time step because the subjects have
    # no previous dependencies at that time.
    # c = coordination[None, None, 1:]  # 1 x 1 x t-1
    c = coordination[1:]  # 1 x 1 x t-1

    T = c.shape[0] - 1
    F = ptt.as_tensor(np.array([[[1.0, 1.0], [0.0, 1.0]]])).repeat(T, axis=0)
    F = ptt.set_subtensor(F[:, 1, 1], 1 - coordination * prev_diff_subject_mask)

    U = ptt.as_tensor(np.array([[[0.0, 0.0], [0.0, 1.0]]])).repeat(T, axis=0)
    U = ptt.set_subtensor(U[:, 1, 1], coordination * prev_diff_subject_mask)

    # We transform the sample using the fundamental matrix so that we learn to generate samples
    # with the underlying system dynamics. If we just compare a sample with the blended_mean, we
    # are assuming the samples follow a random gaussian walk. Since we know the system dynamics,
    # we can add that to the log-probability such that the samples are effectively coming from the
    # component's posterior.
    prev_same_transformed = ptt.batched_tensordot(F, prev_same.T, axes=[(2,), (1,)]).T
    prev_other_transformed = ptt.batched_tensordot(
        U, prev_others.T, axes=[(2,), (1,)]
    ).T

    blended_mean = prev_other_transformed + prev_same_transformed

    # blended_mean = (prev_others - prev_same) * c + prev_same

    # Match the dimensions of the standard deviation with that of the blended mean
    sd = sigma[:, :, None]

    # Index samples starting from the second index (i = 1) so that we can effectively compare
    # current values against previous ones (prev_others and prev_same).
    total_logp += pm.logp(
        pm.Normal.dist(mu=blended_mean, sigma=sd, shape=blended_mean.shape),
        sample[..., 1:],
    ).sum()

    return total_logp


#
#
# def random(
#         initial_mean: np.ndarray,
#         sigma: np.ndarray,
#         coordination: np.ndarray,
#         self_dependent: bool,
#         rng: Optional[np.random.Generator] = None,
#         size: Optional[Tuple[int]] = None,
# ) -> np.ndarray:
#     """
#     Generates samples from of a non-serial latent component for prior predictive checks.
#
#     @param initial_mean: (subject x dimension) mean at t0 for each subject.
#     @param sigma: (subject x dimension) a series of standard deviations. At each time the standard
#         deviation is associated with the subject at that time.
#     @param coordination: (time) a series of coordination values.
#     @param self_dependent: a boolean indicating whether subjects depend on their previous values.
#     @param rng: random number generator.
#     @param size: size of the sample.
#
#     @return: a serial latent component sample.
#     """
#
#     # TODO: Unify this with the class sampling method.
#
#     T = coordination.shape[-1]
#     N = initial_mean.shape[0]
#
#     noise = rng.normal(loc=0, scale=1, size=size) * sigma[:, :, None]
#
#     sample = np.zeros_like(noise)
#
#     # Sample from prior in the initial time step
#     sample[..., 0] = rng.normal(loc=initial_mean, scale=sigma, size=noise.shape[:-1])
#
#     sum_matrix_others = (ptt.ones((N, N)) - ptt.eye(N)) / (N - 1)
#     for t in np.arange(1, T):
#         prev_others = np.dot(sum_matrix_others, sample[..., t - 1])  # s x d
#
#         if self_dependent:
#             # Previous sample from the same subject
#             prev_same = sample[..., t - 1]
#         else:
#             # No dependency on the same subject. Sample from prior.
#             prev_same = initial_mean
#
#         blended_mean = (prev_others - prev_same) * coordination[t] + prev_same
#
#         sample[..., t] = rng.normal(loc=blended_mean, scale=sigma)
#
#     return sample + noise
