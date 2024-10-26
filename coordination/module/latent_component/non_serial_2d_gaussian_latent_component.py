from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm
import pytensor as pt

from coordination.common.types import TensorTypes
from coordination.module.constants import (DEFAULT_LATENT_MEAN_PARAM,
                                           DEFAULT_LATENT_SD_PARAM,
                                           DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_SAMPLING_RELATIVE_FREQUENCY,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS)
from coordination.module.latent_component.non_serial_gaussian_latent_component import \
    NonSerialGaussianLatentComponent
from coordination.module.common_cause.common_cause_gaussian_2d import CommonCauseGaussian2D
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
            self_dependent: bool = True,
            mean_mean_a0: np.ndarray = DEFAULT_LATENT_MEAN_PARAM,
            sd_mean_a0: np.ndarray = DEFAULT_LATENT_SD_PARAM,
            sd_sd_a: np.ndarray = DEFAULT_LATENT_SD_PARAM,
            share_mean_a0_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
            share_mean_a0_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
            share_sd_a_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
            share_sd_a_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
            subject_names: Optional[List[str]] = None,
            # [TODO] Coodination Samples passed as a 1 x 2 x T [c1 is coordination, c2 is individual]
            coordination_samples: Optional[ModuleSamples] = None,
            coordination_random_variable: Optional[pm.Distribution] = None,
            latent_component_random_variable: Optional[pm.Distribution] = None,
            mean_a0_random_variable: Optional[pm.Distribution] = None,
            sd_a_random_variable: Optional[pm.Distribution] = None,
            sampling_relative_frequency: float = DEFAULT_SAMPLING_RELATIVE_FREQUENCY,
            time_steps_in_coordination_scale: Optional[np.array] = None,
            observed_values: Optional[TensorTypes] = None,
            mean_a0: Optional[Union[float, np.ndarray]] = None,
            sd_a: Optional[Union[float, np.ndarray]] = None,
            initial_samples: Optional[np.ndarray] = None,
            asymmetric_coordination: bool = False,
            single_chain: bool = False,
            common_cause: bool = False,
            common_cause_samples: Optional[ModuleSamples] = None,
            common_cause_random_variable: Optional[pm.Distribution] = None,
    ):
        """
        Creates a non-serial 2D Gaussian latent component.

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
        @param single_chain: whether to fit a single chain for all subjects.
        @param common_cause: whether to use a common cause chain or not.
        @param common_cause_samples: optional common cause samples to be used in a call to
            draw_samples. This variable must be set before such a call if common cause is used.
        @param common_cause_random_variable: an optional common cause random variable.
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
            mean_a0=mean_a0,
            sd_a=sd_a,
            initial_samples=initial_samples,
            asymmetric_coordination=asymmetric_coordination,
            single_chain=single_chain,
            common_cause=common_cause,
            common_cause_samples=common_cause_samples,
            common_cause_random_variable=common_cause_random_variable,
        )

        self.subject_names = subject_names
        self.sampling_relative_frequency = sampling_relative_frequency
        self.common_cause = common_cause

    def _draw_from_system_dynamics(
            self,
            sampled_coordination: np.ndarray,
            time_steps_in_coordination_scale: np.ndarray,
            mean_a0: np.ndarray,
            sd_a: np.ndarray,
            init_values: Optional[np.ndarray] = None,
            sampled_common_cause: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draws values with the following updating equations for the state of the component at time
        t:

        S_a(t) = S_a(t-1) + V_a(t-1)dt
        V_a(t) = (1 - C(t))*V_a(t-1) + C(t)*V_b(t-1),


        NEW: A(t) = c1(t-1) * A(t-1) + c2(t-1) * B(t-1) + (1 - c1(t-1) - c2(t-1)) * X_t

        where, "S" is position, "V" speed, "V_a" is the previous velocities of the subject and V_b
        the scaled sum of the previous velocities of other subjects: (V1 + V2 + ... Vn-1) /
        (n - 1). We set dt to be 1, meaning it jumps one time step in the component's scale instead of "n"
        time steps in the coordination scale when there are gaps.

        If we are modeling a common-cause phenomenon, then the equations are as follows:

        S_a(t) = S_a(t-1) + V_a(t-1)dt
        V_a(t) = (1 - C(t))*V_a(t-1) + C(t)*V_c(t),

        where V_c is the velocity of the common cause variable.

        @param sampled_coordination: sampled values of coordination (all series included).
        @param time_steps_in_coordination_scale: an array of indices representing time steps in the
        coordination scale that match those of the component's scale.
        @param mean_a0: initial mean of the latent component.
        @param sd_a: standard deviation of the Gaussian transition distribution.
        @param sampled_common_cause: optional sampled values of common cause (all series included).

        @return: sampled values.
        """
        zero_valued_diag_matrix = np.ones((self.num_subjects, self.num_subjects)) - np.eye(
            self.num_subjects)
        sum_matrix_others = zero_valued_diag_matrix / (self.num_subjects - 1)

        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros(
            (num_series, 1 if self.single_chain else self.num_subjects, self.dimension_size,
             num_time_steps)
        )

        # This function can be called to continue sampling from a specific time step. In that case,
        # the initial time step is not t = 0.
        t0 = 0 if init_values is None else init_values.shape[-1]
        if init_values is not None:
            values[..., :t0] = init_values

        for t in range(t0, num_time_steps):
            if t == 0:
                values[..., 0] = norm(loc=mean_a0, scale=sd_a).rvs(
                    size=(num_series, 1 if self.single_chain else self.num_subjects,
                          self.dimension_size)
                )
            else:
                if self.self_dependent:
                    prev_same = values[..., t - 1]  # n x s x d
                else:
                    prev_same = mean_a0  # n x s x d

                if self.single_chain:
                    blended_mean = prev_same
                else:
                    # ================= [TODO] SPLIT ====================
                    # [TODO!!] c1 and c2 should be sum to 1, this is norm.
                    # c = sampled_coordination[:, time_steps_in_coordination_scale[t]]  # n
                    c1 = sampled_coordination[0, 0, time_steps_in_coordination_scale[t - 1]]
                    c2 = sampled_coordination[0, 1, time_steps_in_coordination_scale[t - 1]]

                    c1 = np.array(c1)
                    c2 = np.array(c2)

                    c_sum = c1 + c2
                    c1 = c1 / c_sum
                    c2 = c2 / c_sum

                    c1 = c1[:, None, None]  # n x 1 x 1
                    c2 = c2[:, None, None]  # n x 1 x 1
                    # ================= [END] SPLIT ====================
                    c_mask = -1 if self.asymmetric_coordination else 1

                    if self.common_cause:
                        X_t_current = sampled_common_cause[..., t].repeat(mean_a0.shape[1], axis=1)
                        # n x s x d
                        prev_others = (
                                np.einsum("ij,kjl->kil", sum_matrix_others, values[..., t - 1])
                                * c_mask
                        )
                        dt_diff = 1
                        F = np.array([[1.0, dt_diff], [0.0, 0.0]])[None, :].repeat(
                            num_series, axis=0
                        )
                        F[:, 1, 1] = c2

                        U1 = np.zeros((num_series, 2, 2))
                        U1[:, 1, 1] = c1

                        U2 = np.zeros((num_series, 2, 2))
                        U2[:, 1, 1] = 1-c1-c2


                        blended_mean = np.einsum("kij,klj->kli", F, prev_same) + np.einsum(
                            "kij,klj->kli", U1, prev_others) + np.einsum("kij,klj->kli", U2, X_t_current)


                    else:
                    # [TODO] if we add a normalizer to c1, we need to normalize back.
                            # Ensure c1 and c2 are numpy arrays

                        epsilon = 1e-8
                        denominator = 1 - c2 + epsilon
                        c = c1 / denominator

                        c = c[:, None, None]  # Shape: n x 1 x 1
                        # n x s x d
                        prev_others = (
                                np.einsum("ij,kjl->kil", sum_matrix_others, values[..., t - 1])
                                * c_mask
                        )

                        # The matrix F multiplied by the state of a component "a" at time t - 1
                        # ([P(t-1), S(t-1)]) gives us:
                        #
                        # P_a(t) = P_a(t-1) + S_a(t-1)dt
                        # S_a(t) = (1 - C(t))*S_a(t-1)
                        #
                        # Then we just need to sum with [0, c(t)*S_b(t-1)] to obtain the updated
                        # state of the component. Which can be accomplished with
                        # U*[P_b(t-1), S_b(t-1)]
                        dt_diff = 1
                        F = np.array([[1.0, dt_diff], [0.0, 0.0]])[None, :].repeat(
                            num_series, axis=0
                        )
                        F[:, 1, 1] = 1 - c

                        U = np.zeros((num_series, 2, 2))
                        U[:, 1, 1] = c

                        blended_mean = np.einsum("kij,klj->kli", F, prev_same) + np.einsum(
                            "kij,klj->kli", U, prev_others
                        )

                values[..., t] = norm(loc=blended_mean, scale=sd_a[None, :]).rvs()

        if self.single_chain:
            return values.repeat(self.num_subjects, axis=1)

        return values

    def _get_extra_logp_params(self) -> Tuple[Union[TensorTypes, pm.Distribution], ...]:
        """
        Gets extra parameters to be passed to the log_prob and random functions.
        """
        return (self.common_cause_random_variable,) if self.common_cause else ()

    def _get_log_prob_fn(self) -> Callable:
        """
        Gets a reference to a log_prob function.
        """
        return common_cause_log_prob if self.common_cause else log_prob

    def _get_random_fn(self) -> Callable:
        """
        Gets a reference to a random function for prior predictive checks.
        """
        return common_cause_random if self.common_cause else random


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################


def common_cause_log_prob(
        sample: ptt.TensorVariable,
        initial_mean: ptt.TensorVariable,
        sigma: ptt.TensorVariable,
        coordination: ptt.TensorVariable,
        self_dependent: ptt.TensorConstant,
        symmetry_mask: int,
        common_cause: ptt.TensorConstant,
) -> ptt.TensorVariable:
    """
    Computes the log-probability function of a sample.

    This does not support a single chain or non self-dependency of the latent component.

    Legend:
    D: number of dimensions
    S: number of subjects
    T: number of time steps

    @param sample: (subject x 2 x time) a single samples series.
    @param initial_mean: (subject x 2) mean at t0 for each subject.
    @param sigma: (subject x 2) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise.
    @param common_cause: (subject x 2 x time) a series of common cause values.
    @return: log-probability of the sample.
    """
    S = sample.shape[0]
    D = sample.shape[1]
    T = sample.shape[2]
    # (S, D, T-1)
    previous_values = sample[:, :, :-1]
    current_values = sample[:, :, 1:]
    # ====================================================================================
    # [TODO] Maybe not need, but I copy here for the sum_matrix_others
    if self_dependent.eval():
        # The component's value for a subject depends on its previous value for the same subject.
        prev_same = sample[..., :-1]  # S x 2 x T-1
    else:
        # The component's value for a subject does not depend on its previous value for the same
        # subject. At every time step, the value from others is blended with a fixed value given
        # by the component's initial mean.
        prev_same = initial_mean[:, :, None].repeat(T - 1, axis=-1)
    if S.eval() == 1:
        # Single chain
        blended_mean = prev_same
    else:
        # Contains the sum of previous values of other subjects for each subject scaled by 1/(S-1).
        # We discard the last value as that is not a previous value of any other.
        sum_matrix_others = (ptt.ones((S, S)) - ptt.eye(S)) / (S - 1)
        prev_others = (
                ptt.tensordot(sum_matrix_others, sample, axes=(1, 0))[..., :-1] * symmetry_mask
        )  # S x 2 x T-1
     # ====================================================================================

    coordination=coordination/coordination.sum(axis=0)[None,:]
    # t0
    total_logp = pm.logp(
        pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(S, D)),
        sample[..., 0]
    )
    total_logp = total_logp.sum()
    # c1 for prev_others
    c1 = coordination[0, :-1]
    # c2 for prev_same
    c2 = coordination[1, :-1]
    c3 = coordination[2, 1: ]
    # 1 x 1 x T-1   
    c1 = c1.dimshuffle('x', 'x', 0)
    print(c1.eval())
    # 1 x 1 x T-1
    c2 = c2.dimshuffle('x', 'x', 0)  
    print(c2.eval())
    c3 = c3.dimshuffle('x', 'x', 0)

    blended_mean = c1 * prev_others[:, :, ] + c2 * prev_same + c3 * common_cause[:, :, 1:]


    sigma = sigma[:, :, None]
    logp = pm.logp(
        pm.Normal.dist(mu=blended_mean, sigma=sigma),
        current_values
    )
    total_logp += logp.sum()
    print(total_logp.eval())
    print(426)
    return total_logp


def log_prob(
        sample: ptt.TensorVariable,
        initial_mean: ptt.TensorVariable,
        sigma: ptt.TensorVariable,
        coordination: ptt.TensorVariable,
        self_dependent: ptt.TensorConstant,
        symmetry_mask: int,
) -> ptt.TensorVariable:
    """
    Computes the log-probability function of a sample.

    Legend:
    D: number of dimensions
    S: number of subjects
    T: number of time steps

    @param sample: (subject x 2 x time) a single samples series.
    @param initial_mean: (subject x 2) mean at t0 for each subject.
    @param sigma: (subject x 2) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise.
    @return: log-probability of the sample.
    """

    S = sample.shape[0]
    D = sample.shape[1]  # This should be 2: position and speed
    T = sample.shape[2]

    # log-probability at the initial time step
    total_logp = pm.logp(
        pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(S, D)), sample[..., 0]
    ).sum()

    if self_dependent.eval():
        # The component's value for a subject depends on its previous value for the same subject.
        prev_same = sample[..., :-1]  # S x 2 x T-1
    else:
        # The component's value for a subject does not depend on its previous value for the same
        # subject. At every time step, the value from others is blended with a fixed value given
        # by the component's initial mean.
        prev_same = initial_mean[:, :, None].repeat(T - 1, axis=-1)

    if S.eval() == 1:
        # Single chain
        blended_mean = prev_same
    else:
        # Contains the sum of previous values of other subjects for each subject scaled by 1/(S-1).
        # We discard the last value as that is not a previous value of any other.
        sum_matrix_others = (ptt.ones((S, S)) - ptt.eye(S)) / (S - 1)
        prev_others = (
                ptt.tensordot(sum_matrix_others, sample, axes=(1, 0))[..., :-1] * symmetry_mask
        )  # S x 2 x T-1

        # Coordination does not affect the component in the first time step because the subjects have
        # no previous dependencies at that time.
        # c = coordination[None, None, 1:]  # 1 x 1 x T-1
        # [TODO] coordination is 2d now, need another normalizer because we regardless of c2.
        c1 = coordination[0,0,1:]
        c2 = coordination[0,1,1:]
        c = c1 / ( 1 - c2 )

        # The dimensions of F and U are: T-1 x 2 x 2
        T = c.shape[0]
        F = ptt.as_tensor(np.array([[[1.0, 1.0], [0.0, 1.0]]])).repeat(T, axis=0)
        F = ptt.set_subtensor(F[:, 1, 1], 1 - c)

        U = ptt.as_tensor(np.array([[[0.0, 0.0], [0.0, 1.0]]])).repeat(T, axis=0)
        U = ptt.set_subtensor(U[:, 1, 1], c)

        # We transform the sample using the fundamental matrix so that we learn to generate samples
        # with the underlying system dynamics. If we just compare a sample with the blended_mean, we
        # are assuming the samples follow a random gaussian walk. Since we know the system dynamics,
        # we can add that to the log-probability such that the samples are effectively coming from the
        # component's posterior.
        #
        # prev_same.T has dimensions T-1 x 2 x S. The first dimension of both F and prev_same.T is T-1
        # and used as the batch dimension. The result of batched_tensordot will have dimensions
        # T-1 x 2 x S. Transposing that results in S x 2 x T-1 as desired.
        prev_same_transformed = ptt.batched_tensordot(F, prev_same.T, axes=[(2,), (1,)]).T
        prev_other_transformed = ptt.batched_tensordot(
            U, prev_others.T, axes=[(2,), (1,)]
        ).T

        blended_mean = prev_other_transformed + prev_same_transformed

    # Match the dimensions of the standard deviation with that of the blended mean by adding
    # another dimension for time.
    sd = sigma[:, :, None]

    # Index samples starting from the second index (i = 1) so that we can effectively compare
    # current values against previous ones (prev_others and prev_same).
    total_logp += pm.logp(
        pm.Normal.dist(mu=blended_mean, sigma=sd, shape=blended_mean.shape),
        sample[..., 1:],
    ).sum()

    return total_logp


def random(
        initial_mean: np.ndarray,
        sigma: np.ndarray,
        coordination: np.ndarray,
        self_dependent: bool,
        symmetry_mask: int,
        rng: Optional[np.random.Generator] = None,
        size: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """
    Generates samples from of a non-serial latent component for prior predictive checks.

    Legend:
    D: number of dimensions
    S: number of subjects
    T: number of time steps

    @param initial_mean: (subject x dimension) mean at t0 for each subject.
    @param sigma: (subject x dimension) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise.
    @param rng: random number generator.
    @param size: size of the sample.

    @return: a serial latent component sample.
    """

    # TODO: Unify this with the class sampling method.

    T = coordination.shape[-1]
    S = initial_mean.shape[0]

    sample = np.zeros(size)

    # Sample from prior in the initial time step
    sample[..., 0] = rng.normal(loc=initial_mean, scale=sigma, size=size[:-1])

    for t in np.arange(1, T):
        if self_dependent:
            # Previous sample from the same subject
            prev_same = sample[..., t - 1]
        else:
            # No dependency on the same subject. Sample from prior.
            prev_same = initial_mean

        if S == 1:
            # Single chain
            blended_mean = prev_same
        else:
            sum_matrix_others = (np.ones((S, S)) - np.eye(S)) / (S - 1)
            prev_others = (
                    np.dot(sum_matrix_others, sample[..., t - 1]) * symmetry_mask
            )  # S x D

            c = coordination[t]
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

            blended_mean = np.einsum("ij,lj->li", F, prev_same) + np.einsum(
                "ij,lj->li", U, prev_others
            )

        sample[..., t] = rng.normal(loc=blended_mean, scale=sigma)

    return sample


def common_cause_random(
        initial_mean: np.ndarray,  # (S, D)
        sigma: np.ndarray,  # (S, D)
        coordination: np.ndarray,  # (T)
        self_dependent: bool,
        symmetry_mask: int,
        rng: Optional[np.random.Generator] = None,
        size: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """
    Generates samples from of a non-serial latent component for prior predictive checks.

    Legend:
    D: number of dimensions
    S: number of subjects
    T: number of time steps

    @param initial_mean: (subject x dimension) mean at t0 for each subject.
    @param sigma: (subject x dimension) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise.
    @param rng: random number generator.
    @param size: size of the sample.

    @return: a serial latent component sample.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = coordination.shape[-1]
    S, D = initial_mean.shape
    if size is None:
        size = (S, D, T)
    # (S, D, T)
    sample = np.zeros(size)
    # (D, T)
    common_cause = np.zeros((D, T))
    # D
    initial_mean_common_cause = initial_mean.mean(axis=0)
    # D
    sigma_c = sigma.mean(axis=0)
    # common cause at t=0
    common_cause[:, 0] = rng.normal(loc=initial_mean_common_cause, scale=sigma_c)
    # At t=0, sample blended_mean
    c = coordination[0]
    # (S, D)
    blended_mean = (1 - c) * initial_mean + c * common_cause[:, 0][None, :]
    sample[:, :, 0] = rng.normal(loc=blended_mean, scale=sigma)

    # For t > 0
    for t in range(1, T):
        # common cause here
        common_cause[:, t] = rng.normal(loc=common_cause[:, t - 1], scale=sigma_c)

        # ------------------------------------------
        if self_dependent:
            prev_same = sample[:, :, t - 1]  # (S, D)
        else:
            prev_same = initial_mean  # (S, D)
        # prev_same = initial_mean  # (S, D)
        # ------------------------------------------

        c = coordination[t]
        dt = 1
        F = np.array([[1, dt], [0, 1 - c]])
        U = np.array([[0, 0], [0, c]])

        blended_mean = (F @ prev_same.T).T + (U @ common_cause[:, t])[None, :]  # (S, D)

        sample[:, :, t] = rng.normal(loc=blended_mean, scale=sigma)

    return sample
