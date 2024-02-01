from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.module.constants import (DEFAULT_DAMPENING_COEFFICIENT,
                                           DEFAULT_DT,
                                           DEFAULT_LATENT_MEAN_PARAM,
                                           DEFAULT_LATENT_SD_PARAM,
                                           DEFAULT_MASS, DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_SAMPLING_RELATIVE_FREQUENCY,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS,
                                           DEFAULT_SPRING_CONSTANT)
from coordination.module.latent_component.non_serial_gaussian_latent_component import \
    NonSerialGaussianLatentComponent
from coordination.module.module import ModuleSamples


class NonSerialMassSpringDamperLatentComponent(NonSerialGaussianLatentComponent):
    """
    This class represents a non-serial latent component with oscillatory dynamics and position
    coupling determined by coordination and no external force. States are blended according to the
    motion equation os coupled oscillators, meaning no damping effect is created by the blending.
    """

    def __init__(
        self,
        uuid: str,
        pymc_model: pm.Model,
        num_subjects: int = DEFAULT_NUM_SUBJECTS,
        spring_constant: np.ndarray = DEFAULT_SPRING_CONSTANT,
        mass: np.ndarray = DEFAULT_MASS,
        dampening_coefficient: np.ndarray = DEFAULT_DAMPENING_COEFFICIENT,
        dt: float = DEFAULT_DT,
        mean_mean_a0: np.ndarray = DEFAULT_LATENT_MEAN_PARAM,
        sd_mean_a0: np.ndarray = DEFAULT_LATENT_SD_PARAM,
        share_mean_a0_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
        sd_sd_a: np.ndarray = DEFAULT_LATENT_SD_PARAM,
        share_mean_a0_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
        share_sd_a_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
        share_sd_a_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
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
    ):
        """
        Creates a non-serial mass-spring-damper latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param spring_constant: spring constant per subject/spring used to calculate the
            fundamental matrix of the motion.
        @param mass: mass per subject/spring used to calculate the fundamental
            matrix of the motion.
        @param dampening_coefficient: dampening coefficient per subject used to calculate the
            fundamental matrix of the motion.
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
        """
        super().__init__(
            uuid=uuid,
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
            sampling_relative_frequency=sampling_relative_frequency,
            mean_a0=mean_a0,
            sd_a=sd_a,
        )

        if spring_constant.ndim != 1:
            raise ValueError(
                f"The spring constant ({spring_constant}) must be a 1D array."
            )

        if len(spring_constant) != num_subjects:
            raise ValueError(
                f"The spring constant ({spring_constant}) must have size {num_subjects}."
            )

        if dampening_coefficient.ndim != 1:
            raise ValueError(
                f"The dampening coefficient ({dampening_coefficient}) must be a 1D "
                "array."
            )

        if len(dampening_coefficient) != num_subjects:
            raise ValueError(
                f"The dampening coefficient ({dampening_coefficient}) must have size "
                f"{num_subjects}."
            )

        if mass.ndim != 1:
            raise ValueError(f"The mass ({mass}) must be a 1D array.")

        if len(mass) != num_subjects:
            raise ValueError(f"The mass ({mass}) must have size {num_subjects}.")

        if dt <= 0:
            raise ValueError(f"The dt ({dt}) must be a positive number.")

        if num_subjects > 2:
            raise NotImplementedError(
                "Dynamics not implemented for more than 2 masses yet."
            )

        self.spring_constant = spring_constant
        self.mass = mass
        self.damping_coefficient = dampening_coefficient
        self.dt = dt

        # Fundamental matrices. One per subject.
        F = []
        for subject in range(num_subjects):
            A = np.array(
                [
                    [0, 1],
                    [
                        -self.spring_constant[subject] / self.mass[subject],
                        -self.damping_coefficient[subject] / self.mass[subject],
                    ],
                ]
            )
            F.append(expm(A * self.dt)[None, ...])

        self.F = np.concatenate(F, axis=0)

    def _draw_from_system_dynamics(
        self,
        sampled_coordination: np.ndarray,
        time_steps_in_coordination_scale: np.ndarray,
        mean_a0: np.ndarray,
        sd_a: np.ndarray,
    ) -> np.ndarray:
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
                values[..., t] = norm(loc=mean_a0[None, :], scale=sd_a[None, :]).rvs(
                    size=(num_series, self.num_subjects, self.dimension_size)
                )
            else:
                # n x 1 x 1
                c = sampled_coordination[:, time_steps_in_coordination_scale[t]][
                    :, None, None
                ]

                # Here, second-order differential equations that describe the dynamics of coupled
                # oscillators with two masses are:
                # acc_a = -k_a/m_a -d_a/m_a + c(x_b - x_a)
                # acc_b = -k_b/m_b -d_b/m_b + c(x_a - x_b)
                # Given a state vector formed by [x_a, v_a, x_b, v_b], we can write the equation
                # in matrix form:
                # v = Ax, where A is:
                # [0,              1, 0,              0]
                # [-(k_a/m_a + c), 0, c,              0]
                # [0,              0, 0,              1]
                # [c,              0, -(k_b/m_b + c), 0]
                f_a = self.spring_constant[0] / self.mass[0]
                f_b = self.spring_constant[1] / self.mass[1]
                d_a = self.damping_coefficient[0]
                d_b = self.damping_coefficient[1]

                A = np.array(
                    [
                        [0, 1, 0, 0],
                        [9, -d_a, 9, 0],  # 9 is a placeholder
                        [0, 0, 0, 1],
                        [9, 0, 9, -d_b],
                    ]
                )[None, :].repeat(num_series, axis=0)

                # Replace the cells with 9 with appropriate values
                A[:, 1, 0] = -(f_a + c)
                A[:, 1, 2] = c
                A[:, 3, 0] = c
                A[:, 3, 2] = -(f_b + c)

                # Find the fundamental matrix by solving the DE.
                F = expm(A * self.dt)

                # Flatten the last two dimensions to put to create the state vector
                # [x_a, v_a, x_b, v_b].
                concatenated_state = values[..., t - 1].reshape((num_series, -1))
                blended_mean = np.einsum("ijk,ik->ij", F, concatenated_state).reshape(
                    (num_series, self.num_subjects, self.dimension_size)
                )

                values[..., t] = norm(loc=blended_mean, scale=sd_a[None, :]).rvs()

        return values

    def _get_extra_logp_params(self) -> Tuple[Union[TensorTypes, pm.Distribution], ...]:
        """
        Gets the motion matrix template.
        """

        f_a = self.spring_constant[0] / self.mass[0]
        f_b = self.spring_constant[1] / self.mass[1]
        d_a = self.damping_coefficient[0]
        d_b = self.damping_coefficient[1]

        # Template motion matrix. Sampled coordination will complement some cells during inference.
        A = np.array(
            [[0, 1, 0, 0], [-f_a, -d_a, 9, 0], [0, 0, 0, 1], [9, 0, -f_b, -d_b]]
        )
        return A, self.dt

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
    A: ptt.TensorConstant,
    dt: ptt.TensorConstant,
) -> float:
    """
    Computes the log-probability function of a sample.

    @param sample: (dimension x time) a single samples series.
    @param initial_mean: (dimension x time) a series of mean at t0. At each time the mean is
        associated with the subject at that time. The initial mean is only used the first time the
        user speaks, but we repeat the values here over time for uniform vector operations (e.g.,
        we can multiply this with other tensors) and we fix the behavior with mask tensors.
    @param sigma: (dimension x time) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values. Axis (time).
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
        Not used by this implementation as self_dependency is fixed to True.
    @param A: (4 x 4) motion matrix template to be complemented with sampled coordination.
    @param dt: the size of each time step to calculate the fundamental matrix of the motion.
    """

    N = sample.shape[0]  # num subjects
    D = sample.shape[1]
    T = sample.shape[2]

    # logp at the initial time step
    total_logp = pm.logp(
        pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(N, D)), sample[..., 0]
    ).sum()

    # Adding another axis for time.
    A = A[None, :].repeat(T - 1, axis=0)

    # Update appropriate cells with samples coordination
    c = coordination[1:]  # T-1

    A = ptt.set_subtensor(A[:, 1, 0], A[:, 1, 0] - c)
    A = ptt.set_subtensor(A[:, 1, 2], c)
    A = ptt.set_subtensor(A[:, 3, 0], c)
    A = ptt.set_subtensor(A[:, 3, 2], A[:, 3, 2] - c)

    # Find the fundamental matrix by solving the DE.
    # F can be obtained with matrix exponential but doing so like below lead to a series of
    # numerical problems. So, I am approximating expm with a power series.
    # F, _ = pt.scan(fn=lambda x, _: ptt.slinalg.Expm()(x), outputs_info=ptt.ones_like(A[0]),
    #                sequences=[A * dt])  # T-1 x 4 x 4

    # Approximation of F with a power series
    I = ptt.eye(N * N)[None, :].repeat(T - 1, axis=0)
    A = A * dt
    A2 = ptt.power(A, 2)
    A3 = ptt.power(A, 3)
    F = I + A + A2 / 2 + A3 / 6

    # Flatten the last two dimensions to create the state vector
    # [x_a, v_a, x_b, v_b].
    concatenated_state = sample[..., :-1].reshape((-1, T - 1)).T  # T-1 x 4
    blended_mean = ptt.batched_tensordot(
        F, concatenated_state, axes=[(2,), (1,)]
    ).T.reshape((N, D, T - 1))

    # # Contains the sum of previous values of other subjects for each subject scaled by 1/(s-1).
    # # We discard the last value as that is not a previous value of any other.
    # sum_matrix_others = (ptt.ones((N, N)) - ptt.eye(N)) / (N - 1)
    #
    # # We transform the sample using the fundamental matrix so that we learn to generate samples
    # # with the underlying system dynamics. If we just compare a sample with the blended_mean, we
    # # are assuming the samples follow a random gaussian walk. Since we know the system dynamics,
    # # we can add that to the log-probability such that the samples are effectively coming from the
    # # component's posterior.
    # transformed_sample = ptt.batched_tensordot(F, sample, axes=[(2,), (1,)])
    # prev_others = ptt.tensordot(sum_matrix_others, transformed_sample, axes=(1, 0))[
    #               ..., :-1
    #               ]
    #
    # # The component's value for a subject depends on its previous value for the same subject.
    # prev_same = transformed_sample[..., :-1]  # N x d x t-1
    #
    # # Coordination does not affect the component in the first time step because the subjects have
    # # no previous dependencies at that time.
    # c = coordination[None, None, 1:]  # 1 x 1 x t-1
    #
    # blended_mean = (prev_others - prev_same) * c + prev_same
    #
    # # Decide which dimension(s) to blend
    # blended_mean = ptt.tensordot(B1, blended_mean, axes=[(1,), (1,)]).swapaxes(
    #     0, 1
    # ) + ptt.tensordot(B2, prev_same, axes=[(1,), (1,)]).swapaxes(0, 1)

    # # We don't blend velocity
    # POSITION_COL = ptt.as_tensor(np.array([[1, 0]])).repeat(N, 0)[..., None]
    # VELOCITY_COL = ptt.as_tensor(np.array([[0, 1]])).repeat(N, 0)[..., None]
    # blended_mean = blended_mean * POSITION_COL + prev_same * VELOCITY_COL

    # Match the dimensions of the standard deviation with that of the blended mean
    sd = sigma[:, :, None]

    # Index samples starting from the second index (i = 1) so that we can effectively compare
    # current values against previous ones (prev_others and prev_same).
    total_logp += pm.logp(
        pm.Normal.dist(mu=blended_mean, sigma=sd, shape=blended_mean.shape),
        sample[..., 1:],
    ).sum()

    return total_logp
