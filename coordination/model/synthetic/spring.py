from typing import Optional

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.model.model import Model
from coordination.model.synthetic.constants import (
    DAMPENING_COEFFICIENTS_SPRING_MODEL, DT_SPRING_MODEL,
    INITIAL_STATE_SPRING_MODEL, MASS_SPRING_MODEL, MEAN_MEAN_A0_SPRING_MODEL,
    MEAN_UC0, NUM_SPRINGS, NUM_TIME_STEPS,
    SAMPLING_RELATIVE_FREQUENCY_SPRING_MODEL, SD_A_SPRING_MODEL,
    SD_MEAN_A0_SPRING_MODEL, SD_MEAN_UC0, SD_O_SPRING_MODEL, SD_SD_A, SD_SD_O,
    SD_SD_UC, SD_UC, SHARE_MEAN_A0_ACROSS_DIMENSIONS,
    SHARE_MEAN_A0_ACROSS_SUBJECT, SHARE_SD_ACROSS_DIMENSIONS,
    SHARE_SD_ACROSS_SUBJECTS, SPRING_CONSTANT_SPRING_MODEL)
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.non_serial_mass_spring_damper_latent_component import \
    NonSerialMassSpringDamperLatentComponent
from coordination.module.module import ModuleSamples
from coordination.module.observation.non_serial_gaussian_observation import \
    NonSerialGaussianObservation


class SpringModel(Model):
    """
    This class represents a spring model where springs are influenced by each other as controlled
    by coordination.

    The variables in the latent component and observations are 2 dimensional. The first dimension
    contains the position of the mass attached to the spring and the second its speed. The speed
    evolves with the laws that govern harmonic oscillators but it is not blended by coordination
    as the mass position is.
    """

    def __init__(
        self,
        pymc_model: Optional[pm.Model] = None,
        num_springs: int = NUM_SPRINGS,
        num_time_steps_in_coordination_scale: int = NUM_TIME_STEPS,
        spring_constant: np.ndarray = SPRING_CONSTANT_SPRING_MODEL,
        mass: np.ndarray = MASS_SPRING_MODEL,
        dampening_coefficient: np.ndarray = DAMPENING_COEFFICIENTS_SPRING_MODEL,
        dt: float = DT_SPRING_MODEL,
        sd_mean_uc0: float = SD_MEAN_UC0,
        sd_sd_uc: float = SD_SD_UC,
        mean_mean_a0: np.ndarray = MEAN_MEAN_A0_SPRING_MODEL,
        sd_mean_a0: np.ndarray = SD_MEAN_A0_SPRING_MODEL,
        sd_sd_a: np.ndarray = SD_SD_A,
        sd_sd_o: np.ndarray = SD_SD_O,
        share_mean_a0_across_subjects: bool = SHARE_MEAN_A0_ACROSS_SUBJECT,
        share_mean_a0_across_dimensions: bool = SHARE_MEAN_A0_ACROSS_DIMENSIONS,
        share_sd_a_across_subjects: bool = SHARE_SD_ACROSS_SUBJECTS,
        share_sd_a_across_dimensions: bool = SHARE_SD_ACROSS_DIMENSIONS,
        share_sd_o_across_subjects: bool = SHARE_SD_ACROSS_SUBJECTS,
        share_sd_o_across_dimensions: bool = SHARE_SD_ACROSS_DIMENSIONS,
        sampling_relative_frequency: float = SAMPLING_RELATIVE_FREQUENCY_SPRING_MODEL,
        coordination_samples: Optional[ModuleSamples] = None,
    ):
        """
        Creates a spring model.

        @param pymc_model: a PyMC model instance where modules are to be created at. If not
            provided, it will be created along with this model instance.
        @param num_springs: the number of springs in the model.
        @param num_time_steps_in_coordination_scale: size of the coordination series.
        @param spring_constant: spring constant per spring used to calculate the fundamental
            matrix of the motion.
        @param mass: mass per spring used to calculate the fundamental matrix of the motion.
        @param dampening_coefficient: dampening coefficient per subject used to calculate the
            fundamental matrix of the motion.
        @param dt: the size of each time step to calculate the fundamental matrix of the motion.
        @param sd_mean_uc0: std of the hyper-prior of mu_uc0.
        @param sd_sd_uc: std of the hyper-prior of sigma_uc (std of the Gaussian random walk of
            the unbounded coordination).
        @param dampening_coefficient: dampening coefficient per subject used to calculate the
            fundamental matrix of the motion.
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
            latent component).
        @param sd_sd_a: std of the hyper-prior of sigma_a (std of the Gaussian random walk of
        the latent component).
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the Gaussian emission
            distribution).
        @param share_mean_a0_across_subjects: whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: whether to use the same mu_a0 for all dimensions.
        @param share_sd_a_across_subjects: whether to use the same sigma_a for all subjects.
        @param share_sd_a_across_dimensions: whether to use the same sigma_a for all dimensions.
        @param share_sd_o_across_subjects: whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: whether to use the same sigma_o for all dimensions.
        @param sampling_relative_frequency: a number larger or equal than 1 indicating the
            frequency in of the latent component with respect to coordination for sample data
            generation. For instance, if frequency is 2, there will be one component sample every
            other time step in the coordination scale.
        @param coordination_samples: coordination samples. If not provided, coordination samples
            will be draw in a call to draw_samples.
        """

        if not pymc_model:
            pymc_model = pm.Model()

        coordination = SigmoidGaussianCoordination(
            pymc_model=pymc_model,
            sd_mean_uc0=sd_mean_uc0,
            sd_sd_uc=sd_sd_uc,
            num_time_steps=num_time_steps_in_coordination_scale,
        )

        # Save a direct reference to state_space and observation for easy access in the parameter
        # setting functions in this class.
        self.state_space = NonSerialMassSpringDamperLatentComponent(
            uuid="state_space",
            pymc_model=pymc_model,
            num_subjects=num_springs,
            spring_constant=spring_constant,
            mass=mass,
            dampening_coefficient=dampening_coefficient,
            dt=dt,
            mean_mean_a0=mean_mean_a0,
            sd_mean_a0=sd_mean_a0,
            sd_sd_a=sd_sd_a,
            share_mean_a0_across_subjects=share_mean_a0_across_subjects,
            share_sd_a_across_subjects=share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=share_sd_a_across_dimensions,
            sampling_relative_frequency=sampling_relative_frequency,
        )

        self.observation = NonSerialGaussianObservation(
            uuid="observation",
            pymc_model=pymc_model,
            num_subjects=num_springs,
            dimension_size=self.state_space.dimension_size,
            sd_sd_o=sd_sd_o,
            share_sd_o_across_subjects=share_sd_o_across_subjects,
            share_sd_o_across_dimensions=share_sd_o_across_dimensions,
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=pymc_model,
            latent_component=self.state_space,
            observations=[self.observation],
        )

        super().__init__(
            name="conversation_model",
            pymc_model=pymc_model,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=coordination_samples,
        )

    def prepare_for_sampling(
        self,
        mean_uc0: float = MEAN_UC0,
        sd_uc: float = SD_UC,
        initial_state: np.ndarray = INITIAL_STATE_SPRING_MODEL,
        sd_a: np.ndarray = SD_A_SPRING_MODEL,
        sd_o: np.ndarray = SD_O_SPRING_MODEL,
    ):
        """
        Sets parameter values for sampling.

        @param mean_uc0: mean of the initial value of the unbounded coordination.
        @param sd_uc: standard deviation of the initial value and random Gaussian walk of the
            unbounded coordination.
        @param initial_state: value of the latent component at t = 0.
        @param sd_a: noise in the Gaussian random walk in the state space.
        @param sd_o: noise in the observation.
        """

        self.coordination.parameters.mean_uc0.value = np.ones(1) * mean_uc0
        self.coordination.parameters.sd_uc.value = np.ones(1) * sd_uc
        self.state_space.parameters.mean_a0.value = initial_state
        self.state_space.parameters.sd_a.value = sd_a
        self.observation.parameters.sd_o.value = sd_o

    def prepare_for_inference(
        self,
        num_time_steps_in_coordination_scale: int,
        time_steps_in_coordination_scale: np.array,
        observed_values: TensorTypes,
    ):
        """
        Sets metadata required for inference.

        @param num_time_steps_in_coordination_scale: size of the coordination series.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param observed_values: observations for the latent component random variable.
        """

        self.coordination.num_time_steps = num_time_steps_in_coordination_scale
        self.state_space.time_steps_in_coordination_scale = (
            time_steps_in_coordination_scale
        )
        self.observation.observed_values = observed_values
        self.observation.time_steps_in_coordination_scale = (
            time_steps_in_coordination_scale
        )

        self.create_random_variables()
