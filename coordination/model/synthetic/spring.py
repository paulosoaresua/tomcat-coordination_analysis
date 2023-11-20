from typing import Optional

from coordination.common.types import TensorTypes

import numpy as np
import pymc as pm

from coordination.module.module import ModuleSamples
from coordination.model.model import Model
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.component_group import ComponentGroup
from coordination.module.latent_component.non_serial_mass_spring_damper_latent_component import \
    NonSerialMassSpringDamperLatentComponent
from coordination.module.observation.non_serial_gaussian_observation import \
    NonSerialGaussianObservation
from coordination.model.synthetic.constants import (
    NUM_SUBJECTS,
    SPRING_NAMES_SPRING_MODEL,
    SPRING_CONSTANT_SPRING_MODEL,
    MASS_SPRING_MODEL,
    DAMPENING_COEFFICIENTS_SPRING_MODEL,
    DT_SPRING_MODEL,
    INITIAL_STATE_SPRING_MODEL,
    SD_MEAN_UC0,
    SD_SD_UC,
    MEAN_MEAN_A0,
    SD_MEAN_A0_SPRING_MODEL,
    SD_SD_A,
    SD_SD_O,
    SHARE_MEAN_A0_ACROSS_SUBJECT,
    SHARE_MEAN_A0_ACROSS_DIMENSIONS,
    SHARE_SD_ACROSS_SUBJECTS,
    SHARE_SD_ACROSS_DIMENSIONS,
    SD_A_SPRING_MODEL,
    SD_O_SPRING_MODEL,
    SAMPLING_RELATIVE_FREQUENCY_SPRING_MODEL
)


class SpringModel(Model):

    def __init__(
            self,
            pymc_model: pm.Model,
            num_springs: int = NUM_SUBJECTS,
            spring_constant: np.ndarray = SPRING_CONSTANT_SPRING_MODEL,
            mass: np.ndarray = MASS_SPRING_MODEL,
            dampening_coefficient: np.ndarray = DAMPENING_COEFFICIENTS_SPRING_MODEL,
            dt: float = DT_SPRING_MODEL,
            sd_mean_uc0: float = SD_MEAN_UC0,
            sd_sd_uc: float = SD_SD_UC,
            mean_mean_a0: np.ndarray = MEAN_MEAN_A0,
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
            coordination_samples: Optional[ModuleSamples] = None):
        coordination = SigmoidGaussianCoordination(
            pymc_model=pymc_model,
            sd_mean_uc0=sd_mean_uc0,
            sd_sd_uc=sd_sd_uc
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
            share_sd_o_across_dimensions=share_sd_o_across_dimensions
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=pymc_model,
            latent_component=self.state_space,
            observations=[self.observation]
        )

        super().__init__(
            name="conversation_model",
            pymc_model=pymc_model,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=coordination_samples
        )

    def prepare_for_sampling(self,
                             initial_state: np.ndarray = INITIAL_STATE_SPRING_MODEL,
                             sd_a: np.ndarray = SD_A_SPRING_MODEL,
                             sd_o: np.ndarray = SD_O_SPRING_MODEL):
        """
        Sets parameter values for sampling.

        @param initial_state: value of the latent component at t = 0.
        @param sd_a: noise in the Gaussian random walk in the state space.
        @param sd_o: noise in the observation.
        """

        self.component_groups[0].latent_component.parameters.mean_a0.value = initial_state
        self.component_groups[0].latent_component.parameters.sd_a.value = sd_a
        self.component_groups[0].observations[0].parameters.sd_o.value = sd_o

    def prepare_for_inference(self,
                              num_time_steps_in_coordination_scale: int,
                              time_steps_in_coordination_scale: np.array,
                              observed_values: TensorTypes):
        """
        Sets metadata required for inference.

        @param num_time_steps_in_coordination_scale: size of the coordination series.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param observed_values: observations for the latent component random variable.
        """

        self.coordination.num_time_steps = num_time_steps_in_coordination_scale
        self.state_space.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.observation.observed_values = observed_values
        self.observation.time_steps_in_coordination_scale = time_steps_in_coordination_scale
