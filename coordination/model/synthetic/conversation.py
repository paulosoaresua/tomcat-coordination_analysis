from typing import Optional

import numpy as np
import pymc as pm

from coordination.module.module import ModuleSamples
from coordination.model.model import Model
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.component_group import ComponentGroup
from coordination.module.latent_component.serial_mass_spring_damper_latent_component import \
    SerialMassSpringDamperLatentComponent
from coordination.module.observation.serial_gaussian_observation import SerialGaussianObservation
from coordination.model.synthetic.constants import (
    NUM_SUBJECTS,
    SUBJECT_NAMES_CONVERSATION_MODEL,
    ANGULAR_FREQUENCIES_CONVERSATION_MODEL,
    DAMPENING_COEFFICIENTS_CONVERSATION_MODEL,
    DT_CONVERSATION_MODEL,
    INITIAL_STATE_CONVERSATION_MODEL,
    SD_MEAN_UC0,
    SD_SD_UC,
    MEAN_MEAN_A0,
    SD_MEAN_A0_CONVERSATION_MODEL,
    SD_SD_A,
    SD_SD_O,
    SHARE_MEAN_A0_ACROSS_SUBJECT,
    SHARE_MEAN_A0_ACROSS_DIMENSIONS,
    SHARE_SD_ACROSS_SUBJECTS,
    SHARE_SD_ACROSS_DIMENSIONS,
    SD_A_CONVERSATION_MODEL,
    SD_O_CONVERSATION_MODEL,
    SAMPLING_TIME_SCALE_DENSITY_CONVERSATIONAL_MODEL,
    ALLOW_SAMPLED_SUBJECT_REPETITION_CONVERSATIONAL_MODEL,
    FIX_SAMPLED_SUBJECT_SEQUENCE_CONVERSATIONAL_MODEL
)


class ConversationModel(Model):

    def __init__(
            self,
            pymc_model: pm.Model,
            num_subjects: int = NUM_SUBJECTS,
            squared_angular_frequency: np.ndarray = ANGULAR_FREQUENCIES_CONVERSATION_MODEL,
            dampening_coefficient: np.ndarray = DAMPENING_COEFFICIENTS_CONVERSATION_MODEL,
            dt: float = DT_CONVERSATION_MODEL,
            sd_mean_uc0: float = SD_MEAN_UC0,
            sd_sd_uc: float = SD_SD_UC,
            mean_mean_a0: np.ndarray = MEAN_MEAN_A0,
            sd_mean_a0: np.ndarray = SD_MEAN_A0_CONVERSATION_MODEL,
            sd_sd_a: np.ndarray = SD_SD_A,
            sd_sd_o: np.ndarray = SD_SD_O,
            share_mean_a0_across_subjects: bool = SHARE_MEAN_A0_ACROSS_SUBJECT,
            share_mean_a0_across_dimensions: bool = SHARE_MEAN_A0_ACROSS_DIMENSIONS,
            share_sd_a_across_subjects: bool = SHARE_SD_ACROSS_SUBJECTS,
            share_sd_a_across_dimensions: bool = SHARE_SD_ACROSS_DIMENSIONS,
            share_sd_o_across_subjects: bool = SHARE_SD_ACROSS_SUBJECTS,
            share_sd_o_across_dimensions: bool = SHARE_SD_ACROSS_DIMENSIONS,
            sampling_time_scale_density: float = SAMPLING_TIME_SCALE_DENSITY_CONVERSATIONAL_MODEL,
            allow_sampled_subject_repetition: bool = ALLOW_SAMPLED_SUBJECT_REPETITION_CONVERSATIONAL_MODEL,
            fix_sampled_subject_sequence: bool = SAMPLING_TIME_SCALE_DENSITY_CONVERSATIONAL_MODEL,
            coordination_samples: Optional[ModuleSamples] = None):
        coordination = SigmoidGaussianCoordination(
            pymc_model=pymc_model,
            sd_mean_uc0=sd_mean_uc0,
            sd_sd_uc=sd_sd_uc
        )

        # angular_frequency^2 = spring_constant / mass
        state_space = SerialMassSpringDamperLatentComponent(
            uuid="state_space",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            spring_constant=squared_angular_frequency,
            mass=np.ones(num_subjects),
            dampening_coefficient=dampening_coefficient,
            dt=dt,
            mean_mean_a0=mean_mean_a0,
            sd_mean_a0=sd_mean_a0,
            sd_sd_a=sd_sd_a,
            share_mean_a0_across_subjects=share_mean_a0_across_subjects,
            share_sd_a_across_subjects=share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=share_sd_a_across_dimensions,
            sampling_time_scale_density=sampling_time_scale_density,
            allow_sampled_subject_repetition=allow_sampled_subject_repetition,
            fix_sampled_subject_sequence=fix_sampled_subject_sequence
        )

        observation = SerialGaussianObservation(
            uuid="observation",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=state_space.dimension_size,
            sd_sd_o=sd_sd_o,
            share_sd_o_across_subjects=share_sd_o_across_subjects,
            share_sd_o_across_dimensions=share_sd_o_across_dimensions
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=pymc_model,
            latent_component=state_space,
            observations=[observation]
        )

        super().__init__(
            name="conversation_model",
            pymc_model=pymc_model,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=coordination_samples
        )

    def set_parameter_values(self,
                             initial_state: np.ndarray = INITIAL_STATE_CONVERSATION_MODEL,
                             sd_a: np.ndarray = SD_A_CONVERSATION_MODEL,
                             sd_o: np.ndarray = SD_O_CONVERSATION_MODEL):
        """
        Sets parameter values for sampling.

        @param initial_state: value of the latent component at t = 0.
        @param sd_a: noise in the Gaussian random walk in the state space.
        @param sd_o: noise in the observation.
        """

        self.component_groups[0].latent_component.parameters.mean_a0.value = initial_state
        self.component_groups[0].latent_component.parameters.sd_a.value = sd_a
        self.component_groups[0].observations[0].parameters.sd_o.value = sd_o
