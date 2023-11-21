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
    MEAN_MEAN_A0_CONVERSATION_MODEL,
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
    FIX_SAMPLED_SUBJECT_SEQUENCE_CONVERSATIONAL_MODEL,
    NUM_TIME_STEPS
)
from coordination.common.types import TensorTypes


class ConversationModel(Model):
    """
    This class represents a conversational model where subjects are talking to each other and
    their voice intensities evolve in an oscillatory fashion and pair-wisely blended by
    coordination.

    The variables in the latent component and observations are 2 dimensional. The first dimension
    contains the voice intensity and the second the speed. Speed evolves with the laws that govern
    harmonic oscillators but it is not blended by coordination as the voice intensity is.
    """

    def __init__(
            self,
            pymc_model: pm.Model,
            num_subjects: int = NUM_SUBJECTS,
            num_time_steps_in_coordination_scale: int = NUM_TIME_STEPS,
            squared_angular_frequency: np.ndarray = ANGULAR_FREQUENCIES_CONVERSATION_MODEL,
            dampening_coefficient: np.ndarray = DAMPENING_COEFFICIENTS_CONVERSATION_MODEL,
            dt: float = DT_CONVERSATION_MODEL,
            sd_mean_uc0: float = SD_MEAN_UC0,
            sd_sd_uc: float = SD_SD_UC,
            mean_mean_a0: np.ndarray = MEAN_MEAN_A0_CONVERSATION_MODEL,
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
            fix_sampled_subject_sequence: bool = FIX_SAMPLED_SUBJECT_SEQUENCE_CONVERSATIONAL_MODEL,
            coordination_samples: Optional[ModuleSamples] = None):
        """
        Creates a conversational model.

        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects in the conversation.
        @param num_time_steps_in_coordination_scale: size of the coordination series.
        @param squared_angular_frequency: squared angular frequency of the oscillatory pattern.
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
        @param sampling_time_scale_density: a number between 0 and 1 indicating percentage of
            time steps someone talks.
        @param allow_sampled_subject_repetition: whether a subject can speak in subsequently
            before others talk.
        @param fix_sampled_subject_sequence: whether the sequence of subjects is fixed
            (0,1,2,...,0,1,2...) or randomized.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param coordination_samples: coordination samples. If not provided, coordination samples
            will be draw in a call to draw_samples.
        """

        coordination = SigmoidGaussianCoordination(
            pymc_model=pymc_model,
            sd_mean_uc0=sd_mean_uc0,
            sd_sd_uc=sd_sd_uc,
            num_time_steps=num_time_steps_in_coordination_scale
        )

        # Save a direct reference to state_space and observation for easy access in the parameter
        # setting functions in this class.
        self.state_space = SerialMassSpringDamperLatentComponent(
            uuid="state_space",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            # angular_frequency^2 = spring_constant / mass
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

        self.observation = SerialGaussianObservation(
            uuid="observation",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
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
                             mean_uc0: float = MEAN_UC0,
                             sd_uc: float = SD_UC,
                             initial_state: np.ndarray = INITIAL_STATE_CONVERSATION_MODEL,
                             sd_a: np.ndarray = SD_A_CONVERSATION_MODEL,
                             sd_o: np.ndarray = SD_O_CONVERSATION_MODEL):
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

    def prepare_for_inference(self,
                              num_time_steps_in_coordination_scale: int,
                              time_steps_in_coordination_scale: np.array,
                              subject_indices: np.ndarray,
                              prev_time_same_subject: np.ndarray,
                              prev_time_diff_subject: np.ndarray,
                              observed_values: TensorTypes):
        """
        Sets metadata required for inference.

        @param num_time_steps_in_coordination_scale: size of the coordination series.
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
        @param observed_values: observations for the latent component random variable.
        """

        self.coordination.num_time_steps = num_time_steps_in_coordination_scale
        self.state_space.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.state_space.subject_indices = subject_indices
        self.state_space.prev_time_same_subject = prev_time_same_subject
        self.state_space.prev_time_diff_subject = prev_time_diff_subject
        self.observation.observed_values = observed_values
        self.observation.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.observation.subject_indices = subject_indices

        self.create_random_variables()
