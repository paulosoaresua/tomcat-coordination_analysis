from typing import List, Optional

import numpy as np
import pymc as pm

from coordination.module.module import ModuleSamples
from coordination.model.model import Model
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.component_group import ComponentGroup
from coordination.module.latent_component.null_latent_component import NullLatentComponent
from coordination.module.latent_component.serial_latent_component import SerialLatentComponent
from coordination.module.observation.serial_gaussian_observation import SerialGaussianObservation
from coordination.module.observation.spike_observation import SpikeObservation
from coordination.model.real.constants import (VocalicConstants,
                                               SemanticLinkConstants,
                                               DEFAULT_NUM_TIME_STEPS,
                                               DEFAULT_NUM_SUBJECTS)
from coordination.common.types import TensorTypes


class VocalicSemanticLinkModel(Model):
    """
    This class represents a vocalic + semantic link model where subjects are talking to each other
    and their speech vocalics and semantic links between their utterances are observed as they
    finish talking.
    """

    def __init__(
            self,
            pymc_model: pm.Model,
            num_subjects: int = DEFAULT_NUM_SUBJECTS,
            num_time_steps_in_coordination_scale: int = DEFAULT_NUM_TIME_STEPS,
            state_space_dimension_size: int = VocalicConstants.STATE_SPACE_DIM_SIZE,
            state_space_dimension_names: List[str] = VocalicConstants.STATE_SPACE_DIM_NAMES,
            self_dependent: bool = VocalicConstants.SELF_DEPENDENT_STATE_SPACE,
            num_vocalic_features: int = VocalicConstants.NUM_VOCALIC_FEATURES,
            vocalic_feature_names: List[str] = VocalicConstants.VOCALIC_FEATURE_NAMES,
            sd_mean_uc0: float = VocalicConstants.SD_MEAN_UC0,
            sd_sd_uc: float = VocalicConstants.SD_SD_UC,
            mean_mean_a0: np.ndarray = VocalicConstants.MEAN_MEAN_A0,
            sd_mean_a0: np.ndarray = VocalicConstants.SD_MEAN_A0,
            sd_sd_a: np.ndarray = VocalicConstants.SD_SD_A,
            sd_sd_o: np.ndarray = VocalicConstants.SD_SD_O,
            share_mean_a0_across_subjects: bool = VocalicConstants.SHARE_MEAN_A0_ACROSS_SUBJECT,
            share_mean_a0_across_dimensions: bool = VocalicConstants.SHARE_MEAN_A0_ACROSS_DIMENSIONS,
            share_sd_a_across_subjects: bool = VocalicConstants.SHARE_SD_A_ACROSS_SUBJECTS,
            share_sd_a_across_dimensions: bool = VocalicConstants.SHARE_SD_A_ACROSS_DIMENSIONS,
            share_sd_o_across_subjects: bool = VocalicConstants.SHARE_SD_O_ACROSS_SUBJECTS,
            share_sd_o_across_dimensions: bool = VocalicConstants.SHARE_SD_O_ACROSS_DIMENSIONS,
            sampling_time_scale_density: float = VocalicConstants.SAMPLING_TIME_SCALE_DENSITY,
            allow_sampled_subject_repetition: bool = VocalicConstants.ALLOW_SAMPLED_SUBJECT_REPETITION,
            fix_sampled_subject_sequence: bool = VocalicConstants.FIX_SAMPLED_SUBJECT_SEQUENCE,
            coordination_samples: Optional[ModuleSamples] = None):
        """
        Creates a vocalic model.

        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects in the conversation.
        @param num_time_steps_in_coordination_scale: size of the coordination series.
        @param state_space_dimension_size: dimension size of the variables in the latent component.
        @param state_space_dimension_names: names of the dimensions in the state space.
        @param self_dependent: whether the latent variables in the component are tied to the past
            values from the same subject. If False, coordination will blend the previous latent
            value of a different subject with the value of the component at time t = 0 for the
            current subject (the latent component's prior for that subject).
        @param num_vocalic_features: number of observed vocalic features. Dimension of the
            observed speech vocalics.
        @param vocalic_feature_names: names of the observed vocalic features.
        @param sd_mean_uc0: std of the hyper-prior of mu_uc0.
        @param sd_sd_uc: std of the hyper-prior of sigma_uc (std of the Gaussian random walk of
            the unbounded coordination).
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
        self.state_space = SerialLatentComponent(
            uuid="state_space",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=state_space_dimension_size,
            self_dependent=self_dependent,
            mean_mean_a0=mean_mean_a0,
            sd_mean_a0=sd_mean_a0,
            sd_sd_a=sd_sd_a,
            share_mean_a0_across_subjects=share_mean_a0_across_subjects,
            share_sd_a_across_subjects=share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=share_sd_a_across_dimensions,
            dimension_names=state_space_dimension_names,
            sampling_time_scale_density=sampling_time_scale_density,
            allow_sampled_subject_repetition=allow_sampled_subject_repetition,
            fix_sampled_subject_sequence=fix_sampled_subject_sequence
        )

        self.observed_vocalics = SerialGaussianObservation(
            uuid="speech_vocalics",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=num_vocalic_features,
            dimension_names=vocalic_feature_names,
            sd_sd_o=sd_sd_o,
            share_sd_o_across_subjects=share_sd_o_across_subjects,
            share_sd_o_across_dimensions=share_sd_o_across_dimensions
        )

        self.observed_semantic_links = SpikeObservation(
            uuid="semantic_link",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            a_p=SemanticLinkConstants.A_P,
            b_p=SemanticLinkConstants.B_P,
            dimension_name="linked",
            sampling_time_scale_density=sampling_time_scale_density,
        )

        vocalic_group = ComponentGroup(
            uuid="vocalic_group",
            pymc_model=pymc_model,
            latent_component=self.state_space,
            observations=[self.observed_vocalics]
        )

        semantic_link_group = ComponentGroup(
            uuid="semantic_link_group",
            pymc_model=pymc_model,
            latent_component=NullLatentComponent(),
            observations=[self.observed_semantic_links]
        )

        super().__init__(
            name="vocalic_semantic_link_model",
            pymc_model=pymc_model,
            coordination=coordination,
            component_groups=[vocalic_group, semantic_link_group],
            coordination_samples=coordination_samples
        )

    def prepare_for_sampling(self,
                             mean_uc0: float = VocalicConstants.MEAN_UC0,
                             sd_uc: float = VocalicConstants.SD_UC,
                             initial_vocalic_state: np.ndarray = VocalicConstants.MEAN_A0,
                             vocalic_sd_a: np.ndarray = VocalicConstants.SD_A,
                             vocalic_sd_o: np.ndarray = VocalicConstants.SD_O,
                             semantic_link_p: float = SemanticLinkConstants.P):
        """
        Sets parameter values for sampling.

        @param mean_uc0: mean of the initial value of the unbounded coordination.
        @param sd_uc: standard deviation of the initial value and random Gaussian walk of the
            unbounded coordination.
        @param initial_vocalic_state: value of the latent vocalics component at t = 0.
        @param vocalic_sd_a: noise in the Gaussian random walk in the vocalics state space.
        @param vocalic_sd_o: noise in the observed speech vocalics.
        @param semantic_link_p: by how much coordination should be weighed to estimate the
            probability of a semantic link to occur.
        """

        self.coordination.parameters.mean_uc0.value = np.ones(1) * mean_uc0
        self.coordination.parameters.sd_uc.value = np.ones(1) * sd_uc
        self.state_space.parameters.mean_a0.value = initial_vocalic_state
        self.state_space.parameters.sd_a.value = vocalic_sd_a
        self.observed_vocalics.parameters.sd_o.value = vocalic_sd_o
        self.observed_semantic_links.parameters.p.value = semantic_link_p

    def prepare_for_inference(self,
                              num_time_steps_in_coordination_scale: int,
                              vocalic_time_steps_in_coordination_scale: np.array,
                              semantic_link_time_steps_in_coordination_scale: np.array,
                              subject_indices: np.ndarray,
                              prev_time_same_subject: np.ndarray,
                              prev_time_diff_subject: np.ndarray,
                              observed_vocalic_values: TensorTypes):
        """
        Sets metadata required for inference.

        @param num_time_steps_in_coordination_scale: size of the coordination series.
        @param vocalic_time_steps_in_coordination_scale: time indexes when speech vocalics
            were observed.
        @param semantic_link_time_steps_in_coordination_scale: time indexes when semantic links
            were observed.
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
        @param observed_vocalic_values: observations for the latent component random variable.
        """

        self.coordination.num_time_steps = num_time_steps_in_coordination_scale
        self.state_space.time_steps_in_coordination_scale = \
            vocalic_time_steps_in_coordination_scale
        self.state_space.subject_indices = subject_indices
        self.state_space.prev_time_same_subject = prev_time_same_subject
        self.state_space.prev_time_diff_subject = prev_time_diff_subject
        self.observed_vocalics.observed_values = observed_vocalic_values
        self.observed_vocalics.time_steps_in_coordination_scale = \
            vocalic_time_steps_in_coordination_scale
        self.observed_vocalics.subject_indices = subject_indices
        self.observed_semantic_links.time_steps_in_coordination_scale = \
            semantic_link_time_steps_in_coordination_scale

        self.create_random_variables()
