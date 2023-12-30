from typing import List, Optional

import numpy as np

from coordination.common.constants import (DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_NUM_TIME_STEPS)
from coordination.model.config_bundle.bundle import ModelConfigBundle
from coordination.model.synthetic.constants import ConversationConstants
from coordination.module.module import ModuleSamples


class ConversationConfigBundle(ModelConfigBundle):
    """
    Container for the different parameters of the conversation model.
    """

    def __init__(
            self,
            num_subjects: int = ConversationConstants.NUM_SUBJECTS,
            num_time_steps_in_coordination_scale: int = DEFAULT_NUM_TIME_STEPS,
            mean_mean_uc0: float = ConversationConstants.MEAN_MEAN_UC0,
            sd_mean_uc0: float = ConversationConstants.SD_MEAN_UC0,
            sd_sd_uc: float = ConversationConstants.SD_SD_UC,
            mean_mean_a0: np.ndarray = ConversationConstants.MEAN_MEAN_A0,
            sd_mean_a0: np.ndarray = ConversationConstants.SD_MEAN_A0,
            sd_sd_a: np.ndarray = ConversationConstants.SD_SD_A,
            sd_sd_o: np.ndarray = ConversationConstants.SD_SD_O,
            share_mean_a0_across_subjects: bool = (
                    ConversationConstants.SHARE_MEAN_A0_ACROSS_SUBJECT),
            share_mean_a0_across_dimensions: bool = (
                    ConversationConstants.SHARE_MEAN_A0_ACROSS_DIMENSIONS),
            share_sd_a_across_subjects: bool = ConversationConstants.SHARE_SD_A_ACROSS_SUBJECTS,
            share_sd_a_across_dimensions: bool = (
                    ConversationConstants.SHARE_SD_A_ACROSS_DIMENSIONS),
            share_sd_o_across_subjects: bool = ConversationConstants.SHARE_SD_O_ACROSS_SUBJECTS,
            share_sd_o_across_dimensions: bool = (
                    ConversationConstants.SHARE_SD_O_ACROSS_DIMENSIONS),
            sampling_time_scale_density: float = ConversationConstants.SAMPLING_TIME_SCALE_DENSITY,
            allow_sampled_subject_repetition: bool = (
                    ConversationConstants.ALLOW_SAMPLED_SUBJECT_REPETITION
            ),
            fix_sampled_subject_sequence: bool = ConversationConstants.FIX_SAMPLED_SUBJECT_SEQUENCE,
            mean_uc0: float = ConversationConstants.MEAN_UC0,
            sd_uc: float = ConversationConstants.SD_UC,
            mean_a0: np.ndarray = ConversationConstants.MEAN_A0,
            sd_a: np.ndarray = ConversationConstants.SD_A,
            sd_o: np.ndarray = ConversationConstants.SD_O,
            time_steps_in_coordination_scale: Optional[np.array] = None,
            subject_indices: Optional[np.array] = None,
            prev_time_same_subject: Optional[np.array] = None,
            prev_time_diff_subject: Optional[np.array] = None,
            observed_values: Optional[np.array] = None,
            coordination_samples: Optional[ModuleSamples] = None,
            num_hidden_layers: int = ConversationConstants.NUM_HIDDEN_LAYERS,
            hidden_dimension_size: int = ConversationConstants.HIDDEN_DIMENSION_SIZE,
            activation: str = ConversationConstants.ACTIVATION,
            weights: List[np.ndarray] = ConversationConstants.WEIGHTS,
            mean_w0: float = ConversationConstants.MEAN_W0,
            sd_w0: float = ConversationConstants.SD_W0,
            observation_normalization: bool = (
            ConversationConstants.OBSERVATION_NORMALIZATION),
            squared_angular_frequency=ConversationConstants.SQUARED_ANGULAR_FREQUENCY,
            dampening_coefficient=ConversationConstants.DAMPENING_COEFFICIENT,
            time_step_size_in_seconds=ConversationConstants.TIME_STEP_SIZE_IN_SECONDS
    ):
        """
        Creates a config bundle for the vocalic model.

        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects in the conversation.
        @param num_time_steps_in_coordination_scale: size of the coordination series.
        @param mean_mean_uc0: mean of the hyper-prior of mu_uc0.
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
        @param mean_uc0: mean of the initial value of the unbounded coordination.
        @param sd_uc: standard deviation of the initial value and random Gaussian walk of the
            unbounded coordination.
        @param initial_state: value of the latent component at t = 0.
        @param sd_a: noise in the Gaussian random walk in the state space.
        @param sd_o: noise in the observation.
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
        @param observed_values: observations vocalic feature values.
        @param coordination_samples: coordination samples. If not provided, coordination samples
            will be draw in a call to draw_samples.
        @param num_hidden_layers: number of hidden layers in the transformation (MLP) from the
            state space to the observed vocalics.
        @param hidden_dimension_size: dimension of each hidden layer in the transformation.
        @param activation: activation used in the transformation.
        @param weights: a list of weights used in the transformation. One matrix per layer.
        @param mean_w0: mean of the weights in the transformation.
        @param sd_w0: standard deviation of the weights in the transformation.
        @param observation_normalization: type of normalization to apply to observed values if
            desired. Valid normalization values are: norm_per_feature or
            norm_per_subject_and_feature.
        @param squared_angular_frequency: squared angular frequency of the oscillator.
        @param dampening_coefficient: dampening coefficient of the oscillator.
        @param time_step_size_in_seconds: the size of each time step.
        """

        self.num_subjects: int = num_subjects
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.mean_mean_uc0 = mean_mean_uc0
        self.sd_mean_uc0 = sd_mean_uc0
        self.sd_sd_uc = sd_sd_uc
        self.mean_mean_a0 = mean_mean_a0
        self.sd_mean_a0 = sd_mean_a0
        self.sd_sd_a = sd_sd_a
        self.sd_sd_o = sd_sd_o
        self.share_mean_a0_across_subjects = share_mean_a0_across_subjects
        self.share_mean_a0_across_dimensions = share_mean_a0_across_dimensions
        self.share_sd_a_across_subjects = share_sd_a_across_subjects
        self.share_sd_a_across_dimensions = share_sd_a_across_dimensions
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_dimensions = share_sd_o_across_dimensions
        self.sampling_time_scale_density = sampling_time_scale_density
        self.allow_sampled_subject_repetition = allow_sampled_subject_repetition
        self.fix_sampled_subject_sequence = fix_sampled_subject_sequence
        self.mean_uc0 = mean_uc0
        self.sd_uc = sd_uc
        self.mean_a0 = mean_a0
        self.sd_a = sd_a
        self.sd_o = sd_o
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.subject_indices = subject_indices
        self.prev_time_same_subject = prev_time_same_subject
        self.prev_time_diff_subject = prev_time_diff_subject
        self.observed_values = observed_values
        self.coordination_samples = coordination_samples
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dimension_size = hidden_dimension_size
        self.activation = activation
        self.weights = weights
        self.mean_w0 = mean_w0
        self.sd_w0 = sd_w0
        self.observation_normalization = observation_normalization
        self.squared_angular_frequency = squared_angular_frequency
        self.dampening_coefficient = dampening_coefficient
        self.time_step_size_in_seconds = time_step_size_in_seconds
