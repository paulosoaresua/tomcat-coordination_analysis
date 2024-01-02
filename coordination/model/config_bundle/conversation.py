from dataclasses import dataclass

import numpy as np

from coordination.model.config_bundle.bundle import ModelConfigBundle


@dataclass
class ConversationConfigBundle(ModelConfigBundle):
    """
    Container for the different parameters of the conversation model.
    """

    num_subjects = 3
    num_time_steps_in_coordination_scale = 100
    observation_normalization = None
    squared_angular_frequency = np.array([1, 0.5, 0.1])
    dampening_coefficient = 0.0
    time_step_size_in_seconds = 0.2
    blend_position = True
    blend_speed = False
    observation_dim_size = 2  # both position and speed are observed.

    # Hyper priors
    mean_mean_uc0 = 0.0
    sd_mean_uc0 = 1.0
    sd_sd_uc = 1.0
    mean_mean_a0 = 0.0
    sd_mean_a0 = 1.0
    sd_sd_a = 1.0
    sd_sd_o = 1.0

    share_mean_a0_across_subjects = False
    share_mean_a0_across_dimensions = False
    share_sd_a_across_subjects = True
    share_sd_a_across_dimensions = True
    share_sd_o_across_subjects = True
    share_sd_o_across_dimensions = True

    sampling_time_scale_density = 1.0
    allow_sampled_subject_repetition = False
    fix_sampled_subject_sequence = True

    # For sampling. These will be cleared and estimated during inference.
    mean_uc0 = 0.0
    sd_uc = 1.0
    # Each subject starts with the same voice intensity and 0 speed.
    mean_a0 = np.array([[1, 0], [1, 0], [1, 0]])
    sd_a = 0.1
    sd_o = 0.01
    # Fixed coordination series for sampling.
    coordination_samples: np.ndarray = None

    # Evidence and metadata filled before inference.
    time_steps_in_coordination_scale: np.ndarray = None
    subject_indices: np.ndarray = None
    prev_time_same_subject: np.ndarray = None
    prev_time_diff_subject: np.ndarray = None
    observed_values: np.ndarray = None

    # To transform a high-dimension state space to a lower dimension observation in case we
    # want to observe position only.
    num_hidden_layers = 0
    hidden_dimension_size = 0
    activation = "linear"
    weights = [np.array([[1], [0]])]  # We observe the position and disregard speed
    mean_w0 = 0.0
    sd_w0 = 1.0
