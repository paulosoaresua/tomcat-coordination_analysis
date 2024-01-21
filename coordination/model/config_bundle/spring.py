from dataclasses import dataclass, field
from typing import List

import numpy as np

from coordination.model.config_bundle.bundle import ModelConfigBundle


@dataclass
class SpringConfigBundle(ModelConfigBundle):
    """
    Container for the different parameters of the spring model.
    """

    num_springs: int = 2
    num_time_steps_in_coordination_scale: int = 100
    observation_normalization: str = None
    spring_constant: np.ndarray = np.array([1, 0.5])
    mass: int = 1
    dampening_coefficient: float = 0.0
    time_step_size_in_seconds: float = 0.2#1.0
    blend_position: bool = True
    blend_speed: bool = False
    observation_dim_size: int = 2  # both position and speed are observed.

    # Hyper priors
    mean_mean_uc0: float = 0.0
    sd_mean_uc0: float = 1.0
    sd_sd_uc: float = 1.0
    mean_mean_a0: float = 0.0
    sd_mean_a0: float = 5.0  # max of mean_a0
    sd_sd_a: float = 1.0
    sd_sd_o: float = 1.0

    share_mean_a0_across_springs: bool = False
    share_mean_a0_across_dimensions: bool = False
    share_sd_a_across_springs: bool = True
    share_sd_a_across_dimensions: bool = True
    share_sd_o_across_springs: bool = True
    share_sd_o_across_dimensions: bool = True

    sampling_relative_frequency: float = 1.0

    # For sampling. These will be cleared and estimated during inference.
    mean_uc0: float = 0.0
    sd_uc: float = 1.0
    # Each subject starts with the same voice intensity and 0 speed.
    mean_a0: np.ndarray = np.array([[1, 0], [1, 0]])
    sd_a: float = 0.1
    sd_o: float = 0.1
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
    num_hidden_layers: int = 0
    hidden_dimension_size: int = 0
    activation: str = "linear"
    # We observe the position and disregard speed
    weights: List[np.ndarray] = field(default_factory=lambda: [np.array([[1], [0]])])
    mean_w0: float = 0.0
    sd_w0: float = 1.0
