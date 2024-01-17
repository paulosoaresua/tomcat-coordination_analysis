from dataclasses import dataclass, field
from typing import List

import numpy as np

from coordination.common.normalization import NORMALIZATION_PER_FEATURE
from coordination.model.config_bundle.bundle import ModelConfigBundle


@dataclass
class FNIRSConfigBundle(ModelConfigBundle):
    """
    Container for the different parameters of the fnirs model.
    """

    num_subjects: int = 3
    observation_normalization: str = NORMALIZATION_PER_FEATURE
    num_channels: int = 2
    channel_names: List[str] = field(default_factory=lambda: ["s1_d1", "s1_d2"])

    # Coordination
    num_time_steps_in_coordination_scale: int = 100
    constant_coordination: bool = False

    # Parameters if coordination is constant
    alpha: float = 1
    beta: float = 1

    # Parameters if coordination is variable
    mean_mean_uc0: float = 0.0
    sd_mean_uc0: float = 1.0
    sd_sd_uc: float = 1.0
    mean_uc0: float = None
    sd_uc: float = None

    # Hyper priors
    mean_mean_a0: float = 0.0
    sd_mean_a0: float = 1.0
    sd_sd_a: float = 1.0
    sd_sd_o: float = 1.0

    share_mean_a0_across_subjects: bool = False
    share_mean_a0_across_dimensions: bool = False
    share_sd_a_across_subjects: bool = True
    share_sd_a_across_dimensions: bool = True
    share_sd_o_across_subjects: bool = True
    share_sd_o_across_dimensions: bool = True

    sampling_relative_frequency: float = 1.0

    # For sampling. Defaults to None for inference.
    mean_uc0: float = None
    sd_uc: float = None
    mean_a0: float = None
    sd_a: float = None
    sd_o: float = None
    # Fixed coordination series for sampling.
    coordination_samples: np.ndarray = None

    # Evidence and metadata filled before inference.
    time_steps_in_coordination_scale: np.ndarray = None
    observed_values: np.ndarray = None

    # To transform a high-dimension state space to a lower dimension observation in case we
    # want to observe position only.
    num_hidden_layers: int = 0
    hidden_dimension_size: int = 0
    activation: str = "linear"
    # Only position is used. # From position to channels
    weights: List[np.ndarray] = field(default_factory=lambda: [np.ones((1, 2))])
    mean_w0: float = 0.0
    sd_w0: float = 1.0
