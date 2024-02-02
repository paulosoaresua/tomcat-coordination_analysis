from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np

from coordination.common.scaler import NORMALIZATION_PER_FEATURE
from coordination.model.config_bundle.bundle import ModelConfigBundle


@dataclass
class BrainBundle(ModelConfigBundle):
    """
    Container for the different parameters of the brain model.
    """

    num_subjects: int = 3
    num_time_steps_in_coordination_scale: int = 100

    # Used for both sampling and inference. It is capped to the number of time steps in
    # coordination scale.
    num_time_steps_to_fit: int = None

    # Fixed value of coordination during inference
    observed_coordination_for_inference: Union[float, np.ndarray] = None

    # Initial sampled values of coordination. In a call to draw samples, the model will start by
    # using the samples given here and move forward sampling in time until the number of time
    # steps desired have been reached.
    initial_coordination_samples: np.ndarray = None
    initial_fnirs_state_space_samples: np.ndarray = None

    # Whether to use a constant model of coordination
    constant_coordination: bool = False

    # Parameters for inference
    mean_mean_uc0: float = 0.0  # Variable coordination
    sd_mean_uc0: float = 1.0
    sd_sd_uc: float = 1.0
    alpha_c: float = 1.0  # Constant coordination
    beta_c: float = 1.0
    # -----------------------
    # State space and observation
    fnirs_mean_mean_a0: float = 0.0
    fnirs_sd_mean_a0: float = 1.0
    fnirs_sd_sd_a: float = 1.0
    fnirs_sd_sd_o: float = 1.0

    # Given parameter values. Required for sampling, not for inference.
    mean_uc0: float = None
    sd_uc: float = None
    fnirs_mean_a0: float = None
    fnirs_sd_a: float = None
    fnirs_sd_o: float = None

    # Sampling settings
    sampling_relative_frequency: float = 1.0

    # Inference settings
    observation_normalization: str = NORMALIZATION_PER_FEATURE

    # Modules settings
    include_ekg: bool = False

    num_fnirs_channels: int = 20
    fnirs_channel_names: List[str] = field(default_factory=lambda: ["s1_d1", "s1_d2"])

    share_mean_a0_across_subjects: bool = True
    share_mean_a0_across_dimensions: bool = True
    share_sd_a_across_subjects: bool = True
    share_sd_a_across_dimensions: bool = True
    share_sd_o_across_subjects: bool = True
    share_sd_o_across_dimensions: bool = True

    # Metadata parameters. These must be filled before inference.
    time_steps_in_coordination_scale: np.ndarray = None
    subject_indices: np.ndarray = None
    prev_time_same_subject: np.ndarray = None
    prev_time_diff_subject: np.ndarray = None
    observed_values: np.ndarray = None

    # Extra parameters for the state space 2d case:
    num_hidden_layers: int = 0
    hidden_dimension_size: int = 0
    activation: str = "linear"
    # Only position is used. # From position to 4 vocalic features
    weights: List[np.ndarray] = field(default_factory=lambda: [np.ones((1, 4))])
    mean_w0: float = 0.0
    sd_w0: float = 1.0

    # To allow splitting features into different groups
    # If provided, it must be list of a dictionaries in the following format:
    # {
    # "name": "name of the group"
    # "features": ["a list vocalic features to include in this group"]
    # "weights": None or fixed weights to transform the latent component of the group to
    # observations. If not given, it will be fit to the data.
    # Example:
    # vocalic_groups = [
    #     {
    #         "name": "pitch_intensity",
    #         "features": ["pitch", "intensity"],
    #         "weights": [np.ones((1, 2))]
    #     },
    #     {
    #         "name": "jitter_shimmer",
    #         "features": ["jitter", "shimmer"],
    #         "weights": [np.ones((1, 2))]
    #     }
    # ]
    vocalic_groups: List[Dict[str, Any]] = None

    # Parameters for the semantic component
    sd_sd_s: float = 5.0
    sd_s: float = None

    semantic_link_time_steps_in_coordination_scale: np.ndarray = None
