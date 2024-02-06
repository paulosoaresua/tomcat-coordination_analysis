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
    asymmetric_coordination: bool = False

    num_fnirs_channels: int = 20
    fnirs_channel_names: List[str] = field(
        default_factory=lambda: [
            "s1_d1",
            "s1_d2",
            "s2_d1",
            "s2_d3",
            "s3_d1",
            "s3_d3",
            "s3_d4",
            "s4_d2",
            "s4_d4",
            "s4_d5",
            "s5_d3",
            "s5_d4",
            "s5_d6",
            "s6_d4",
            "s6_d6",
            "s6_d7",
            "s7_d5",
            "s7_d7",
            "s8_d6",
            "s8_d7",
        ]
    )

    fnirs_share_mean_a0_across_subjects: bool = True
    fnirs_share_mean_a0_across_dimensions: bool = True
    fnirs_share_sd_a_across_subjects: bool = True
    fnirs_share_sd_a_across_dimensions: bool = True
    fnirs_share_sd_o_across_subjects: bool = True
    fnirs_share_sd_o_across_dimensions: bool = True

    # Metadata parameters. These must be filled before inference.
    fnirs_time_steps_in_coordination_scale: np.ndarray = None
    fnirs_observed_values: np.ndarray = None

    # To allow splitting features into different groups
    # If provided, it must be list of a dictionaries in the following format:
    # {
    # "name": "name of the group"
    # "features": ["a list fnirs channels to include in this group"]
    # observations. If not given, it will be fit to the data.
    # Example:
    # fnirs_groups = [
    #     {
    #         "name": "s1_d1_s1_d2",
    #         "features": ["s1_d1", "s1_d2"],
    #         "symmetric_coordination": True,
    #     },
    #     {
    #         "name": "s3_d1_s3_d2",
    #         "features": ["s3_d1", "s3_d2"],
    #         "symmetric_coordination": True,
    #     }
    # ]
    fnirs_groups: List[Dict[str, Any]] = None
