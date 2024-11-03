from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np

from coordination.common.scaler import NORMALIZATION_PER_FEATURE
from coordination.model.config_bundle.bundle import ModelConfigBundle

from typing import Callable, List, Optional, Tuple, Union


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
    initial_gsr_state_space_samples: np.ndarray = None
    initial_vocalic_state_space_samples: np.ndarray = None

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

    gsr_mean_mean_a0: float = 0.0
    gsr_sd_mean_a0: float = 1.0
    gsr_sd_sd_a: float = 1.0
    gsr_sd_sd_o: float = 1.0

    vocalic_mean_mean_a0: float = 0.0
    vocalic_sd_mean_a0: float = 1.0
    vocalic_sd_sd_a: float = 1.0
    vocalic_sd_sd_o: float = 1.0

    semantic_link_sd_sd_s: float = 1.0

    # Given parameter values. Required for sampling, not for inference.
    mean_uc0: float = None
    sd_uc: float = None
    fnirs_mean_a0: float = None
    fnirs_sd_a: float = None
    fnirs_sd_o: float = None

    gsr_mean_a0: float = None
    gsr_sd_a: float = None
    gsr_sd_o: float = None

    vocalic_mean_a0: float = None
    vocalic_sd_a: float = None
    vocalic_sd_o: float = None

    semantic_link_sd_s: float = None

    # Sampling settings
    sampling_relative_frequency: float = 1.0
    vocalic_sampling_time_scale_density: float = 1.0
    vocalic_allow_sampled_subject_repetition: bool = False
    vocalic_fix_sampled_subject_sequence: bool = True

    # Inference settings
    observation_normalization: str = NORMALIZATION_PER_FEATURE

    # Modules settings
    include_ekg: bool = False
    include_gsr: bool = False
    include_vocalic: bool = False
    asymmetric_coordination: bool = False
    gsr_asymmetric_coordination: bool = False
    self_dependent_latent_states: bool = True
    use_1d_state_space: bool = False
    include_semantic: bool = False

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

    num_vocalic_features: int = 4
    vocalic_feature_names: List[str] = field(
        default_factory=lambda: ["pitch", "intensity", "jitter", "shimmer"]
    )

    fnirs_share_mean_a0_across_subjects: bool = True
    fnirs_share_mean_a0_across_dimensions: bool = True
    fnirs_share_sd_a_across_subjects: bool = True
    fnirs_share_sd_a_across_dimensions: bool = True
    fnirs_share_sd_o_across_subjects: bool = True
    fnirs_share_sd_o_across_dimensions: bool = True
    fnirs_share_fnirs_latent_state_across_subjects: bool = False

    gsr_share_mean_a0_across_subjects: bool = True
    gsr_share_mean_a0_across_dimensions: bool = True
    gsr_share_sd_a_across_subjects: bool = True
    gsr_share_sd_a_across_dimensions: bool = True
    gsr_share_sd_o_across_subjects: bool = True

    vocalic_share_mean_a0_across_subjects: bool = True
    vocalic_share_mean_a0_across_dimensions: bool = True
    vocalic_share_sd_a_across_subjects: bool = True
    vocalic_share_sd_a_across_dimensions: bool = True
    vocalic_share_sd_o_across_subjects: bool = True
    vocalic_share_sd_o_across_dimensions: bool = True

    # Metadata parameters. These must be filled before inference.
    fnirs_time_steps_in_coordination_scale: np.ndarray = None
    fnirs_observed_values: np.ndarray = None

    gsr_time_steps_in_coordination_scale: np.ndarray = None
    gsr_observed_values: np.ndarray = None

    vocalic_time_steps_in_coordination_scale: np.ndarray = None
    vocalic_subject_indices: np.ndarray = None
    vocalic_prev_time_same_subject: np.ndarray = None
    vocalic_prev_time_diff_subject: np.ndarray = None
    vocalic_observed_values: np.ndarray = None

    semantic_link_time_steps_in_coordination_scale: np.ndarray = None

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
    #         "asymmetric_coordination": False,
    #     },
    #     {
    #         "name": "s3_d1_s3_d2",
    #         "features": ["s3_d1", "s3_d2"],
    #         "asymmetric_coordination": False,
    #     }
    # ]
    fnirs_groups: List[Dict[str, Any]] = None

    # 1103
    common_cause: bool = False
    mean_mean_cc0: float = 0.0
    sd_mean_cc0: float = 1.0
    sd_sd_cc: float = 1.0
    mean_cc0: float = None  # mean_cc0 ~ N(mean_mean_cc0, sd_mean_cc0^2)
    sd_cc: float = None  # sd_cc ~ HN(sd_sd_cc)
    share_mean_cc0_across_dimensions: bool = False
    share_sd_cc_across_dimensions: bool = False
    initial_common_cause_samples: np.ndarray = None

    
    coordination_mode: str = "sigmoid" # Or "dirichlet"
    

    observed_individualism_for_inference: Optional[float] = None
    observed_common_cause_for_inference: Optional[float] = None
    constant_individualism: bool = False
    constant_common_cause: bool = False

    # # For independent model
    # observed_common_cause_for_inference = 0
    # observed_coordination_for_inference = 0
    # constant_coordination = True
    # constant_common_cause = True

    # # Coordination Model:
    # observed_common_cause_for_inference = 0
    # constant_common_cause = True
